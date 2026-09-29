"""Probe native ComfyUI XTTS CPU→GPU→CPU residency, without inference or node registration."""

import argparse
import sys
from pathlib import Path
from subprocess import run
from time import perf_counter

NODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(NODE_DIR / "vendor"))
sys.path.insert(0, str(NODE_DIR.parents[1]))  # The running container's ComfyUI checkout.

import torch
import TTS
from comfy import model_management as mm
from comfy.model_patcher import ModelPatcher
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts


def rss_mib() -> int:
    """Read this process's resident memory, excluding system page cache."""
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) // 1024
    raise RuntimeError("VmRSS missing from procfs")


def gpu_memory() -> str:
    """Read the actual GPU's used/free VRAM from nvidia-smi."""
    return run(
        ["nvidia-smi", "--query-gpu=memory.used,memory.free", "--format=csv,noheader,nounits"],
        text=True, capture_output=True, check=True,
    ).stdout.strip()


def device_counts(tensors) -> dict[str, int]:
    """Count registered parameters or buffers by their actual device."""
    counts: dict[str, int] = {}
    for tensor in tensors:
        key = str(tensor.device)
        counts[key] = counts.get(key, 0) + 1
    return counts


def unique_storage_bytes(model: Xtts) -> int:
    """Account for shared parameter/buffer storages only once in VRAM planning."""
    storages = {}
    for tensor in list(model.parameters()) + list(model.buffers()):
        storage = tensor.untyped_storage()
        storages[storage.data_ptr()] = storage.nbytes()
    return sum(storages.values())


def main() -> None:
    """Load one checkpoint and exercise only ComfyUI's native model residency API."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint_dir", type=Path)
    checkpoint_dir = parser.parse_args().checkpoint_dir.resolve(strict=True)
    for filename in ("config.json", "model.pth", "vocab.json", "speakers_xtts.pth"):
        if not (checkpoint_dir / filename).is_file():
            raise FileNotFoundError(checkpoint_dir / filename)
    if Path(TTS.__file__).resolve() != NODE_DIR / "vendor/TTS/__init__.py" or TTS.__version__ != "0.24.3":
        raise RuntimeError(f"Unexpected Coqui TTS source/version: {TTS.__file__}, {TTS.__version__}")
    device = mm.get_torch_device()
    if device.type != "cuda" or torch.cuda.get_device_name(device) != "Tesla V100-SXM2-16GB":
        raise RuntimeError(f"Expected V100 through ComfyUI, got {device}")
    if mm.loaded_models():
        raise RuntimeError("Expected an isolated empty model-management registry")
    print(f"runtime torch={torch.__version__} TTS={TTS.__version__} device={device}", flush=True)
    print(f"before nvidia_smi_used_free_mib={gpu_memory()} rss_mib={rss_mib()}", flush=True)

    config = XttsConfig()
    config.load_json(str(checkpoint_dir / "config.json"))
    model = Xtts.init_from_config(config)
    model.load_checkpoint(
        config,
        checkpoint_dir=str(checkpoint_dir),
        vocab_path=str(checkpoint_dir / "vocab.json"),
        use_deepspeed=False,
    )
    model_identity = id(model)
    if set(device_counts(model.parameters())) != {"cpu"}:
        raise RuntimeError(f"Expected CPU checkpoint before transfer: {device_counts(model.parameters())}")
    size_bytes = unique_storage_bytes(model)
    comfy_estimate = mm.module_size(model)
    patcher = ModelPatcher(model, load_device=device, offload_device=torch.device("cpu"), size=size_bytes)
    if patcher.model_size() != size_bytes:
        raise RuntimeError("Patcher size does not match unique XTTS storage size")
    print(
        f"checkpoint_cpu rss_mib={rss_mib()} size_bytes={size_bytes} "
        f"comfy_state_dict_size_bytes={comfy_estimate} "
        f"parameters={device_counts(model.parameters())} buffers={device_counts(model.buffers())}",
        flush=True,
    )
    torch.cuda.reset_peak_memory_stats(device)
    gpu_started = perf_counter()
    try:
        mm.load_models_gpu([patcher])
        torch.cuda.synchronize(device)
        to_gpu_s = perf_counter() - gpu_started
        gpu_counts = device_counts(model.parameters())
        if set(gpu_counts) != {str(device)}:
            raise RuntimeError(f"XTTS parameters not fully on V100: {gpu_counts}")
        if patcher not in mm.loaded_models():
            raise RuntimeError("XTTS not registered with ComfyUI model management")
        print(
            f"on_gpu seconds={to_gpu_s:.3f} parameters={gpu_counts} "
            f"buffers={device_counts(model.buffers())} "
            f"torch_allocated_mib={torch.cuda.memory_allocated(device) // (1024**2)} "
            f"torch_peak_allocated_mib={torch.cuda.max_memory_allocated(device) // (1024**2)} "
            f"torch_peak_reserved_mib={torch.cuda.max_memory_reserved(device) // (1024**2)} "
            f"nvidia_smi_used_free_mib={gpu_memory()} rss_mib={rss_mib()}",
            flush=True,
        )
    finally:
        # Native targeted offload; never unload unrelated models or touch CUDA manually.
        if patcher in mm.loaded_models():
            offload_started = perf_counter()
            mm.unload_model_and_clones(patcher, unload_additional_models=False)
            torch.cuda.synchronize(device)
            offload_s = perf_counter() - offload_started
            print(f"offload_api_seconds={offload_s:.3f}", flush=True)

    cpu_counts = device_counts(model.parameters())
    if set(cpu_counts) != {"cpu"} or patcher in mm.loaded_models():
        raise RuntimeError(f"ComfyUI did not fully offload XTTS: {cpu_counts}")
    if torch.cuda.memory_allocated(device) != 0:
        raise RuntimeError(f"GPU tensor memory remains: {torch.cuda.memory_allocated(device)} bytes")
    if patcher.model is not model or id(model) != model_identity:
        raise RuntimeError("Model identity changed; checkpoint may have been reloaded")
    print(
        f"BACK_ON_CPU parameters={cpu_counts} buffers={device_counts(model.buffers())} same_model_instance=True "
        f"torch_allocated_mib={torch.cuda.memory_allocated(device) // (1024**2)} "
        f"torch_reserved_mib={torch.cuda.memory_reserved(device) // (1024**2)} "
        f"nvidia_smi_used_free_mib={gpu_memory()} rss_mib={rss_mib()}",
        flush=True,
    )


if __name__ == "__main__":
    main()

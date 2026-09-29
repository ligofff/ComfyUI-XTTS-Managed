"""Two non-streaming banana XTTS inferences through native ComfyUI GPU residency."""

import argparse
from datetime import datetime, timezone
import gc
import math
from pathlib import Path
from subprocess import run
from threading import Event, Thread
from time import perf_counter

import numpy as np
import soundfile as sf

from test_xtts_lifecycle import (
    TTS, Xtts, XttsConfig, ModelPatcher, clear_gpt_prefix_cache, device_counts, mm, torch,
    unique_storage_bytes,
)

TEXTS = ("Привет, это проверка переноса голоса.", "Это повторная проверка голоса.")
# Live AllTalk /api/currentsettings: temperature_set, repetitionpenalty_set, generationspeed_set.
TEMPERATURE = 0.75
REPETITION_PENALTY = 10.0
SPEED = 1.0


def nvidia_used_mib() -> int:
    """Sample V100 device memory, including CUDA context and other processes."""
    output = run(
        ["nvidia-smi", "--id=0", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    return int(output)


def sample_vram(stop: Event, samples: list[int], errors: list[Exception]) -> None:
    """Observe nvidia-smi usage throughout both conditioning and synthesis."""
    while not stop.wait(0.25):
        try:
            samples.append(nvidia_used_mib())
        except Exception as error:
            errors.append(error)
            return


def cuda_inventory() -> list[str]:
    """Describe live CUDA tensors when targeted offload leaves allocations behind."""
    gc.collect()
    seen: set[int] = set()
    result = []
    for obj in gc.get_objects():
        try:
            if type(obj) is not torch.Tensor or obj.device.type != "cuda":
                continue
            ptr = obj.untyped_storage().data_ptr()
            if ptr in seen:
                continue
            seen.add(ptr)
            result.append(f"shape={tuple(obj.shape)} dtype={obj.dtype} bytes={obj.untyped_storage().nbytes()} type={type(obj).__name__}")
        except (ReferenceError, RuntimeError):
            continue
    return result


def synthesize(model: Xtts, reference: Path, text: str, destination: Path, device) -> tuple[float, float, float, int]:
    """Follow AllTalk's non-streaming conditioning → inference → WAV path once."""
    torch.cuda.synchronize(device)
    started = perf_counter()
    gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
        audio_path=[str(reference)],
        gpt_cond_len=model.config.gpt_cond_len,
        max_ref_length=model.config.max_ref_len,
        sound_norm_refs=model.config.sound_norm_refs,
    )
    torch.cuda.synchronize(device)
    conditioning_s = perf_counter() - started
    print(f"conditioning_seconds={conditioning_s:.3f}", flush=True)

    started = perf_counter()
    output = model.inference(
        text=text,
        language="ru",
        gpt_cond_latent=gpt_cond_latent,
        speaker_embedding=speaker_embedding,
        temperature=TEMPERATURE,
        length_penalty=float(model.config.length_penalty),
        repetition_penalty=REPETITION_PENALTY,
        top_k=int(model.config.top_k),
        top_p=float(model.config.top_p),
        speed=SPEED,
        enable_text_splitting=True,
    )
    if isinstance(output, dict) and "wav" in output:
        wav_source = output["wav"]
        if isinstance(wav_source, torch.Tensor) and wav_source.device.type == "cuda":
            waveform = wav_source.detach().cpu().numpy().astype(np.float32)
        else:
            waveform = np.asarray(wav_source, dtype=np.float32)
    else:
        waveform = np.asarray(output, dtype=np.float32)
    # The XTTS result is CPU-resident; GPT's cached prefix is a non-parameter
    # inference cache that the native module offload cannot see.
    gpt_prefix_bytes = clear_gpt_prefix_cache(model)
    del output, gpt_cond_latent, speaker_embedding
    torch.cuda.synchronize(device)
    synthesis_s = perf_counter() - started
    print(f"synthesis_seconds={synthesis_s:.3f} gpt_prefix_cache_bytes={gpt_prefix_bytes}", flush=True)

    # TorchCodec is incompatible with this container; write the XTTS waveform as 24 kHz PCM16.
    if destination.exists():
        raise FileExistsError(destination)
    if waveform.ndim != 1:
        raise RuntimeError(f"Expected mono XTTS waveform, got {waveform.shape}")
    sf.write(destination, waveform, 24000, subtype="PCM_16")
    info = sf.info(destination)
    samples, sample_rate = sf.read(destination, dtype="float32")
    if (info.samplerate != 24000 or info.channels != 1 or info.frames <= 0
            or not math.isfinite(info.duration) or info.duration <= 0
            or not np.isfinite(samples).all() or not np.any(samples != 0)
            or sample_rate != 24000):
        raise RuntimeError(f"Invalid generated WAV: {destination}, {info}")
    return conditioning_s, synthesis_s, info.duration, destination.stat().st_size


def main() -> None:
    """Run two separate conditionings/inferences around native ComfyUI offload/reload."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint_dir", type=Path)
    parser.add_argument("reference_wav", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    checkpoint_dir = args.checkpoint_dir.resolve(strict=True)
    reference = args.reference_wav.resolve(strict=True)
    output_dir = args.output_dir.resolve()
    if not sf.info(reference).duration > 0:
        raise ValueError(f"Empty reference WAV: {reference}")
    if TTS.__version__ != "0.24.3":
        raise RuntimeError(f"Expected vendored TTS 0.24.3, got {TTS.__version__}")
    device = mm.get_torch_device()
    if device.type != "cuda" or torch.cuda.get_device_name(device) != "Tesla V100-SXM2-16GB":
        raise RuntimeError(f"Expected V100 through ComfyUI, got {device}")
    if mm.loaded_models():
        raise RuntimeError("Expected empty separate-process model registry")

    config = XttsConfig()
    config.load_json(str(checkpoint_dir / "config.json"))
    model = Xtts.init_from_config(config)
    model.load_checkpoint(
        config, checkpoint_dir=str(checkpoint_dir),
        vocab_path=str(checkpoint_dir / "vocab.json"), use_deepspeed=False,
    )
    model_identity = id(model)
    if set(device_counts(model.parameters())) != {"cpu"}:
        raise RuntimeError("XTTS checkpoint did not load on CPU")
    size = unique_storage_bytes(model)
    patcher = ModelPatcher(model, load_device=device, offload_device=torch.device("cpu"), size=size)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    print(
        f"loaded_on_cpu model_id={model_identity} model_size_bytes={size} "
        f"conditioning=(gpt_cond_len={model.config.gpt_cond_len}, max_ref_len={model.config.max_ref_len}, "
        f"sound_norm_refs={model.config.sound_norm_refs}) "
        f"inference=(temperature={TEMPERATURE}, repetition_penalty={REPETITION_PENALTY}, "
        f"speed={SPEED}, length_penalty={model.config.length_penalty}, "
        f"top_k={model.config.top_k}, top_p={model.config.top_p})",
        flush=True,
    )

    for index, text in enumerate(TEXTS, start=1):
        if id(model) != model_identity or set(device_counts(model.parameters())) != {"cpu"}:
            raise RuntimeError("Model instance was recreated or did not return to CPU")
        destination = output_dir / f"banana_poc_{run_id}_pass{index}.wav"
        torch.cuda.reset_peak_memory_stats(device)
        before_gpu = nvidia_used_mib()
        print(f"pass={index} before_gpu nvidia_used_mib={before_gpu}", flush=True)
        started = perf_counter()
        mm.load_models_gpu([patcher])
        torch.cuda.synchronize(device)
        load_s = perf_counter() - started
        if set(device_counts(model.parameters())) != {str(device)} or patcher not in mm.loaded_models():
            raise RuntimeError(f"XTTS not managed on V100: {device_counts(model.parameters())}")
        print(f"pass={index} loaded_gpu_seconds={load_s:.3f}", flush=True)
        stop = Event()
        samples = [nvidia_used_mib()]
        errors: list[Exception] = []
        monitor = Thread(target=sample_vram, args=(stop, samples, errors), daemon=True)
        monitor.start()
        try:
            conditioning_s, synthesis_s, duration, file_bytes = synthesize(model, reference, text, destination, device)
            samples.append(nvidia_used_mib())
        finally:
            stop.set()
            monitor.join()
            # Free non-parameter inference caches so native offload can reclaim all VRAM.
            clear_gpt_prefix_cache(model)
            # Offload even on inference failure; the separate process must not keep VRAM.
            if patcher in mm.loaded_models():
                started = perf_counter()
                mm.unload_model_and_clones(patcher, unload_additional_models=False)
                torch.cuda.synchronize(device)
                offload_s = perf_counter() - started
                print(f"pass={index} offload_seconds={offload_s:.3f}", flush=True)
            cache_started = perf_counter()
            mm.soft_empty_cache()
            torch.cuda.synchronize(device)
            print(f"pass={index} soft_empty_cache_seconds={perf_counter() - cache_started:.3f}", flush=True)
        if errors:
            raise RuntimeError("nvidia-smi sampler failed") from errors[0]
        if set(device_counts(model.parameters())) != {"cpu"} or patcher in mm.loaded_models():
            raise RuntimeError(f"XTTS did not offload to CPU: {device_counts(model.parameters())}")
        allocated_after = torch.cuda.memory_allocated(device)
        runtime_inventory = cuda_inventory() if allocated_after else []
        if runtime_inventory:
            print(f"cuda_inventory={runtime_inventory}", flush=True)
            raise RuntimeError(f"XTTS left live CUDA tensors after offload: {runtime_inventory}")
        post_gpu = nvidia_used_mib()
        if post_gpu > before_gpu + 16:
            print(
                f"runtime_nvidia_delta_mib={post_gpu - before_gpu} "
                f"before={before_gpu} after={post_gpu} (no live CUDA tensors)",
                flush=True,
            )
        if allocated_after:
            print(
                f"non_model_cuda_allocator_bytes={allocated_after} "
                f"nvidia_post_offload_mib={post_gpu} (no live CUDA tensors)",
                flush=True,
            )
        print(
            f"PASS_OK pass={index} wav={destination} duration_s={duration:.3f} bytes={file_bytes} "
            f"conditioning_s={conditioning_s:.3f} synthesis_s={synthesis_s:.3f} "
            f"peak_torch_allocated_mib={torch.cuda.max_memory_allocated(device) // (1024**2)} "
            f"peak_torch_reserved_mib={torch.cuda.max_memory_reserved(device) // (1024**2)} "
            f"peak_nvidia_used_mib={max(samples)} samples={len(samples)} "
            f"post_nvidia_used_mib={nvidia_used_mib()} post_torch_allocated_bytes={allocated_after} "
            f"same_model_instance={id(model) == model_identity}",
            flush=True,
        )


if __name__ == "__main__":
    main()

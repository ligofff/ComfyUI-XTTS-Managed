"""ComfyUI-managed XTTS nodes for the vendored Coqui TTS 0.24.3 model."""

from __future__ import annotations

import gc
import logging
from pathlib import Path
from threading import RLock
from typing import Any

import torch

from comfy import model_management as mm
from comfy.model_patcher import ModelPatcher
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts


MODEL_TYPE = "XTTS_MANAGED_MODEL"
DEFAULT_MODEL_PATH = "/root/ComfyUI/models/xtts/xttsv2_banana"
DEFAULT_REFERENCE_PATH = "/root/ComfyUI/models/xtts/voice_refs/female_03.wav"

_REQUIRED_FILES = ("config.json", "model.pth", "vocab.json", "speakers_xtts.pth")
_LOGGER = logging.getLogger("XTTSManaged")
_MODEL_CACHE: dict[str, "ManagedXTTS"] = {}
_MODEL_CACHE_LOCK = RLock()


class ManagedXTTS:
    """CPU-resident XTTS model plus its native ComfyUI patcher and caches."""

    def __init__(self, checkpoint_dir: Path, model: Xtts, patcher: ModelPatcher):
        self.checkpoint_dir = checkpoint_dir
        self.model = model
        self.patcher = patcher
        self.conditioning_cache: dict[tuple[str, int, int], tuple[torch.Tensor, torch.Tensor]] = {}
        self.lock = RLock()
        self.checkpoint_load_count = 1


def _unique_storage_bytes(model: torch.nn.Module) -> int:
    """Count shared parameter and buffer storages only once for patcher sizing."""
    storages: dict[int, int] = {}
    for tensor in list(model.parameters()) + list(model.buffers()):
        storage = tensor.untyped_storage()
        storages[storage.data_ptr()] = storage.nbytes()
    return sum(storages.values())


def _resolve_checkpoint(path_text: str) -> Path:
    """Resolve and validate the explicit XTTS checkpoint directory."""
    checkpoint_dir = Path(path_text).expanduser().resolve(strict=True)
    if not checkpoint_dir.is_dir():
        raise ValueError(f"XTTS checkpoint is not a directory: {checkpoint_dir}")
    missing = [name for name in _REQUIRED_FILES if not (checkpoint_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"XTTS checkpoint is missing {missing}: {checkpoint_dir}")
    return checkpoint_dir


def _resolve_reference(model: ManagedXTTS, path_text: str) -> Path:
    """Resolve the explicit reference WAV path used for conditioning cache keys."""
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = model.checkpoint_dir.parent / "voice_refs" / path
    path = path.resolve(strict=True)
    if not path.is_file():
        raise FileNotFoundError(f"XTTS reference audio is not a file: {path}")
    return path


def _reference_key(path: Path) -> tuple[str, int, int]:
    """Build the requested path + mtime + size conditioning cache key."""
    stat = path.stat()
    return str(path), stat.st_mtime_ns, stat.st_size


def _model_owned_cuda_tensors(model: torch.nn.Module) -> list[str]:
    """Find non-parameter CUDA tensors retained by XTTS module attributes."""
    parameter_ids = {id(tensor) for tensor in model.parameters()}
    parameter_ids.update(id(tensor) for tensor in model.buffers())
    seen: set[int] = set()
    found: list[str] = []

    def walk(value: Any, path: str) -> None:
        value_id = id(value)
        if value_id in seen:
            return
        seen.add(value_id)
        if isinstance(value, torch.Tensor):
            if id(value) not in parameter_ids and value.device.type == "cuda":
                found.append(f"{path}: shape={tuple(value.shape)} dtype={value.dtype}")
            return
        if isinstance(value, dict):
            for key, item in value.items():
                walk(item, f"{path}[{key!r}]")
            return
        if isinstance(value, (list, tuple, set)):
            for index, item in enumerate(value):
                walk(item, f"{path}[{index}]")
            return
        if isinstance(value, torch.nn.Module):
            for name, item in value.__dict__.items():
                if name in {"_parameters", "_buffers"}:
                    continue
                walk(item, f"{path}.{name}")

    walk(model, "model")
    return found


def _clear_xtts_runtime_cache(model: Xtts) -> None:
    """Clear XTTS's non-parameter prefix cache and reject other retained CUDA tensors."""
    gpt_inference = getattr(getattr(model, "gpt", None), "gpt_inference", None)
    if gpt_inference is not None and getattr(gpt_inference, "cached_prefix_emb", None) is not None:
        gpt_inference.cached_prefix_emb = None
    gc.collect()
    retained = _model_owned_cuda_tensors(model)
    if retained:
        raise RuntimeError(f"XTTS retained non-parameter CUDA tensors after inference: {retained}")


def _load_managed_model(checkpoint_dir: Path) -> ManagedXTTS:
    """Load one CPU XTTS instance and create its native ComfyUI patcher."""
    cache_key = str(checkpoint_dir)
    with _MODEL_CACHE_LOCK:
        cached = _MODEL_CACHE.get(cache_key)
        if cached is not None:
            _LOGGER.info(
                "[XTTSManaged] model_cache_hit path=%s model_id=%d checkpoint_load_count=%d",
                checkpoint_dir, id(cached.model), cached.checkpoint_load_count,
            )
            return cached

        config = XttsConfig()
        config.load_json(str(checkpoint_dir / "config.json"))
        model = Xtts.init_from_config(config)
        model.load_checkpoint(
            config,
            checkpoint_dir=str(checkpoint_dir),
            vocab_path=str(checkpoint_dir / "vocab.json"),
            use_deepspeed=False,
        )
        if set(str(tensor.device) for tensor in model.parameters()) != {"cpu"}:
            raise RuntimeError("XTTS checkpoint did not load into CPU RAM")
        device = mm.get_torch_device()
        patcher = ModelPatcher(
            model,
            load_device=device,
            offload_device=torch.device("cpu"),
            size=_unique_storage_bytes(model),
        )
        managed = ManagedXTTS(checkpoint_dir, model, patcher)
        _MODEL_CACHE[cache_key] = managed
        _LOGGER.info(
            "[XTTSManaged] checkpoint_loaded path=%s model_id=%d checkpoint_load_count=%d patcher_id=%d",
            checkpoint_dir, id(model), managed.checkpoint_load_count, id(patcher),
        )
        return managed


class XTTSManagedLoader:
    """Load and cache banana XTTS in CPU RAM under ComfyUI model management."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_path": ("STRING", {"default": DEFAULT_MODEL_PATH, "multiline": False}),
            }
        }

    RETURN_TYPES = (MODEL_TYPE,)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "audio/xtts"

    def load_model(self, model_path: str):
        managed = _load_managed_model(_resolve_checkpoint(model_path))
        return (managed,)


class XTTSManagedGenerate:
    """Generate 24 kHz Russian or multilingual audio through native ComfyUI residency."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (MODEL_TYPE,),
                "text": ("STRING", {"default": "Привет, это проверка переноса голоса.", "multiline": True}),
                "language": (["ru", "en", "es", "fr", "de", "it", "pt", "pl", "tr", "zh-cn", "ja", "ko"], {"default": "ru"}),
                "reference_audio": ("STRING", {"default": DEFAULT_REFERENCE_PATH, "multiline": False}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = "audio/xtts"

    def generate(self, model: ManagedXTTS, text: str, language: str, reference_audio: str):
        if not isinstance(model, ManagedXTTS):
            raise TypeError(f"Expected {MODEL_TYPE}, got {type(model).__name__}")
        if not text.strip():
            raise ValueError("XTTS text must not be empty")
        reference_path = _resolve_reference(model, reference_audio)

        with model.lock:
            mm.load_models_gpu([model.patcher])
            device = model.patcher.load_device
            _LOGGER.info(
                "[XTTSManaged] gpu_resident model_id=%d device=%s loaded_models=%d",
                id(model.model), device, len(mm.loaded_models()),
            )
            if any(tensor.device != device for tensor in model.model.parameters()):
                raise RuntimeError(f"XTTS parameters are not on the managed device {device}")

            key = _reference_key(reference_path)
            cached = model.conditioning_cache.get(key)
            if cached is None:
                _LOGGER.info("[XTTSManaged] conditioning_cache_miss key=%s", key)
                gpt_cond_latent, speaker_embedding = model.model.get_conditioning_latents(
                    audio_path=[str(reference_path)],
                    gpt_cond_len=model.model.config.gpt_cond_len,
                    max_ref_length=model.model.config.max_ref_len,
                    sound_norm_refs=model.model.config.sound_norm_refs,
                )
                cached = (gpt_cond_latent.detach().cpu(), speaker_embedding.detach().cpu())
                model.conditioning_cache[key] = cached
            else:
                _LOGGER.info("[XTTSManaged] conditioning_cache_hit key=%s", key)
            gpt_cond_latent, speaker_embedding = cached

            output = None
            wav = None
            try:
                output = model.model.inference(
                    text=text,
                    language=language,
                    gpt_cond_latent=gpt_cond_latent,
                    speaker_embedding=speaker_embedding,
                    temperature=float(model.model.config.temperature),
                    length_penalty=float(model.model.config.length_penalty),
                    repetition_penalty=float(model.model.config.repetition_penalty),
                    top_k=int(model.model.config.top_k),
                    top_p=float(model.model.config.top_p),
                    speed=1.0,
                    enable_text_splitting=True,
                )
                wav = output["wav"]
                if isinstance(wav, torch.Tensor):
                    if wav.ndim != 1:
                        raise RuntimeError(f"XTTS returned unexpected waveform shape: {tuple(wav.shape)}")
                    waveform = wav.detach().cpu().reshape(1, 1, -1)
                else:
                    shape = getattr(wav, "shape", None)
                    if shape is None or len(shape) != 1:
                        raise RuntimeError(f"XTTS returned unexpected waveform: {type(wav).__name__} {shape}")
                    waveform = torch.as_tensor(wav, dtype=torch.float32).reshape(1, 1, -1)
                return ({"waveform": waveform, "sample_rate": 24000},)
            finally:
                del output, wav, gpt_cond_latent, speaker_embedding
                _clear_xtts_runtime_cache(model.model)
                _LOGGER.info(
                    "[XTTSManaged] inference_cleanup model_id=%d conditioning_cache_entries=%d",
                    id(model.model), len(model.conditioning_cache),
                )


NODE_CLASS_MAPPINGS = {
    "XTTSManagedLoader": XTTSManagedLoader,
    "XTTSManagedGenerate": XTTSManagedGenerate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "XTTSManagedLoader": "XTTS Managed Loader",
    "XTTSManagedGenerate": "XTTS Managed Generate",
}

"""Safe XTTS custom-node entrypoint; inference nodes await managed GPU integration."""

import sys
from pathlib import Path

_VENDOR = Path(__file__).resolve().parent / "vendor"
sys.path.insert(0, str(_VENDOR))

import TTS

if Path(TTS.__file__).resolve() != _VENDOR / "TTS" / "__init__.py":
    raise RuntimeError(f"Expected vendored Coqui TTS, got {TTS.__file__}")
if TTS.__version__ != "0.24.3":
    raise RuntimeError(f"Expected Coqui TTS 0.24.3, got {TTS.__version__}")

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

# No inference nodes are registered until ComfyUI-managed model loading exists.
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

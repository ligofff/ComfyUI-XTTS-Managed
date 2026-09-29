"""ComfyUI entrypoint for the vendored, ComfyUI-managed XTTS nodes."""

import sys
from pathlib import Path

_VENDOR = Path(__file__).resolve().parent / "vendor"
if str(_VENDOR) not in sys.path:
    sys.path.insert(0, str(_VENDOR))

import TTS

if Path(TTS.__file__).resolve() != _VENDOR / "TTS" / "__init__.py":
    raise RuntimeError(f"Expected vendored Coqui TTS, got {TTS.__file__}")
if TTS.__version__ != "0.24.3":
    raise RuntimeError(f"Expected Coqui TTS 0.24.3, got {TTS.__version__}")

from .managed_nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

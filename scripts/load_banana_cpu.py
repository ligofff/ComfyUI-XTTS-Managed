"""Standalone CPU-only checkpoint load probe; not a ComfyUI node or inference path."""

import argparse
import os
import sys
from pathlib import Path
from time import perf_counter

# Keep CUDA invisible before importing torch or the vendored TTS package.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
VENDOR = Path(__file__).resolve().parents[1] / "vendor"
sys.path.insert(0, str(VENDOR))

import torch
import TTS
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts


def memory_mib() -> tuple[int, int]:
    """Return process current and peak RSS, in MiB, from procfs."""
    values = {}
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith(("VmRSS:", "VmHWM:")):
            name, kib, _unit = line.split()
            values[name.rstrip(":")] = int(kib) // 1024
    return values["VmRSS"], values["VmHWM"]


def main() -> None:
    """Load banana using the same XttsConfig/init/load_checkpoint calls as AllTalk."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint_dir", type=Path)
    checkpoint_dir = parser.parse_args().checkpoint_dir.resolve(strict=True)
    for filename in ("config.json", "vocab.json", "model.pth", "speakers_xtts.pth"):
        if not (checkpoint_dir / filename).is_file():
            raise FileNotFoundError(checkpoint_dir / filename)
    if torch.cuda.is_available():
        raise RuntimeError("CUDA must be hidden for this CPU-only checkpoint probe")
    if Path(TTS.__file__).resolve() != VENDOR / "TTS" / "__init__.py":
        raise RuntimeError(f"Unexpected TTS package: {TTS.__file__}")
    print(f"runtime python={sys.version.split()[0]} torch={torch.__version__} TTS={TTS.__version__}", flush=True)
    print(f"start rss_mib={memory_mib()[0]} peak_mib={memory_mib()[1]}", flush=True)
    started = perf_counter()
    config = XttsConfig()
    config.load_json(str(checkpoint_dir / "config.json"))
    print(f"config rss_mib={memory_mib()[0]} peak_mib={memory_mib()[1]}", flush=True)
    model = Xtts.init_from_config(config)
    print(f"init rss_mib={memory_mib()[0]} peak_mib={memory_mib()[1]}", flush=True)
    model.load_checkpoint(
        config,
        checkpoint_dir=str(checkpoint_dir),
        vocab_path=str(checkpoint_dir / "vocab.json"),
        use_deepspeed=False,
    )
    devices = {str(parameter.device) for parameter in model.parameters()}
    if devices != {"cpu"}:
        raise RuntimeError(f"Expected all parameters on CPU, got {devices}")
    rss, peak = memory_mib()
    print(f"CHECKPOINT_LOADED seconds={perf_counter() - started:.1f} rss_mib={rss} peak_mib={peak} device=cpu", flush=True)


if __name__ == "__main__":
    main()

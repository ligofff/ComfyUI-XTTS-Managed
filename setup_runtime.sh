#!/usr/bin/env bash
# Install only the vendored XTTS packages absent from the base ComfyUI image.
# The base image owns torch/torchaudio; this script never installs or downgrades them.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REQUIREMENTS="$ROOT/requirements.txt"

python3 - "$REQUIREMENTS" <<'PY'
import importlib.metadata as metadata
import sys

requirements = {
    "coqpit": "0.0.17",
    "coqui-tts-trainer": "0.1.7",
    "librosa": "0.11.0",
    "soundfile": "0.13.1",
    "num2words": "0.5.14",
    "spacy": "3.8.12",
}
for package, expected in requirements.items():
    try:
        actual = metadata.version(package)
    except metadata.PackageNotFoundError:
        continue
    if actual != expected:
        raise SystemExit(
            f"{package}=={actual} is already installed; expected {expected}. "
            "Refusing to downgrade or replace an existing package."
        )
try:
    torchaudio = metadata.version("torchaudio")
except metadata.PackageNotFoundError:
    raise SystemExit(
        "torchaudio is missing from the base ComfyUI image. "
        "Install a torch-matched base image package first; XTTS setup will not install it."
    )
print(f"[XTTS] base torchaudio={torchaudio}")
PY

if python3 - "$REQUIREMENTS" <<'PY'
import importlib.metadata as metadata

required = {
    "coqpit": "0.0.17",
    "coqui-tts-trainer": "0.1.7",
    "librosa": "0.11.0",
    "soundfile": "0.13.1",
    "num2words": "0.5.14",
    "spacy": "3.8.12",
}
missing = []
for package, expected in required.items():
    try:
        actual = metadata.version(package)
    except metadata.PackageNotFoundError:
        missing.append(f"{package}=={expected}")
    else:
        print(f"[XTTS] {package}=={actual}")
if missing:
    print("[XTTS] installing missing packages:")
    print("[XTTS] " + " ".join(missing))
    raise SystemExit(10)
PY
then
    :
else
    status=$?
    if [[ "$status" != 10 ]]; then
        exit "$status"
    fi
    python3 -m pip install --user --no-cache-dir --upgrade-strategy only-if-needed -r "$REQUIREMENTS"
fi

python3 - "$ROOT" <<'PY'
import importlib.metadata as metadata
import importlib.util
import sys
from pathlib import Path

root = Path(sys.argv[1])
expected = {
    "coqpit": "0.0.17",
    "coqui-tts-trainer": "0.1.7",
    "librosa": "0.11.0",
    "soundfile": "0.13.1",
    "num2words": "0.5.14",
    "spacy": "3.8.12",
}
for package, version in expected.items():
    actual = metadata.version(package)
    if actual != version:
        raise SystemExit(f"{package}: expected {version}, got {actual}")
for module in ("coqpit", "trainer", "librosa", "soundfile", "num2words", "spacy"):
    if importlib.util.find_spec(module) is None:
        raise SystemExit(f"missing import after setup: {module}")
print("[XTTS] dependency verification passed")
PY

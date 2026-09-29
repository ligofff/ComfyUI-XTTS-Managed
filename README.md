# ComfyUI-XTTS-Managed

ComfyUI custom nodes for the vendored Coqui TTS `0.24.3` XTTS model. The
active implementation is deliberately limited to two nodes:

- `XTTSManagedLoader`
- `XTTSManagedGenerate`

The old AIFSH implementation is retained under `legacy/` for history only and
is not imported by `__init__.py`.

## Runtime paths

The verified banana model directory is:

```text
/root/ComfyUI/models/xtts/xttsv2_banana
```

The verified Russian reference audio is:

```text
/root/ComfyUI/models/xtts/voice_refs/female_03.wav
```

The ready-to-submit API workflow is:

```text
workflows/xtts_managed_minimal_api.json
```

It is API-format JSON for `POST /prompt` and contains:

```text
XTTSManagedLoader -> XTTSManagedGenerate -> SaveAudio
```

## Runtime setup

The base ComfyUI image supplies `torch`, `torchaudio`, `transformers`,
`tokenizers`, `scipy`, `numpy`, `einops`, `fsspec` and `safetensors`.
`torchaudio` is intentionally not installed by this project: vendored XTTS
uses the base image's torch-matched `torchaudio` for MelSpectrogram and
resampling. `soundfile` is used for reference-audio loading instead of the
broken `torchaudio.load`/TorchCodec path.

The packages absent from the base image and pinned for this fork are in
`requirements.txt`:

```text
coqpit==0.0.17
coqui-tts-trainer==0.1.7
librosa==0.11.0
soundfile==0.13.1
num2words==0.5.14
spacy==3.8.12
```

Run the reproducible setup explicitly after installing the custom node:

```bash
bash /root/ComfyUI/custom_nodes/ComfyUI-XTTS-Managed/setup_runtime.sh
```

The setup script verifies the base `torchaudio` first, installs only the
missing pinned packages, refuses to replace a different already-installed
version, and verifies imports afterward. It never installs or downgrades
`torch` or `torchaudio`.

## Node behavior

### `XTTSManagedLoader`

- Loads `xttsv2_banana` once into CPU RAM.
- Creates one native ComfyUI `ModelPatcher` for that model.
- Caches the managed object by resolved model directory.
- Repeated workflow executions reuse the same XTTS instance and do not call
  `load_checkpoint` again.

### `XTTSManagedGenerate`

Inputs:

- managed model
- text
- language, with `ru` available
- reference audio path

Before inference it calls only ComfyUI's
`model_management.load_models_gpu([patcher])`. ComfyUI therefore owns GPU
residency and can evict XTTS automatically when another workflow needs VRAM.
The node does not call manual CUDA transfers, `torch.cuda.empty_cache()` or
`unload_model_and_clones()`.

Conditioning is cached in CPU RAM by:

```text
resolved reference path + mtime_ns + file size
```

The cached `gpt_cond_latent` and `speaker_embedding` remain in CPU RAM; the
existing XTTS inference path places them on the model's actual device when
used. After every inference, `gpt_inference.cached_prefix_emb` is cleared and
remaining non-parameter CUDA tensors owned by the XTTS module are checked.

The output is the standard ComfyUI `AUDIO` object:

```python
{"waveform": [batch, channels, samples], "sample_rate": 24000}
```

`SaveAudio` performs file encoding; the XTTS node does not write audio files.

## API example

```bash
curl -X POST http://127.0.0.1:8188/prompt \
  -H 'Content-Type: application/json' \
  --data-binary @workflows/xtts_managed_minimal_api.json
```

The workflow uses the default paths above and generates Russian speech from
`female_03.wav`. Change the `text`, `language` or `reference_audio` values in
the JSON inputs before submitting.

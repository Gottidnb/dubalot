# dubalot

Translate a foreign-language video to your native language **with the same
voice** – zero-shot voice cloning + automatic lip sync.

## Features

| Feature | How it works |
|---------|--------------|
| **Voice cloning** | [Coqui TTS / XTTS v2](https://github.com/coqui-ai/TTS) — zero-shot multilingual voice cloning from a ≥ 6-second reference clip |
| **Lip sync** | [Wav2Lip](https://github.com/Rudrabha/Wav2Lip) when a checkpoint is available, otherwise ffmpeg audio-replacement fallback |
| **Transcription** | [OpenAI Whisper](https://github.com/openai/whisper) — auto-detects source language |
| **Translation** | [deep-translator](https://github.com/nidhaloff/deep-translator) (Google Translate backend) |

## Installation

```bash
pip install -r requirements.txt
# ffmpeg must be installed and available on $PATH
```

Or install as a package (also registers the `dubalot` CLI command):

```bash
pip install -e .
```

## Quick start

### CLI

```bash
dubalot -i foreign_video.mp4 -t es -o video_es.mp4
```

Full options:

```
usage: dubalot [-h] -i INPUT -o OUTPUT [-t LANG] [--whisper-model MODEL]
               [--wav2lip-checkpoint PATH] [--wav2lip-script PATH]

  -i, --input              Path to the source video file
  -o, --output             Path for the dubbed output video
  -t, --target-language    Target language BCP-47 code (e.g. en, es, fr). Default: en
  --whisper-model          Whisper model size: tiny, base, small, medium, large. Default: base
  --wav2lip-checkpoint     Optional path to a Wav2Lip .pth checkpoint for lip animation
  --wav2lip-script         Optional path to Wav2Lip inference.py script
```

### Python API

```python
from dubalot import DubalotPipeline

pipeline = DubalotPipeline(target_language="en")   # dub to English
pipeline.run("foreign_video.mp4", "dubbed_output.mp4")
```

## Component usage

### Voice cloning only

```python
from dubalot import VoiceCloner

cloner = VoiceCloner()

# Extract a reference clip from the source video
ref_audio = cloner.extract_reference_audio("source.mp4", "reference.wav")

# Synthesise speech in the cloned voice
cloner.synthesise(
    text="Hello, this is the dubbed line.",
    reference_audio=ref_audio,
    output_path="dubbed_speech.wav",
    language="en",
)
```

### Lip sync only

```python
from dubalot import LipSyncer

syncer = LipSyncer(
    # Optional – use Wav2Lip for animated lips
    wav2lip_checkpoint="Wav2Lip/checkpoints/wav2lip.pth",
    wav2lip_script="Wav2Lip/inference.py",
)

syncer.sync(
    video_path="original.mp4",
    audio_path="dubbed_speech.wav",
    output_path="final.mp4",
)
```

## Wav2Lip setup (optional, for animated lip sync)

1. Clone [Wav2Lip](https://github.com/Rudrabha/Wav2Lip) and download the
   pretrained checkpoint (`wav2lip.pth` or `wav2lip_gan.pth`).
2. Pass the paths to `LipSyncer`:
   ```python
   LipSyncer(
       wav2lip_checkpoint="Wav2Lip/checkpoints/wav2lip.pth",
       wav2lip_script="Wav2Lip/inference.py",
   )
   ```
   Or via the CLI:
   ```bash
   dubalot -i video.mp4 -t es -o video_es.mp4 \
     --wav2lip-checkpoint Wav2Lip/checkpoints/wav2lip.pth \
     --wav2lip-script Wav2Lip/inference.py
   ```

Without Wav2Lip, `LipSyncer` falls back to simple audio replacement using
`ffmpeg` – the video plays correctly with the dubbed audio but the lip
movements are not re-animated.

## Demo checklist

Follow these steps to run a full end-to-end demo:

- [ ] **System prerequisites**
  - Python ≥ 3.9
  - `ffmpeg` installed and on `$PATH` (`ffmpeg -version` should work)
  - GPU with CUDA (optional, but strongly recommended for TTS speed)

- [ ] **Install Python dependencies**
  ```bash
  pip install -r requirements.txt
  # or: pip install -e .
  ```

- [ ] **Prepare a test video** – any short MP4 with speech (≥ 6 s of audio)

- [ ] **Run dubbing via CLI** (ffmpeg fallback, no GPU models required beyond Whisper/TTS):
  ```bash
  dubalot -i video.mp4 -t es -o video_es.mp4
  ```

- [ ] **Run dubbing via Python**:
  ```python
  from dubalot import DubalotPipeline
  pipeline = DubalotPipeline(target_language="es")
  pipeline.run("video.mp4", "video_es.mp4")
  ```

- [ ] *(Optional)* **Enable animated lip sync with Wav2Lip**:
  1. Clone Wav2Lip: `git clone https://github.com/Rudrabha/Wav2Lip`
  2. Download `wav2lip.pth` checkpoint into `Wav2Lip/checkpoints/`
  3. Run:
     ```bash
     dubalot -i video.mp4 -t es -o video_es.mp4 \
       --wav2lip-checkpoint Wav2Lip/checkpoints/wav2lip.pth \
       --wav2lip-script Wav2Lip/inference.py
     ```

> **GPU / ML notes**
> - Whisper model weights are downloaded automatically on first run (~150 MB for `base`).
> - Coqui TTS XTTS v2 weights are downloaded automatically on first use (~1.8 GB).
> - A GPU is not required but speeds up TTS synthesis significantly.

## Running tests

```bash
pip install pytest
python -m pytest tests/ -v
```


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

## Quick start

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

Without Wav2Lip, `LipSyncer` falls back to simple audio replacement using
`ffmpeg` – the video plays correctly with the dubbed audio but the lip
movements are not re-animated.

## Running tests

```bash
pip install pytest
python -m pytest tests/ -v
```

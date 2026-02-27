# dubalot
Translate foreign video to native language with the same voice.

## Requirements

- Python 3.8+
- [ffmpeg](https://ffmpeg.org/download.html) installed and available on `PATH`

## Video Editing

`video_editor.py` provides the core video editing utilities used in the dubbing pipeline:

| Function | Description |
|---|---|
| `extract_audio(video_path, output_path=None)` | Extract the audio track from a video file as a WAV |
| `replace_audio(video_path, audio_path, output_path)` | Replace the audio track of a video with a new audio file |
| `trim_video(video_path, output_path, start_time, end_time=None)` | Trim a video to the specified time range |

### Example

```python
from video_editor import extract_audio, replace_audio, trim_video

# Extract original audio
extract_audio("input.mp4", "original_audio.wav")

# … translate / synthesise new audio …

# Replace audio with translated version
replace_audio("input.mp4", "translated_audio.wav", "dubbed.mp4")

# Trim to a specific segment (seconds)
trim_video("dubbed.mp4", "segment.mp4", start_time=10, end_time=60)
```

## Running Tests

```bash
python -m pytest tests/
```

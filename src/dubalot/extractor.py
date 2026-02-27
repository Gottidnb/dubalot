"""
extractor.py – extract audio from video and separate speech from background.

Two-stage process:
  1. Dump the full audio track from the video using MoviePy / ffmpeg.
  2. Separate the *vocals* (speech) from the *background* (music, ambient noise)
     using Demucs (optional) so that background can be re-mixed after TTS.

If Demucs is not installed, the full audio is used as the vocal track and no
background-only stem is produced; the pipeline still works but background audio
will not be preserved independently.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path
from typing import Tuple

logger = logging.getLogger(__name__)


def extract_audio(video_path: Path, output_path: Path) -> Path:
    """Extract the audio track from *video_path* and write it to *output_path*.

    The output is always a mono/stereo 16-bit PCM WAV file at 16 kHz so that
    downstream Whisper transcription works without resampling.

    Args:
        video_path: Path to the source video file.
        output_path: Destination path for the extracted WAV file.

    Returns:
        *output_path* after successful extraction.

    Raises:
        FileNotFoundError: If *video_path* does not exist.
        RuntimeError: If audio extraction fails.
    """
    video_path = Path(video_path)
    output_path = Path(output_path)

    if not video_path.exists():
        raise FileNotFoundError(f"Input video not found: {video_path}")

    logger.info("Extracting audio from %s", video_path)

    # Use ffmpeg directly for reliable extraction; MoviePy is an optional thin
    # wrapper around ffmpeg and may not always be installed.
    try:
        _ffmpeg_extract(video_path, output_path)
    except (FileNotFoundError, subprocess.CalledProcessError):
        # Fall back to MoviePy if ffmpeg is not on PATH
        _moviepy_extract(video_path, output_path)

    logger.info("Audio extracted to %s", output_path)
    return output_path


def separate_audio_stems(
    audio_path: Path, output_dir: Path
) -> Tuple[Path, Path | None]:
    """Separate *audio_path* into a vocals stem and a background stem.

    Uses **Demucs** (``pip install demucs``) when available.  Falls back to
    returning the original audio as the vocal stem with no separate background
    stem when Demucs is not installed.

    Args:
        audio_path: Path to the full mixed audio (WAV).
        output_dir: Directory in which Demucs should write its stems.

    Returns:
        ``(vocals_path, background_path)`` where *background_path* may be
        ``None`` if Demucs is unavailable.
    """
    audio_path = Path(audio_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Try Demucs
    if shutil.which("demucs") is not None:
        return _demucs_separate(audio_path, output_dir)

    try:
        import demucs  # noqa: F401 – presence check only

        return _demucs_python_separate(audio_path, output_dir)
    except ImportError:
        pass

    # Graceful fallback: treat full audio as vocals, no background stem
    logger.warning(
        "Demucs is not installed.  Background audio will not be preserved "
        "independently.  Install it with:  pip install demucs"
    )
    return audio_path, None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _ffmpeg_extract(video_path: Path, output_path: Path) -> None:
    """Use the system ffmpeg binary to extract audio."""
    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(video_path),
        "-vn",               # drop video
        "-ar", "16000",      # 16 kHz sample rate (Whisper optimum)
        "-ac", "1",          # mono
        "-acodec", "pcm_s16le",
        str(output_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def _moviepy_extract(video_path: Path, output_path: Path) -> None:
    """Fall back to MoviePy for audio extraction."""
    from moviepy.editor import VideoFileClip  # type: ignore

    with VideoFileClip(str(video_path)) as clip:
        if clip.audio is None:
            raise RuntimeError(f"Video {video_path} has no audio track.")
        clip.audio.write_audiofile(
            str(output_path),
            fps=16000,
            nbytes=2,
            ffmpeg_params=["-ac", "1"],
            logger=None,
        )


def _demucs_separate(audio_path: Path, output_dir: Path) -> Tuple[Path, Path]:
    """Run Demucs via its CLI and return (vocals, no_vocals) paths."""
    logger.info("Running Demucs source separation on %s", audio_path)
    cmd = [
        "demucs",
        "--two-stems=vocals",
        "-o", str(output_dir),
        str(audio_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)

    stem = audio_path.stem
    # Demucs writes to: output_dir/htdemucs/<stem>/vocals.wav
    vocals = output_dir / "htdemucs" / stem / "vocals.wav"
    background = output_dir / "htdemucs" / stem / "no_vocals.wav"

    if not vocals.exists() or not background.exists():
        raise RuntimeError(
            f"Demucs did not produce expected output files under {output_dir}"
        )

    logger.info("Source separation complete: vocals=%s, background=%s", vocals, background)
    return vocals, background


def _demucs_python_separate(
    audio_path: Path, output_dir: Path
) -> Tuple[Path, Path]:
    """Use Demucs Python API for source separation."""
    import torch
    from demucs.pretrained import get_model  # type: ignore
    from demucs.apply import apply_model  # type: ignore
    import soundfile as sf  # type: ignore
    import numpy as np

    logger.info("Running Demucs (Python API) source separation")
    model = get_model("htdemucs")
    model.eval()

    wav, sr = sf.read(str(audio_path), always_2d=True)
    wav = wav.T  # (channels, samples)
    wav_tensor = torch.tensor(wav, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        sources = apply_model(model, wav_tensor)

    source_names = model.sources  # e.g. ["drums", "bass", "other", "vocals"]
    vocals_idx = source_names.index("vocals")
    background_idxs = [i for i in range(len(source_names)) if i != vocals_idx]

    vocals_wav = sources[0, vocals_idx].numpy().T
    background_wav = sum(
        sources[0, i].numpy().T for i in background_idxs
    )

    vocals_path = output_dir / "vocals.wav"
    background_path = output_dir / "background.wav"

    sf.write(str(vocals_path), vocals_wav, sr)
    sf.write(str(background_path), background_wav, sr)

    return vocals_path, background_path

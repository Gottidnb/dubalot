"""
mixer.py – mix translated speech with the background audio track.

The mixer combines:
  • *speech_path*    – the TTS-generated, time-stretched speech track
  • *background_path* – the background-only stem from source separation
                        (may be ``None`` if Demucs was unavailable)

into a single mixed WAV file that the lip-sync and video-composition steps
can use.  When no background stem is available the speech track is returned
as-is, which still produces correct output but without the original background
ambience.

Volume balancing
----------------
The background is attenuated by *bg_gain* (default −3 dB) to prevent it from
masking the foreground speech.  The speech level is normalised to −3 dBFS.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf  # type: ignore
import librosa  # type: ignore

logger = logging.getLogger(__name__)

# Background gain relative to speech (in linear scale, not dB)
_DEFAULT_BG_GAIN = 0.7   # ≈ −3 dB
_SPEECH_TARGET_LUFS = 0.5  # normalised peak target (linear)


def mix_audio(
    speech_path: Path,
    background_path: Optional[Path],
    output_path: Path,
    bg_gain: float = _DEFAULT_BG_GAIN,
) -> Path:
    """Mix *speech_path* with *background_path* and write the result.

    Both files are resampled to a common sample rate (the higher of the two)
    before mixing.  The output has the same duration as the speech track.

    Args:
        speech_path: Path to the translated speech WAV.
        background_path: Path to the background-only WAV, or ``None``.
        output_path: Destination path for the mixed WAV.
        bg_gain: Linear gain applied to the background track (0.0–1.0).

    Returns:
        *output_path* after successful mixing.
    """
    speech_path = Path(speech_path)
    output_path = Path(output_path)

    speech, speech_sr = _load_mono(speech_path)

    if background_path is None or not Path(background_path).exists():
        logger.info(
            "No background stem available; using speech track only."
        )
        sf.write(str(output_path), speech, speech_sr)
        return output_path

    background, bg_sr = _load_mono(Path(background_path))

    # Resample background to speech sample rate
    if bg_sr != speech_sr:
        background = librosa.resample(background, orig_sr=bg_sr, target_sr=speech_sr)

    # Match lengths (trim or pad background to speech duration)
    background = _match_length(background, len(speech))

    # Normalise speech peak
    speech = _normalise(speech, _SPEECH_TARGET_LUFS)

    # Apply gain to background
    background = background * bg_gain

    mixed = speech + background

    # Final peak normalisation to prevent clipping
    peak = np.max(np.abs(mixed))
    if peak > 1.0:
        mixed /= peak

    sf.write(str(output_path), mixed, speech_sr)
    logger.info("Mixed audio written to %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_mono(path: Path) -> tuple[np.ndarray, int]:
    """Load an audio file as a mono float32 array."""
    audio, sr = sf.read(str(path), always_2d=False, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio, sr


def _match_length(audio: np.ndarray, target_length: int) -> np.ndarray:
    """Trim or zero-pad *audio* to exactly *target_length* samples."""
    if len(audio) >= target_length:
        return audio[:target_length]
    return np.pad(audio, (0, target_length - len(audio)))


def _normalise(audio: np.ndarray, target_peak: float) -> np.ndarray:
    """Peak-normalise *audio* to *target_peak*."""
    peak = np.max(np.abs(audio))
    if peak == 0:
        return audio
    return audio * (target_peak / peak)

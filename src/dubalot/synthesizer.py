"""
synthesizer.py – text-to-speech with voice cloning using Coqui XTTS v2.

For each translated segment the synthesiser:
  1. Generates speech from translated text, cloning the speaker's voice from
     a reference clip (extracted from the original audio).
  2. Time-stretches the generated audio to fit the original segment's duration
     so that the translated speech stays in sync with the video.

XTTS v2 is a multilingual model that naturally produces a slight accent in the
target language when given a reference speaker that is non-native – satisfying
the "keep voices the same with a bit of accent" requirement.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import List, Optional

import numpy as np
import soundfile as sf  # type: ignore
import librosa  # type: ignore

from .transcriber import TranscriptionSegment

logger = logging.getLogger(__name__)

# Default XTTS v2 model identifier
_XTTS_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"

# Duration tolerance: if the generated audio is within this fraction of the
# target duration, skip time-stretching to avoid artefacts.
_STRETCH_TOLERANCE = 0.05


def synthesize_segments(
    segments: List[TranscriptionSegment],
    voice_reference: Path,
    target_language: str,
    output_path: Path,
    device: str = "cpu",
    model_name: str = _XTTS_MODEL,
) -> Path:
    """Generate translated speech and write a single WAV to *output_path*.

    Each segment is synthesised individually (preserving voice identity via
    XTTS v2 voice cloning), then time-stretched to match the original segment
    duration, and finally assembled into a full-length audio file with silence
    gaps where no speech occurs.

    Args:
        segments: Translated transcription segments (with original timing).
        voice_reference: Path to a clean mono WAV clip of the original speaker
            (3–30 s is ideal for XTTS).
        target_language: BCP-47 language code accepted by XTTS v2.
        output_path: Destination WAV path for the assembled speech track.
        device: ``"cpu"`` or ``"cuda"``.
        model_name: XTTS model identifier (override for custom models).

    Returns:
        *output_path* after successful synthesis.
    """
    from TTS.api import TTS  # type: ignore

    voice_reference = Path(voice_reference)
    output_path = Path(output_path)

    if not segments:
        # Write silence of 1 second so downstream code always gets a valid file
        _write_silence(output_path, duration=1.0)
        return output_path

    # Determine total duration from the last segment's end time
    total_duration = segments[-1].end
    sample_rate = 24000  # XTTS v2 native sample rate

    logger.info("Loading TTS model: %s on %s", model_name, device)
    tts = TTS(model_name, gpu=(device == "cuda"))

    # Allocate output buffer (stereo ← XTTS outputs stereo; fold to mono later)
    n_samples = int(total_duration * sample_rate) + sample_rate  # +1 s margin
    output_buffer = np.zeros(n_samples, dtype=np.float32)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        for idx, seg in enumerate(segments):
            if not seg.text.strip():
                continue

            seg_audio_path = tmp_dir / f"seg_{idx:04d}.wav"
            logger.debug("Synthesising segment %d: '%s'", idx, seg.text)

            tts.tts_to_file(
                text=seg.text,
                speaker_wav=str(voice_reference),
                language=target_language,
                file_path=str(seg_audio_path),
            )

            seg_audio, sr = sf.read(str(seg_audio_path), always_2d=False, dtype="float32")
            # Ensure mono
            if seg_audio.ndim > 1:
                seg_audio = seg_audio.mean(axis=1)
            # Resample to output sample rate if necessary
            if sr != sample_rate:
                seg_audio = librosa.resample(seg_audio, orig_sr=sr, target_sr=sample_rate)

            seg_audio = _fit_to_duration(seg_audio, seg.duration, sample_rate)

            start_sample = int(seg.start * sample_rate)
            end_sample = start_sample + len(seg_audio)
            if end_sample > len(output_buffer):
                # Extend buffer if last segment overruns
                output_buffer = np.pad(output_buffer, (0, end_sample - len(output_buffer)))
            output_buffer[start_sample:end_sample] += seg_audio

    # Normalise to avoid clipping
    peak = np.max(np.abs(output_buffer))
    if peak > 1.0:
        output_buffer /= peak

    sf.write(str(output_path), output_buffer, sample_rate)
    logger.info("Synthesised speech track written to %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _fit_to_duration(
    audio: np.ndarray, target_duration: float, sample_rate: int
) -> np.ndarray:
    """Time-stretch *audio* so that it lasts exactly *target_duration* seconds.

    Librosa's phase-vocoder is used for stretching; if the required rate is
    within :data:`_STRETCH_TOLERANCE` of 1.0 the audio is returned unchanged.
    """
    current_duration = len(audio) / sample_rate
    if current_duration == 0 or target_duration <= 0:
        return audio

    rate = current_duration / target_duration  # >1 → speed up, <1 → slow down

    if abs(rate - 1.0) <= _STRETCH_TOLERANCE:
        return audio

    # Clamp to avoid extreme distortion
    rate = float(np.clip(rate, 0.5, 2.0))
    stretched = librosa.effects.time_stretch(audio, rate=rate)

    # Trim or pad to exact length
    target_samples = int(target_duration * sample_rate)
    if len(stretched) > target_samples:
        stretched = stretched[:target_samples]
    elif len(stretched) < target_samples:
        stretched = np.pad(stretched, (0, target_samples - len(stretched)))

    return stretched


def _write_silence(path: Path, duration: float = 1.0, sample_rate: int = 24000) -> None:
    """Write a silent WAV file."""
    silence = np.zeros(int(duration * sample_rate), dtype=np.float32)
    sf.write(str(path), silence, sample_rate)

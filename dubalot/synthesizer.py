"""Text-to-speech synthesis with voice cloning using Coqui TTS.

Given translated segments and a reference audio clip (used to capture the
speaker's voice characteristics), this module generates a new audio file
where each segment is spoken in the target language using the cloned voice.
"""

from __future__ import annotations

import os
import tempfile
from typing import List

from dubalot.transcriber import Segment


def synthesize(
    segments: List[Segment],
    reference_audio: str,
    output_path: str,
    sample_rate: int = 24000,
    model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2",
) -> str:
    """Generate dubbed audio for *segments* using the voice in *reference_audio*.

    Parameters
    ----------
    segments:
        Translated segments (from :func:`dubalot.translator.translate`).
    reference_audio:
        Path to a short (~6 s) audio clip of the original speaker.  This
        clip is used by the XTTS model to clone the speaker's voice.
    output_path:
        Destination path for the synthesised audio file (WAV).
    sample_rate:
        Sample rate of the output audio in Hz.
    model_name:
        Coqui TTS model identifier.  Defaults to XTTS v2 which supports
        multilingual voice cloning.

    Returns
    -------
    str
        The path to the generated audio file (*output_path*).
    """
    from TTS.api import TTS  # lazy import
    import numpy as np
    from scipy.io.wavfile import write as wav_write

    tts = TTS(model_name=model_name, progress_bar=False)

    # Determine total duration from the last segment end time
    if not segments:
        raise ValueError("No segments provided for synthesis.")

    total_duration = segments[-1].end
    audio_buffer = np.zeros(int(total_duration * sample_rate), dtype=np.float32)

    with tempfile.TemporaryDirectory() as tmpdir:
        for i, seg in enumerate(segments):
            if not seg.text.strip():
                continue

            seg_path = os.path.join(tmpdir, f"seg_{i:04d}.wav")
            tts.tts_to_file(
                text=seg.text,
                speaker_wav=reference_audio,
                language=seg.language,
                file_path=seg_path,
            )

            # Load synthesised segment and place it at the correct position
            from scipy.io.wavfile import read as wav_read

            seg_rate, seg_audio = wav_read(seg_path)
            if seg_audio.dtype != np.float32:
                seg_audio = seg_audio.astype(np.float32) / np.iinfo(seg_audio.dtype).max

            if seg_rate != sample_rate:
                # Resample via linear interpolation (simple fallback)
                duration_samples = len(seg_audio)
                original_times = np.linspace(0, 1, duration_samples)
                target_samples = int(duration_samples * sample_rate / seg_rate)
                target_times = np.linspace(0, 1, target_samples)
                seg_audio = np.interp(target_times, original_times, seg_audio).astype(np.float32)

            start_sample = int(seg.start * sample_rate)
            end_sample = start_sample + len(seg_audio)

            # Extend buffer if needed
            if end_sample > len(audio_buffer):
                audio_buffer = np.concatenate(
                    [audio_buffer, np.zeros(end_sample - len(audio_buffer), dtype=np.float32)]
                )

            audio_buffer[start_sample:end_sample] += seg_audio

    # Normalise to avoid clipping
    max_val = np.max(np.abs(audio_buffer))
    if max_val > 0:
        audio_buffer = audio_buffer / max_val * 0.95

    wav_write(output_path, sample_rate, audio_buffer)
    return output_path

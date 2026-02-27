"""Speech-to-text transcription using OpenAI Whisper.

Whisper returns timestamped segments which are needed so that the
synthesised audio can be placed at the correct position in the final video.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

try:
    import whisper
except ImportError:
    whisper = None  # type: ignore[assignment]


@dataclass
class Segment:
    """A single transcribed segment with its timing information."""

    start: float  # seconds
    end: float    # seconds
    text: str
    language: str


def transcribe(audio_path: str, model_name: str = "base") -> List[Segment]:
    """Transcribe *audio_path* and return a list of timed :class:`Segment` objects.

    Parameters
    ----------
    audio_path:
        Path to the audio file to transcribe (any format accepted by Whisper).
    model_name:
        Whisper model size.  Choices: ``tiny``, ``base``, ``small``,
        ``medium``, ``large``.  Larger models are more accurate but slower.

    Returns
    -------
    list[Segment]
        Chronologically ordered segments with start/end times and text.
    """
    if whisper is None:
        raise ImportError(
            "openai-whisper is not installed. Run: pip install openai-whisper"
        )

    model = whisper.load_model(model_name)
    result = model.transcribe(audio_path, task="transcribe")

    detected_language: str = result.get("language", "unknown")
    segments: List[Segment] = [
        Segment(
            start=seg["start"],
            end=seg["end"],
            text=seg["text"].strip(),
            language=detected_language,
        )
        for seg in result["segments"]
    ]
    return segments

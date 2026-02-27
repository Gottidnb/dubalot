"""
transcriber.py – speech-to-text using OpenAI Whisper.

Produces a list of ``TranscriptionSegment`` objects, each containing the
original text, start/end timestamps, and optional word-level timestamps used
for fine-grained audio alignment downstream.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class Word:
    """A single transcribed word with timing information."""

    text: str
    start: float
    end: float


@dataclass
class TranscriptionSegment:
    """One utterance / sentence from the transcription."""

    text: str
    start: float  # seconds
    end: float    # seconds
    words: List[Word] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return self.end - self.start


def transcribe(
    audio_path: Path,
    model_size: str = "base",
    language: Optional[str] = None,
    device: str = "cpu",
    word_timestamps: bool = True,
) -> List[TranscriptionSegment]:
    """Transcribe *audio_path* with Whisper and return timed segments.

    Args:
        audio_path: Path to a WAV (or any audio) file.
        model_size: Whisper model name – ``tiny``, ``base``, ``small``,
            ``medium``, or ``large-v3``.  Larger models are more accurate
            but slower.
        language: BCP-47 language code (e.g. ``"en"``, ``"fr"``) or ``None``
            / ``"auto"`` to let Whisper detect the language automatically.
        device: ``"cpu"`` or ``"cuda"``.
        word_timestamps: When ``True``, request per-word timestamps from
            Whisper (requires model ≥ ``base``).

    Returns:
        List of :class:`TranscriptionSegment` ordered by start time.

    Raises:
        FileNotFoundError: If *audio_path* does not exist.
        RuntimeError: If Whisper fails to transcribe.
    """
    import whisper  # type: ignore

    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    logger.info(
        "Loading Whisper model '%s' on %s", model_size, device
    )
    model = whisper.load_model(model_size, device=device)

    transcribe_kwargs: dict = {"word_timestamps": word_timestamps}
    if language and language.lower() != "auto":
        transcribe_kwargs["language"] = language

    logger.info("Transcribing %s", audio_path)
    result = model.transcribe(str(audio_path), **transcribe_kwargs)

    segments: List[TranscriptionSegment] = []
    for seg in result.get("segments", []):
        words: List[Word] = []
        if word_timestamps:
            for w in seg.get("words", []):
                words.append(Word(text=w["word"], start=w["start"], end=w["end"]))
        segments.append(
            TranscriptionSegment(
                text=seg["text"].strip(),
                start=seg["start"],
                end=seg["end"],
                words=words,
            )
        )

    logger.info(
        "Transcription complete: %d segment(s), detected language='%s'",
        len(segments),
        result.get("language"),
    )
    return segments

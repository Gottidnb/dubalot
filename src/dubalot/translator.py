"""
translator.py – translate transcription segments to a target language.

Uses ``deep-translator`` (GoogleTranslator backend by default) which works
without an API key for reasonable text volumes.  Long segments are chunked
to stay within the 5 000-character limit of the free Google Translate tier.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from typing import List

from .transcriber import TranscriptionSegment

logger = logging.getLogger(__name__)

# GoogleTranslator character limit per request
_CHUNK_LIMIT = 4900


def translate_segments(
    segments: List[TranscriptionSegment],
    target_language: str,
    source_language: str = "auto",
) -> List[TranscriptionSegment]:
    """Return a copy of *segments* with ``text`` translated to *target_language*.

    Timing information (``start``, ``end``, ``words``) is preserved unchanged
    so that the synthesiser can match the original pacing.

    Args:
        segments: Transcription segments produced by :func:`~dubalot.transcriber.transcribe`.
        target_language: BCP-47 / ISO 639-1 language code understood by
            Google Translate (e.g. ``"es"``, ``"fr"``, ``"de"``, ``"ja"``).
        source_language: Source language code or ``"auto"`` for auto-detection.

    Returns:
        New list of :class:`TranscriptionSegment` with translated text.

    Raises:
        ImportError: If ``deep-translator`` is not installed.
    """
    from deep_translator import GoogleTranslator  # type: ignore

    if not segments:
        return []

    logger.info(
        "Translating %d segment(s) from '%s' to '%s'",
        len(segments),
        source_language,
        target_language,
    )

    translator = GoogleTranslator(source=source_language, target=target_language)

    translated: List[TranscriptionSegment] = []
    for seg in segments:
        translated_text = _translate_text(translator, seg.text)
        new_seg = deepcopy(seg)
        new_seg.text = translated_text
        translated.append(new_seg)

    logger.info("Translation complete")
    return translated


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _translate_text(translator: object, text: str) -> str:
    """Translate *text*, chunking if it exceeds the service character limit."""
    if not text.strip():
        return text

    if len(text) <= _CHUNK_LIMIT:
        return translator.translate(text)  # type: ignore[union-attr]

    # Split on sentence boundaries and translate each chunk independently
    chunks = _chunk_text(text, _CHUNK_LIMIT)
    return " ".join(translator.translate(chunk) for chunk in chunks if chunk.strip())  # type: ignore[union-attr]


def _chunk_text(text: str, limit: int) -> List[str]:
    """Split *text* into chunks no longer than *limit* characters."""
    sentences = text.replace("! ", "!\n").replace(". ", ".\n").replace("? ", "?\n").split("\n")
    chunks: List[str] = []
    current = ""
    for sentence in sentences:
        if len(current) + len(sentence) + 1 > limit:
            if current:
                chunks.append(current.strip())
            current = sentence
        else:
            current = (current + " " + sentence).strip()
    if current:
        chunks.append(current.strip())
    return chunks

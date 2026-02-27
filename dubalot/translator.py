"""Text translation using deep-translator (Google Translate backend).

Each :class:`~dubalot.transcriber.Segment` produced by the transcription
step is translated independently so that the timing information is preserved.
"""

from __future__ import annotations

from typing import List

from dubalot.transcriber import Segment

try:
    from deep_translator import GoogleTranslator
except ImportError:
    GoogleTranslator = None  # type: ignore[assignment,misc]


def translate(segments: List[Segment], target_language: str) -> List[Segment]:
    """Translate the text of every *segment* into *target_language*.

    Parameters
    ----------
    segments:
        Segments returned by :func:`dubalot.transcriber.transcribe`.
    target_language:
        BCP-47 language tag accepted by Google Translate, e.g. ``"en"``,
        ``"es"``, ``"fr"``, ``"de"``, ``"ja"``.

    Returns
    -------
    list[Segment]
        New segment objects with translated text and updated language field.
    """
    if GoogleTranslator is None:
        raise ImportError(
            "deep-translator is not installed. Run: pip install deep-translator"
        )

    translator = GoogleTranslator(source="auto", target=target_language)

    translated: List[Segment] = []
    for seg in segments:
        translated_text = translator.translate(seg.text) if seg.text.strip() else seg.text
        translated.append(
            Segment(
                start=seg.start,
                end=seg.end,
                text=translated_text,
                language=target_language,
            )
        )
    return translated

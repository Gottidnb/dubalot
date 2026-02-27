"""Tests for dubalot.translator."""

from __future__ import annotations

import sys
from copy import deepcopy
from unittest.mock import MagicMock, patch

import pytest

from dubalot.transcriber import TranscriptionSegment
from dubalot.translator import _chunk_text, translate_segments


def _make_mock_deep_translator(side_effects):
    mock_translator_instance = MagicMock()
    mock_translator_instance.translate.side_effect = side_effects
    mock_cls = MagicMock(return_value=mock_translator_instance)
    mock_module = MagicMock()
    mock_module.GoogleTranslator = mock_cls
    return mock_module, mock_cls, mock_translator_instance


class TestChunkText:
    def test_short_text_is_single_chunk(self) -> None:
        result = _chunk_text("Hello world.", 100)
        assert result == ["Hello world."]

    def test_long_text_is_split(self) -> None:
        sentence = "This is a sentence. "
        long_text = sentence * 10
        chunks = _chunk_text(long_text, 50)
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk) <= 50 + len(sentence)

    def test_empty_text(self) -> None:
        assert _chunk_text("", 100) == []


class TestTranslateSegments:
    def _make_segments(self):
        return [
            TranscriptionSegment(text="Hello world", start=0.0, end=1.5),
            TranscriptionSegment(text="Goodbye", start=2.0, end=3.0),
        ]

    def test_translates_text(self) -> None:
        segments = self._make_segments()
        mock_module, _, _ = _make_mock_deep_translator(["Hola mundo", "Adiós"])

        with patch.dict(sys.modules, {"deep_translator": mock_module}):
            result = translate_segments(segments, target_language="es")

        assert result[0].text == "Hola mundo"
        assert result[1].text == "Adiós"

    def test_preserves_timestamps(self) -> None:
        segments = self._make_segments()
        mock_module, _, _ = _make_mock_deep_translator(["Hola mundo", "Adiós"])

        with patch.dict(sys.modules, {"deep_translator": mock_module}):
            result = translate_segments(segments, target_language="es")

        assert result[0].start == 0.0
        assert result[0].end == 1.5
        assert result[1].start == 2.0
        assert result[1].end == 3.0

    def test_does_not_modify_original(self) -> None:
        segments = self._make_segments()
        original_texts = [s.text for s in segments]
        mock_module, _, _ = _make_mock_deep_translator(["Hola mundo", "Adiós"])

        with patch.dict(sys.modules, {"deep_translator": mock_module}):
            translate_segments(segments, target_language="es")

        assert [s.text for s in segments] == original_texts

    def test_empty_segments_returns_empty(self) -> None:
        mock_module, _, _ = _make_mock_deep_translator([])
        with patch.dict(sys.modules, {"deep_translator": mock_module}):
            result = translate_segments([], target_language="es")
        assert result == []

    def test_target_language_passed_to_translator(self) -> None:
        segments = self._make_segments()
        mock_module, mock_cls, _ = _make_mock_deep_translator(["t1", "t2"])

        with patch.dict(sys.modules, {"deep_translator": mock_module}):
            translate_segments(segments, target_language="ja", source_language="en")
            mock_cls.assert_called_once_with(source="en", target="ja")


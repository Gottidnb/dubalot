"""Tests for dubalot.transcriber."""

from unittest.mock import MagicMock, patch

import pytest

from dubalot.transcriber import Segment, transcribe


def _make_whisper_result(segments, language="en"):
    return {
        "language": language,
        "segments": [
            {"start": s["start"], "end": s["end"], "text": s["text"]}
            for s in segments
        ],
    }


class TestSegment:
    def test_fields(self):
        seg = Segment(start=0.0, end=2.5, text="Hello world", language="en")
        assert seg.start == 0.0
        assert seg.end == 2.5
        assert seg.text == "Hello world"
        assert seg.language == "en"

    def test_equality(self):
        a = Segment(0.0, 1.0, "hi", "en")
        b = Segment(0.0, 1.0, "hi", "en")
        assert a == b

    def test_inequality(self):
        a = Segment(0.0, 1.0, "hi", "en")
        b = Segment(0.0, 1.0, "bye", "en")
        assert a != b


class TestTranscribe:
    @patch("dubalot.transcriber.whisper")
    def test_returns_segments(self, mock_whisper):
        raw = [
            {"start": 0.0, "end": 1.5, "text": "  Hello  "},
            {"start": 1.5, "end": 3.0, "text": " World "},
        ]
        mock_model = MagicMock()
        mock_model.transcribe.return_value = _make_whisper_result(raw, language="en")
        mock_whisper.load_model.return_value = mock_model

        result = transcribe("fake.wav", model_name="base")

        assert len(result) == 2
        assert result[0] == Segment(start=0.0, end=1.5, text="Hello", language="en")
        assert result[1] == Segment(start=1.5, end=3.0, text="World", language="en")

    @patch("dubalot.transcriber.whisper")
    def test_detects_language(self, mock_whisper):
        raw = [{"start": 0.0, "end": 2.0, "text": "Hola"}]
        mock_model = MagicMock()
        mock_model.transcribe.return_value = _make_whisper_result(raw, language="es")
        mock_whisper.load_model.return_value = mock_model

        result = transcribe("fake.wav")
        assert result[0].language == "es"

    @patch("dubalot.transcriber.whisper")
    def test_empty_audio(self, mock_whisper):
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"language": "en", "segments": []}
        mock_whisper.load_model.return_value = mock_model

        result = transcribe("silent.wav")
        assert result == []

    @patch("dubalot.transcriber.whisper")
    def test_model_size_passed(self, mock_whisper):
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"language": "en", "segments": []}
        mock_whisper.load_model.return_value = mock_model

        transcribe("fake.wav", model_name="small")
        mock_whisper.load_model.assert_called_once_with("small")

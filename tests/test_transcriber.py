"""Tests for dubalot.transcriber."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from dubalot.transcriber import TranscriptionSegment, Word, transcribe


class TestTranscriptionSegment:
    def test_duration(self) -> None:
        seg = TranscriptionSegment(text="hello", start=1.0, end=3.5)
        assert seg.duration == pytest.approx(2.5)

    def test_words_default_empty(self) -> None:
        seg = TranscriptionSegment(text="hi", start=0.0, end=1.0)
        assert seg.words == []


class TestTranscribe:
    def test_raises_if_audio_missing(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            transcribe(tmp_path / "nonexistent.wav")

    def _make_mock_whisper(self, result: dict) -> MagicMock:
        mock_model = MagicMock()
        mock_model.transcribe.return_value = result
        mock_whisper = MagicMock()
        mock_whisper.load_model.return_value = mock_model
        return mock_whisper

    def test_returns_segments(self, tmp_wav: Path) -> None:
        fake_result = {
            "language": "en",
            "segments": [
                {
                    "text": " Hello world",
                    "start": 0.0,
                    "end": 1.5,
                    "words": [
                        {"word": "Hello", "start": 0.0, "end": 0.7},
                        {"word": "world", "start": 0.8, "end": 1.5},
                    ],
                },
                {
                    "text": " Goodbye",
                    "start": 2.0,
                    "end": 3.0,
                    "words": [{"word": "Goodbye", "start": 2.0, "end": 3.0}],
                },
            ],
        }
        mock_whisper = self._make_mock_whisper(fake_result)

        with patch.dict(sys.modules, {"whisper": mock_whisper}):
            segments = transcribe(tmp_wav, model_size="base")

        assert len(segments) == 2
        assert segments[0].text == "Hello world"
        assert segments[0].start == 0.0
        assert segments[0].end == 1.5
        assert len(segments[0].words) == 2
        assert segments[0].words[0] == Word(text="Hello", start=0.0, end=0.7)
        assert segments[1].text == "Goodbye"

    def test_empty_segments(self, tmp_wav: Path) -> None:
        mock_whisper = self._make_mock_whisper({"language": "en", "segments": []})
        with patch.dict(sys.modules, {"whisper": mock_whisper}):
            segments = transcribe(tmp_wav)
        assert segments == []

    def test_language_passed_to_whisper(self, tmp_wav: Path) -> None:
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"language": "fr", "segments": []}
        mock_whisper = MagicMock()
        mock_whisper.load_model.return_value = mock_model

        with patch.dict(sys.modules, {"whisper": mock_whisper}):
            transcribe(tmp_wav, language="fr")
            _, kwargs = mock_model.transcribe.call_args
            assert kwargs.get("language") == "fr"

    def test_auto_language_not_passed(self, tmp_wav: Path) -> None:
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"language": "en", "segments": []}
        mock_whisper = MagicMock()
        mock_whisper.load_model.return_value = mock_model

        with patch.dict(sys.modules, {"whisper": mock_whisper}):
            transcribe(tmp_wav, language="auto")
            _, kwargs = mock_model.transcribe.call_args
            assert "language" not in kwargs


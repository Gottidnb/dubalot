"""
Tests for dubalot.pipeline.DubalotPipeline.

All heavyweight dependencies (Whisper, deep-translator, VoiceCloner,
LipSyncer) are mocked so the tests run without model downloads or
external processes.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch, call

import pytest

from dubalot.pipeline import DubalotPipeline
from dubalot.voice_clone import VoiceCloner
from dubalot.lip_sync import LipSyncer


# ---------------------------------------------------------------------------
# DubalotPipeline.__init__
# ---------------------------------------------------------------------------

class TestDubalotPipelineInit:
    def test_defaults(self):
        p = DubalotPipeline()
        assert p.target_language == "en"
        assert p.whisper_model == "base"
        assert isinstance(p.voice_cloner, VoiceCloner)
        assert isinstance(p.lip_syncer, LipSyncer)

    def test_custom_language(self):
        p = DubalotPipeline(target_language="es")
        assert p.target_language == "es"

    def test_custom_components(self):
        vc = VoiceCloner(device="cpu")
        ls = LipSyncer()
        p = DubalotPipeline(voice_cloner=vc, lip_syncer=ls)
        assert p.voice_cloner is vc
        assert p.lip_syncer is ls


# ---------------------------------------------------------------------------
# DubalotPipeline.translate
# ---------------------------------------------------------------------------

class TestTranslate:
    def test_same_language_returns_original(self):
        p = DubalotPipeline(target_language="en")
        result = p.translate("Hello", source="en", target="en")
        assert result == "Hello"

    def test_calls_google_translator(self):
        mock_translator_instance = MagicMock()
        mock_translator_instance.translate.return_value = "Hola"
        mock_translator_cls = MagicMock(return_value=mock_translator_instance)

        mock_deep_translator = MagicMock()
        mock_deep_translator.GoogleTranslator = mock_translator_cls

        with patch.dict("sys.modules", {"deep_translator": mock_deep_translator}):
            p = DubalotPipeline()
            result = p.translate("Hello", source="en", target="es")
        assert result == "Hola"

    def test_long_text_is_chunked(self):
        long_text = "a" * 10000
        mock_translator_instance = MagicMock()
        mock_translator_instance.translate.side_effect = lambda t: t.upper()
        mock_translator_cls = MagicMock(return_value=mock_translator_instance)

        mock_deep_translator = MagicMock()
        mock_deep_translator.GoogleTranslator = mock_translator_cls

        with patch.dict("sys.modules", {"deep_translator": mock_deep_translator}):
            p = DubalotPipeline()
            result = p.translate(long_text, source="en", target="fr")
        # Should have been called multiple times for chunks
        assert mock_translator_instance.translate.call_count > 1
        # Result should consist entirely of upper-cased content (A's and spaces from joining)
        assert result.replace(" ", "") == "A" * 10000


# ---------------------------------------------------------------------------
# DubalotPipeline.transcribe (public wrapper)
# ---------------------------------------------------------------------------

class TestTranscribe:
    def test_delegates_to_whisper(self, tmp_path):
        audio = tmp_path / "audio.wav"
        audio.write_bytes(b"\x00")

        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"text": " Hello world", "language": "en"}

        mock_whisper = MagicMock()
        mock_whisper.load_model.return_value = mock_model

        p = DubalotPipeline()
        with patch.dict("sys.modules", {"whisper": mock_whisper}):
            text, lang = p.transcribe(str(audio))
        assert text == "Hello world"
        assert lang == "en"


# ---------------------------------------------------------------------------
# DubalotPipeline.run – full pipeline
# ---------------------------------------------------------------------------

class TestPipelineRun:
    def test_missing_video_raises(self):
        p = DubalotPipeline()
        with pytest.raises(FileNotFoundError, match="Video file"):
            p.run("/nonexistent/video.mp4", "/tmp/out.mp4")

    def test_full_pipeline_calls_each_stage(self, tmp_path):
        video = tmp_path / "video.mp4"
        video.write_bytes(b"\x00")
        out = tmp_path / "out.mp4"

        mock_vc = MagicMock(spec=VoiceCloner)
        mock_vc.extract_reference_audio.return_value = str(tmp_path / "ref.wav")
        mock_vc.synthesise.return_value = str(tmp_path / "dubbed.wav")

        mock_ls = MagicMock(spec=LipSyncer)
        mock_ls.sync.return_value = str(out)

        p = DubalotPipeline(
            target_language="es",
            voice_cloner=mock_vc,
            lip_syncer=mock_ls,
        )
        p._transcribe = MagicMock(return_value=("Hello world", "en"))
        p._translate = MagicMock(return_value="Hola mundo")

        result = p.run(str(video), str(out))

        # Each stage must be called exactly once
        mock_vc.extract_reference_audio.assert_called_once()
        p._transcribe.assert_called_once()
        p._translate.assert_called_once_with("Hello world", "en", "es")
        mock_vc.synthesise.assert_called_once()
        mock_ls.sync.assert_called_once()
        assert result == os.path.abspath(str(out))

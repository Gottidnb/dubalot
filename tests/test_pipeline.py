"""Tests for the end-to-end pipeline (dubalot.pipeline)."""

import os
from unittest.mock import MagicMock, patch, call

import pytest

from dubalot.pipeline import translate_video
from dubalot.transcriber import Segment


SAMPLE_SEGMENTS = [
    Segment(0.0, 2.0, "Hola, ¿cómo estás?", "es"),
    Segment(2.0, 4.0, "Estoy bien, gracias.", "es"),
]

TRANSLATED_SEGMENTS = [
    Segment(0.0, 2.0, "Hello, how are you?", "en"),
    Segment(2.0, 4.0, "I am fine, thank you.", "en"),
]


class TestTranslateVideo:
    def test_raises_for_missing_input(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            translate_video(
                str(tmp_path / "missing.mp4"),
                str(tmp_path / "out.mp4"),
                target_language="en",
            )

    @patch("dubalot.pipeline.merge_audio")
    @patch("dubalot.pipeline.synthesize")
    @patch("dubalot.pipeline.translate")
    @patch("dubalot.pipeline.transcribe")
    @patch("dubalot.pipeline.extract_audio")
    def test_full_pipeline(
        self,
        mock_extract,
        mock_transcribe,
        mock_translate,
        mock_synthesize,
        mock_merge,
        tmp_path,
    ):
        # Create a fake input video file
        input_video = tmp_path / "input.mp4"
        input_video.touch()
        output_video = str(tmp_path / "output.mp4")

        mock_extract.return_value = "/tmp/audio.wav"
        mock_transcribe.return_value = SAMPLE_SEGMENTS
        mock_translate.return_value = TRANSLATED_SEGMENTS
        mock_synthesize.return_value = "/tmp/dubbed.wav"
        mock_merge.return_value = output_video

        result = translate_video(
            str(input_video),
            output_video,
            target_language="en",
        )

        assert result == output_video
        mock_extract.assert_called_once()
        mock_transcribe.assert_called_once()
        mock_translate.assert_called_once_with(SAMPLE_SEGMENTS, target_language="en")
        mock_synthesize.assert_called_once()
        mock_merge.assert_called_once()

    @patch("dubalot.pipeline.merge_audio")
    @patch("dubalot.pipeline.synthesize")
    @patch("dubalot.pipeline.translate")
    @patch("dubalot.pipeline.transcribe")
    @patch("dubalot.pipeline.extract_audio")
    def test_uses_reference_audio_when_provided(
        self,
        mock_extract,
        mock_transcribe,
        mock_translate,
        mock_synthesize,
        mock_merge,
        tmp_path,
    ):
        input_video = tmp_path / "input.mp4"
        input_video.touch()
        ref_audio = str(tmp_path / "ref.wav")

        mock_extract.return_value = "/tmp/audio.wav"
        mock_transcribe.return_value = SAMPLE_SEGMENTS
        mock_translate.return_value = TRANSLATED_SEGMENTS
        mock_synthesize.return_value = "/tmp/dubbed.wav"
        mock_merge.return_value = str(tmp_path / "output.mp4")

        translate_video(
            str(input_video),
            str(tmp_path / "output.mp4"),
            target_language="en",
            reference_audio=ref_audio,
        )

        # When reference_audio is provided it should be passed to synthesize
        mock_synthesize.assert_called_once()
        call_kwargs = mock_synthesize.call_args
        assert call_kwargs.kwargs.get("reference_audio") == ref_audio

    @patch("dubalot.pipeline.merge_audio")
    @patch("dubalot.pipeline.synthesize")
    @patch("dubalot.pipeline.translate")
    @patch("dubalot.pipeline.transcribe")
    @patch("dubalot.pipeline.extract_audio")
    def test_whisper_model_forwarded(
        self,
        mock_extract,
        mock_transcribe,
        mock_translate,
        mock_synthesize,
        mock_merge,
        tmp_path,
    ):
        input_video = tmp_path / "input.mp4"
        input_video.touch()

        mock_extract.return_value = "/tmp/audio.wav"
        mock_transcribe.return_value = []
        mock_translate.return_value = []
        mock_synthesize.return_value = "/tmp/dubbed.wav"
        mock_merge.return_value = str(tmp_path / "output.mp4")

        translate_video(
            str(input_video),
            str(tmp_path / "output.mp4"),
            target_language="en",
            whisper_model="small",
        )

        mock_transcribe.assert_called_once()
        _, kwargs = mock_transcribe.call_args
        assert kwargs.get("model_name") == "small"

"""Tests for dubalot.__main__ (CLI)."""

from unittest.mock import patch

import pytest

from dubalot.__main__ import main, _build_parser


class TestParser:
    def test_required_args(self):
        parser = _build_parser()
        args = parser.parse_args(["in.mp4", "out.mp4", "--target-language", "en"])
        assert args.input_video == "in.mp4"
        assert args.output_video == "out.mp4"
        assert args.target_language == "en"

    def test_default_whisper_model(self):
        parser = _build_parser()
        args = parser.parse_args(["in.mp4", "out.mp4", "--target-language", "en"])
        assert args.whisper_model == "base"

    def test_custom_whisper_model(self):
        parser = _build_parser()
        args = parser.parse_args(["in.mp4", "out.mp4", "--target-language", "es", "--whisper-model", "small"])
        assert args.whisper_model == "small"

    def test_invalid_whisper_model(self):
        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["in.mp4", "out.mp4", "--target-language", "en", "--whisper-model", "xlarge"])

    def test_reference_audio_default_none(self):
        parser = _build_parser()
        args = parser.parse_args(["in.mp4", "out.mp4", "--target-language", "en"])
        assert args.reference_audio is None

    def test_reference_audio_provided(self):
        parser = _build_parser()
        args = parser.parse_args(["in.mp4", "out.mp4", "--target-language", "en", "--reference-audio", "ref.wav"])
        assert args.reference_audio == "ref.wav"


class TestMain:
    @patch("dubalot.__main__.translate_video")
    def test_success_returns_0(self, mock_translate):
        mock_translate.return_value = "out.mp4"
        rc = main(["in.mp4", "out.mp4", "--target-language", "en"])
        assert rc == 0

    @patch("dubalot.__main__.translate_video")
    def test_file_not_found_returns_1(self, mock_translate):
        mock_translate.side_effect = FileNotFoundError("Input video not found: in.mp4")
        rc = main(["in.mp4", "out.mp4", "--target-language", "en"])
        assert rc == 1

    @patch("dubalot.__main__.translate_video")
    def test_value_error_returns_1(self, mock_translate):
        mock_translate.side_effect = ValueError("No segments provided")
        rc = main(["in.mp4", "out.mp4", "--target-language", "en"])
        assert rc == 1

    @patch("dubalot.__main__.translate_video")
    def test_passes_args_to_pipeline(self, mock_translate):
        mock_translate.return_value = "out.mp4"
        main([
            "in.mp4", "out.mp4",
            "--target-language", "fr",
            "--whisper-model", "medium",
            "--reference-audio", "ref.wav",
        ])
        mock_translate.assert_called_once_with(
            input_video="in.mp4",
            output_video="out.mp4",
            target_language="fr",
            whisper_model="medium",
            tts_model="tts_models/multilingual/multi-dataset/xtts_v2",
            reference_audio="ref.wav",
        )

"""Tests for dubalot.video_processor."""

import os
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from dubalot.video_processor import extract_audio, merge_audio


class TestExtractAudio:
    @patch("dubalot.video_processor.VideoFileClip")
    def test_extracts_audio(self, MockClip):
        mock_clip = MagicMock()
        mock_audio = MagicMock()
        mock_clip.audio = mock_audio
        MockClip.return_value = mock_clip

        result = extract_audio("video.mp4", "/tmp/out.wav")

        mock_audio.write_audiofile.assert_called_once_with("/tmp/out.wav", logger=None)
        mock_clip.close.assert_called_once()
        assert result == "/tmp/out.wav"

    @patch("dubalot.video_processor.VideoFileClip")
    def test_raises_when_no_audio(self, MockClip):
        mock_clip = MagicMock()
        mock_clip.audio = None
        MockClip.return_value = mock_clip

        with pytest.raises(ValueError, match="no audio track"):
            extract_audio("silent_video.mp4", "/tmp/out.wav")


class TestMergeAudio:
    @patch("dubalot.video_processor.AudioFileClip")
    @patch("dubalot.video_processor.VideoFileClip")
    def test_merges_audio(self, MockVideoClip, MockAudioClip):
        mock_video = MagicMock()
        mock_audio = MagicMock()
        mock_final = MagicMock()

        MockVideoClip.return_value = mock_video
        MockAudioClip.return_value = mock_audio
        mock_video.set_audio.return_value = mock_final

        result = merge_audio("video.mp4", "dubbed.wav", "/tmp/result.mp4")

        mock_video.set_audio.assert_called_once_with(mock_audio)
        mock_final.write_videofile.assert_called_once_with(
            "/tmp/result.mp4", codec="libx264", audio_codec="aac", logger=None
        )
        mock_video.close.assert_called_once()
        mock_audio.close.assert_called_once()
        mock_final.close.assert_called_once()
        assert result == "/tmp/result.mp4"

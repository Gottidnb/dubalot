"""
Tests for video_editor module.
"""

import os
import subprocess
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from video_editor import extract_audio, replace_audio, trim_video


class TestExtractAudio(unittest.TestCase):

    def test_raises_if_video_not_found(self):
        with self.assertRaises(FileNotFoundError):
            extract_audio("/nonexistent/video.mp4")

    def test_uses_provided_output_path(self):
        with (
            tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as video_f,
            tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as audio_f,
        ):
            video_path = video_f.name
            audio_path = audio_f.name

        try:
            mock_result = MagicMock(returncode=0)
            with patch("subprocess.run", return_value=mock_result) as mock_run:
                result = extract_audio(video_path, audio_path)
                self.assertEqual(result, audio_path)
                args = mock_run.call_args[0][0]
                self.assertIn(audio_path, args)
        finally:
            os.unlink(video_path)
            os.unlink(audio_path)

    def test_creates_temp_file_when_no_output_path(self):
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            video_path = f.name

        try:
            mock_result = MagicMock(returncode=0)
            with patch("subprocess.run", return_value=mock_result):
                result = extract_audio(video_path)
                self.assertTrue(result.endswith(".wav"))
                os.unlink(result)
        finally:
            os.unlink(video_path)

    def test_raises_on_ffmpeg_failure(self):
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            video_path = f.name

        try:
            mock_result = MagicMock(returncode=1, stderr="ffmpeg error")
            with patch("subprocess.run", return_value=mock_result):
                with self.assertRaises(RuntimeError):
                    extract_audio(video_path)
        finally:
            os.unlink(video_path)

    def test_ffmpeg_command_includes_no_video_flag(self):
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            video_path = f.name

        try:
            mock_result = MagicMock(returncode=0)
            with patch("subprocess.run", return_value=mock_result) as mock_run:
                extract_audio(video_path, "/tmp/out.wav")
                args = mock_run.call_args[0][0]
                self.assertIn("-vn", args)
                self.assertIn(video_path, args)
        finally:
            os.unlink(video_path)


class TestReplaceAudio(unittest.TestCase):

    def test_raises_if_video_not_found(self):
        with self.assertRaises(FileNotFoundError):
            replace_audio("/nonexistent/video.mp4", "/tmp/audio.wav", "/tmp/out.mp4")

    def test_raises_if_audio_not_found(self):
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            video_path = f.name

        try:
            with self.assertRaises(FileNotFoundError):
                replace_audio(video_path, "/nonexistent/audio.wav", "/tmp/out.mp4")
        finally:
            os.unlink(video_path)

    def test_returns_output_path(self):
        with (
            tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as video_f,
            tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as audio_f,
        ):
            video_path = video_f.name
            audio_path = audio_f.name

        try:
            mock_result = MagicMock(returncode=0)
            with patch("subprocess.run", return_value=mock_result):
                result = replace_audio(video_path, audio_path, "/tmp/out.mp4")
                self.assertEqual(result, "/tmp/out.mp4")
        finally:
            os.unlink(video_path)
            os.unlink(audio_path)

    def test_raises_on_ffmpeg_failure(self):
        with (
            tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as video_f,
            tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as audio_f,
        ):
            video_path = video_f.name
            audio_path = audio_f.name

        try:
            mock_result = MagicMock(returncode=1, stderr="ffmpeg error")
            with patch("subprocess.run", return_value=mock_result):
                with self.assertRaises(RuntimeError):
                    replace_audio(video_path, audio_path, "/tmp/out.mp4")
        finally:
            os.unlink(video_path)
            os.unlink(audio_path)

    def test_ffmpeg_command_maps_video_and_audio_streams(self):
        with (
            tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as video_f,
            tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as audio_f,
        ):
            video_path = video_f.name
            audio_path = audio_f.name

        try:
            mock_result = MagicMock(returncode=0)
            with patch("subprocess.run", return_value=mock_result) as mock_run:
                replace_audio(video_path, audio_path, "/tmp/out.mp4")
                args = mock_run.call_args[0][0]
                self.assertIn("-map", args)
                self.assertIn("0:v:0", args)
                self.assertIn("1:a:0", args)
        finally:
            os.unlink(video_path)
            os.unlink(audio_path)


class TestTrimVideo(unittest.TestCase):

    def test_raises_if_video_not_found(self):
        with self.assertRaises(FileNotFoundError):
            trim_video("/nonexistent/video.mp4", "/tmp/out.mp4", 0, 10)

    def test_raises_on_negative_start_time(self):
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            video_path = f.name

        try:
            with self.assertRaises(ValueError):
                trim_video(video_path, "/tmp/out.mp4", -1, 10)
        finally:
            os.unlink(video_path)

    def test_raises_when_end_time_not_greater_than_start_time(self):
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            video_path = f.name

        try:
            with self.assertRaises(ValueError):
                trim_video(video_path, "/tmp/out.mp4", 10, 5)
            with self.assertRaises(ValueError):
                trim_video(video_path, "/tmp/out.mp4", 10, 10)
        finally:
            os.unlink(video_path)

    def test_returns_output_path(self):
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            video_path = f.name

        try:
            mock_result = MagicMock(returncode=0)
            with patch("subprocess.run", return_value=mock_result):
                result = trim_video(video_path, "/tmp/out.mp4", 0, 30)
                self.assertEqual(result, "/tmp/out.mp4")
        finally:
            os.unlink(video_path)

    def test_raises_on_ffmpeg_failure(self):
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            video_path = f.name

        try:
            mock_result = MagicMock(returncode=1, stderr="ffmpeg error")
            with patch("subprocess.run", return_value=mock_result):
                with self.assertRaises(RuntimeError):
                    trim_video(video_path, "/tmp/out.mp4", 0, 30)
        finally:
            os.unlink(video_path)

    def test_no_end_time_omits_to_flag(self):
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            video_path = f.name

        try:
            mock_result = MagicMock(returncode=0)
            with patch("subprocess.run", return_value=mock_result) as mock_run:
                trim_video(video_path, "/tmp/out.mp4", 5)
                args = mock_run.call_args[0][0]
                self.assertNotIn("-to", args)
        finally:
            os.unlink(video_path)

    def test_end_time_includes_to_flag(self):
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            video_path = f.name

        try:
            mock_result = MagicMock(returncode=0)
            with patch("subprocess.run", return_value=mock_result) as mock_run:
                trim_video(video_path, "/tmp/out.mp4", 5, 30)
                args = mock_run.call_args[0][0]
                self.assertIn("-to", args)
                self.assertIn("30", args)
        finally:
            os.unlink(video_path)


if __name__ == "__main__":
    unittest.main()

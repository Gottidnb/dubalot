"""
Tests for dubalot.voice_clone.VoiceCloner.

Heavy ML dependencies (TTS, torch, ffmpeg) are mocked so the tests run
without downloading multi-GB model weights.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from dubalot.voice_clone import VoiceCloner


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_dummy_wav(path: str) -> None:
    """Write a minimal valid WAV header so os.path.isfile passes."""
    with open(path, "wb") as f:
        # 44-byte PCM WAV header with 0 samples
        f.write(
            b"RIFF$\x00\x00\x00WAVEfmt "
            b"\x10\x00\x00\x00\x01\x00\x01\x00"
            b"D\xac\x00\x00\x88X\x01\x00\x02\x00\x10\x00"
            b"data\x00\x00\x00\x00"
        )


# ---------------------------------------------------------------------------
# VoiceCloner.__init__
# ---------------------------------------------------------------------------

class TestVoiceClonerInit:
    def test_default_model(self):
        vc = VoiceCloner()
        assert vc.model_name == VoiceCloner.DEFAULT_MODEL

    def test_custom_model(self):
        vc = VoiceCloner(model_name="my/custom/model")
        assert vc.model_name == "my/custom/model"

    def test_explicit_device(self):
        vc = VoiceCloner(device="cpu")
        assert vc.device == "cpu"

    @patch("dubalot.voice_clone.VoiceCloner._default_device", return_value="cpu")
    def test_auto_device(self, _mock):
        vc = VoiceCloner()
        assert vc.device == "cpu"

    def test_tts_not_loaded_at_init(self):
        vc = VoiceCloner()
        assert vc._tts is None


# ---------------------------------------------------------------------------
# VoiceCloner.synthesise
# ---------------------------------------------------------------------------

class TestVoiceClonerSynthesise:
    def test_missing_reference_audio_raises(self):
        vc = VoiceCloner()
        with pytest.raises(FileNotFoundError, match="Reference audio"):
            vc.synthesise(
                text="Hello",
                reference_audio="/nonexistent/ref.wav",
                output_path="/tmp/out.wav",
            )

    def test_empty_text_raises(self, tmp_path):
        ref = tmp_path / "ref.wav"
        _make_dummy_wav(str(ref))
        vc = VoiceCloner()
        with pytest.raises(ValueError, match="empty"):
            vc.synthesise(
                text="   ",
                reference_audio=str(ref),
                output_path=str(tmp_path / "out.wav"),
            )

    def test_synthesise_calls_tts(self, tmp_path):
        ref = tmp_path / "ref.wav"
        _make_dummy_wav(str(ref))
        out = tmp_path / "out.wav"

        mock_tts = MagicMock()
        mock_tts.tts_to_file = MagicMock()

        vc = VoiceCloner(device="cpu")
        vc._tts = mock_tts  # inject mock – skips model download

        result = vc.synthesise(
            text="Hello world",
            reference_audio=str(ref),
            output_path=str(out),
            language="en",
        )

        mock_tts.tts_to_file.assert_called_once_with(
            text="Hello world",
            speaker_wav=str(ref),
            language="en",
            file_path=str(out),
        )
        assert result == os.path.abspath(str(out))

    def test_synthesise_returns_abs_path(self, tmp_path):
        ref = tmp_path / "ref.wav"
        _make_dummy_wav(str(ref))
        out = tmp_path / "out.wav"

        mock_tts = MagicMock()
        vc = VoiceCloner(device="cpu")
        vc._tts = mock_tts

        result = vc.synthesise("Hi", str(ref), str(out))
        assert os.path.isabs(result)


# ---------------------------------------------------------------------------
# VoiceCloner.extract_reference_audio
# ---------------------------------------------------------------------------

class TestExtractReferenceAudio:
    def test_missing_video_raises(self):
        vc = VoiceCloner()
        with pytest.raises(FileNotFoundError, match="Video file"):
            vc.extract_reference_audio("/nonexistent/video.mp4")

    def test_auto_creates_tmp_file(self, tmp_path):
        video = tmp_path / "video.mp4"
        video.write_bytes(b"\x00")  # dummy file

        mock_ffmpeg = MagicMock()
        mock_stream = MagicMock()
        mock_ffmpeg.input.return_value = mock_stream
        mock_stream.output.return_value.overwrite_output.return_value.run = MagicMock()

        with patch.dict("sys.modules", {"ffmpeg": mock_ffmpeg}):
            vc = VoiceCloner()
            result = vc.extract_reference_audio(str(video))

        assert result.endswith(".wav")
        assert os.path.isabs(result)

    def test_custom_output_path(self, tmp_path):
        video = tmp_path / "video.mp4"
        video.write_bytes(b"\x00")
        out = tmp_path / "ref.wav"

        mock_ffmpeg = MagicMock()
        mock_stream = MagicMock()
        mock_ffmpeg.input.return_value = mock_stream
        mock_stream.output.return_value.overwrite_output.return_value.run = MagicMock()

        with patch.dict("sys.modules", {"ffmpeg": mock_ffmpeg}):
            vc = VoiceCloner()
            result = vc.extract_reference_audio(str(video), output_path=str(out))

        assert result == os.path.abspath(str(out))


# ---------------------------------------------------------------------------
# VoiceCloner._default_device
# ---------------------------------------------------------------------------

class TestDefaultDevice:
    def test_returns_cuda_when_available(self):
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        with patch.dict("sys.modules", {"torch": mock_torch}):
            assert VoiceCloner._default_device() == "cuda"

    def test_returns_cpu_when_cuda_unavailable(self):
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        with patch.dict("sys.modules", {"torch": mock_torch}):
            assert VoiceCloner._default_device() == "cpu"

    def test_returns_cpu_when_torch_missing(self):
        with patch.dict("sys.modules", {"torch": None}):
            assert VoiceCloner._default_device() == "cpu"

"""
Tests for dubalot.lip_sync.LipSyncer.

subprocess calls (Wav2Lip, ffmpeg) and filesystem calls are mocked so the
tests run without any external binaries.
"""

from __future__ import annotations

import os
import subprocess
from unittest.mock import MagicMock, patch, call

import pytest

from dubalot.lip_sync import LipSyncer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_dummy_file(path: str) -> None:
    with open(path, "wb") as f:
        f.write(b"\x00")


# ---------------------------------------------------------------------------
# LipSyncer.__init__
# ---------------------------------------------------------------------------

class TestLipSyncerInit:
    def test_defaults(self):
        ls = LipSyncer()
        assert ls.wav2lip_checkpoint is None
        assert ls.wav2lip_script is None
        assert ls.face_det_checkpoint is None

    def test_custom_paths(self, tmp_path):
        ckpt = str(tmp_path / "model.pth")
        script = str(tmp_path / "inference.py")
        ls = LipSyncer(wav2lip_checkpoint=ckpt, wav2lip_script=script)
        assert ls.wav2lip_checkpoint == ckpt
        assert ls.wav2lip_script == script


# ---------------------------------------------------------------------------
# LipSyncer.sync – input validation
# ---------------------------------------------------------------------------

class TestLipSyncerInputValidation:
    def test_missing_video_raises(self, tmp_path):
        audio = tmp_path / "audio.wav"
        _make_dummy_file(str(audio))
        ls = LipSyncer()
        with pytest.raises(FileNotFoundError, match="Video file"):
            ls.sync("/nonexistent/video.mp4", str(audio), str(tmp_path / "out.mp4"))

    def test_missing_audio_raises(self, tmp_path):
        video = tmp_path / "video.mp4"
        _make_dummy_file(str(video))
        ls = LipSyncer()
        with pytest.raises(FileNotFoundError, match="Audio file"):
            ls.sync(str(video), "/nonexistent/audio.wav", str(tmp_path / "out.mp4"))


# ---------------------------------------------------------------------------
# LipSyncer._wav2lip_available
# ---------------------------------------------------------------------------

class TestWav2LipAvailable:
    def test_false_when_no_checkpoint(self):
        ls = LipSyncer()
        assert ls._wav2lip_available() is False

    def test_false_when_checkpoint_missing_from_disk(self, tmp_path):
        ls = LipSyncer(
            wav2lip_checkpoint=str(tmp_path / "model.pth"),
            wav2lip_script=str(tmp_path / "inference.py"),
        )
        assert ls._wav2lip_available() is False

    def test_true_when_both_files_exist(self, tmp_path):
        ckpt = tmp_path / "model.pth"
        script = tmp_path / "inference.py"
        ckpt.write_bytes(b"\x00")
        script.write_bytes(b"\x00")
        ls = LipSyncer(
            wav2lip_checkpoint=str(ckpt), wav2lip_script=str(script)
        )
        assert ls._wav2lip_available() is True


# ---------------------------------------------------------------------------
# LipSyncer._sync_wav2lip
# ---------------------------------------------------------------------------

class TestSyncWav2Lip:
    def test_calls_subprocess(self, tmp_path):
        ckpt = tmp_path / "model.pth"
        script = tmp_path / "inference.py"
        video = tmp_path / "video.mp4"
        audio = tmp_path / "audio.wav"
        out = tmp_path / "out.mp4"
        for p in (ckpt, script, video, audio):
            p.write_bytes(b"\x00")

        mock_result = MagicMock()
        mock_result.returncode = 0

        with patch("subprocess.run", return_value=mock_result) as mock_run:
            ls = LipSyncer(wav2lip_checkpoint=str(ckpt), wav2lip_script=str(script))
            result = ls._sync_wav2lip(str(video), str(audio), str(out))

        cmd_args = mock_run.call_args[0][0]
        assert "--checkpoint_path" in cmd_args
        assert "--face" in cmd_args
        assert "--audio" in cmd_args
        assert "--outfile" in cmd_args
        assert result == os.path.abspath(str(out))

    def test_raises_on_nonzero_exit(self, tmp_path):
        ckpt = tmp_path / "model.pth"
        script = tmp_path / "inference.py"
        video = tmp_path / "video.mp4"
        audio = tmp_path / "audio.wav"
        out = tmp_path / "out.mp4"
        for p in (ckpt, script, video, audio):
            p.write_bytes(b"\x00")

        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "CUDA error"

        with patch("subprocess.run", return_value=mock_result):
            ls = LipSyncer(wav2lip_checkpoint=str(ckpt), wav2lip_script=str(script))
            with pytest.raises(RuntimeError, match="Wav2Lip inference failed"):
                ls._sync_wav2lip(str(video), str(audio), str(out))


# ---------------------------------------------------------------------------
# LipSyncer._sync_ffmpeg
# ---------------------------------------------------------------------------

class TestSyncFfmpeg:
    def test_raises_when_ffmpeg_not_on_path(self, tmp_path):
        with patch("shutil.which", return_value=None):
            with pytest.raises(EnvironmentError, match="ffmpeg not found"):
                LipSyncer._sync_ffmpeg("/v.mp4", "/a.wav", "/out.mp4")

    def test_calls_ffmpeg_with_correct_args(self, tmp_path):
        video = str(tmp_path / "video.mp4")
        audio = str(tmp_path / "audio.wav")
        out = str(tmp_path / "out.mp4")

        mock_result = MagicMock()
        mock_result.returncode = 0

        with patch("shutil.which", return_value="/usr/bin/ffmpeg"), \
             patch("subprocess.run", return_value=mock_result) as mock_run:
            result = LipSyncer._sync_ffmpeg(video, audio, out)

        cmd = mock_run.call_args[0][0]
        assert "ffmpeg" in cmd
        assert "-map" in cmd
        assert "0:v:0" in cmd
        assert "1:a:0" in cmd
        assert result == os.path.abspath(out)

    def test_raises_on_nonzero_exit(self):
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "No such file"

        with patch("shutil.which", return_value="/usr/bin/ffmpeg"), \
             patch("subprocess.run", return_value=mock_result):
            with pytest.raises(RuntimeError, match="ffmpeg audio replacement failed"):
                LipSyncer._sync_ffmpeg("/v.mp4", "/a.wav", "/out.mp4")


# ---------------------------------------------------------------------------
# LipSyncer.sync – routing logic
# ---------------------------------------------------------------------------

class TestSyncRouting:
    def test_uses_ffmpeg_fallback_when_wav2lip_unavailable(self, tmp_path):
        video = tmp_path / "video.mp4"
        audio = tmp_path / "audio.wav"
        out = tmp_path / "out.mp4"
        video.write_bytes(b"\x00")
        audio.write_bytes(b"\x00")

        with patch.object(LipSyncer, "_wav2lip_available", return_value=False), \
             patch.object(LipSyncer, "_sync_ffmpeg", return_value=str(out)) as mock_ffmpeg:
            ls = LipSyncer()
            result = ls.sync(str(video), str(audio), str(out))

        mock_ffmpeg.assert_called_once()
        assert result == str(out)

    def test_uses_wav2lip_when_available(self, tmp_path):
        video = tmp_path / "video.mp4"
        audio = tmp_path / "audio.wav"
        out = tmp_path / "out.mp4"
        video.write_bytes(b"\x00")
        audio.write_bytes(b"\x00")

        with patch.object(LipSyncer, "_wav2lip_available", return_value=True), \
             patch.object(LipSyncer, "_sync_wav2lip", return_value=str(out)) as mock_w2l:
            ls = LipSyncer()
            result = ls.sync(str(video), str(audio), str(out))

        mock_w2l.assert_called_once()
        assert result == str(out)

"""Tests for dubalot.extractor."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf

from dubalot.extractor import extract_audio, separate_audio_stems


# ---------------------------------------------------------------------------
# extract_audio
# ---------------------------------------------------------------------------


class TestExtractAudio:
    def test_raises_if_input_missing(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            extract_audio(tmp_path / "nonexistent.mp4", tmp_path / "out.wav")

    def test_ffmpeg_extraction(self, tmp_path: Path, tmp_wav: Path) -> None:
        """When ffmpeg succeeds it should produce the output file."""
        output = tmp_path / "extracted.wav"

        # Patch subprocess.run to simulate ffmpeg copying the tmp_wav
        def fake_ffmpeg(cmd, **_kwargs):
            shutil.copy2(str(tmp_wav), str(output))
            result = MagicMock()
            result.returncode = 0
            return result

        with patch("dubalot.extractor.subprocess.run", side_effect=fake_ffmpeg):
            result = extract_audio(tmp_wav, output)

        assert result == output
        assert output.exists()

    def test_falls_back_to_moviepy_if_ffmpeg_missing(
        self, tmp_path: Path, tmp_wav: Path
    ) -> None:
        """If ffmpeg is not on PATH, MoviePy fallback should be tried."""
        output = tmp_path / "extracted.wav"

        def raise_not_found(*_args, **_kwargs):
            raise FileNotFoundError("ffmpeg not found")

        mock_clip = MagicMock()
        mock_clip.__enter__ = MagicMock(return_value=mock_clip)
        mock_clip.__exit__ = MagicMock(return_value=False)
        mock_clip.audio = MagicMock()
        mock_clip.audio.write_audiofile = MagicMock(
            side_effect=lambda path, **kw: shutil.copy2(str(tmp_wav), path)
        )

        with (
            patch("dubalot.extractor.subprocess.run", side_effect=raise_not_found),
            patch("dubalot.extractor._moviepy_extract") as mock_mp,
        ):
            mock_mp.side_effect = lambda vp, op: shutil.copy2(str(tmp_wav), str(op))
            result = extract_audio(tmp_wav, output)

        assert result == output


# ---------------------------------------------------------------------------
# separate_audio_stems
# ---------------------------------------------------------------------------


class TestSeparateAudioStems:
    def test_fallback_when_demucs_absent(
        self, tmp_path: Path, tmp_wav: Path
    ) -> None:
        """Without Demucs the original path should be returned as vocals."""
        with (
            patch("dubalot.extractor.shutil.which", return_value=None),
            patch.dict("sys.modules", {"demucs": None}),
        ):
            vocals, background = separate_audio_stems(tmp_wav, tmp_path / "stems")

        assert vocals == tmp_wav
        assert background is None

    def test_demucs_cli_called(self, tmp_path: Path, tmp_wav: Path) -> None:
        """When demucs CLI is on PATH it should be invoked via subprocess."""
        stems_dir = tmp_path / "stems"
        vocals_dir = stems_dir / "htdemucs" / tmp_wav.stem
        vocals_dir.mkdir(parents=True)

        # Create expected output files so the path check passes
        fake_vocals = vocals_dir / "vocals.wav"
        fake_bg = vocals_dir / "no_vocals.wav"
        shutil.copy2(str(tmp_wav), str(fake_vocals))
        shutil.copy2(str(tmp_wav), str(fake_bg))

        def fake_run(cmd, **_kwargs):
            return MagicMock(returncode=0)

        with (
            patch("dubalot.extractor.shutil.which", return_value="/usr/bin/demucs"),
            patch("dubalot.extractor.subprocess.run", side_effect=fake_run),
        ):
            vocals, background = separate_audio_stems(tmp_wav, stems_dir)

        assert vocals == fake_vocals
        assert background == fake_bg

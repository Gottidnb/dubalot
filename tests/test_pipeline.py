"""Tests for dubalot.pipeline (integration-style, all external deps mocked)."""

from __future__ import annotations

import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf

from dubalot.pipeline import translate_video
from dubalot.transcriber import TranscriptionSegment


def _write_wav(path: Path, duration: float = 1.0, sr: int = 16000) -> Path:
    n = int(duration * sr)
    sf.write(str(path), np.zeros(n, dtype=np.float32), sr)
    return path


@pytest.fixture()
def fake_video(tmp_path: Path) -> Path:
    """Create a dummy file that stands in for an MP4."""
    p = tmp_path / "input.mp4"
    p.write_bytes(b"\x00" * 64)
    return p


class TestTranslateVideo:
    def test_raises_if_input_missing(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            translate_video(
                input_path=str(tmp_path / "nope.mp4"),
                output_path=str(tmp_path / "out.mp4"),
                target_language="es",
            )

    def test_full_pipeline_runs(self, tmp_path: Path, fake_video: Path) -> None:
        """Smoke test: all steps are mocked and the pipeline completes."""
        output = tmp_path / "output.mp4"
        vocals_wav = tmp_path / "vocals.wav"
        _write_wav(vocals_wav)

        segments = [TranscriptionSegment(text="Hello", start=0.0, end=1.0)]
        translated_segments = [TranscriptionSegment(text="Hola", start=0.0, end=1.0)]

        with (
            patch("dubalot.pipeline.extract_audio") as mock_extract,
            patch("dubalot.pipeline.separate_audio_stems") as mock_separate,
            patch("dubalot.pipeline.transcribe", return_value=segments),
            patch(
                "dubalot.pipeline.translate_segments",
                return_value=translated_segments,
            ),
            patch("dubalot.pipeline.synthesize_segments") as mock_synth,
            patch("dubalot.pipeline.mix_audio") as mock_mix,
            patch("dubalot.pipeline.apply_lip_sync") as mock_lipsync,
            patch("dubalot.pipeline._compose_final_video") as mock_compose,
        ):
            mock_extract.side_effect = lambda vp, ap: _write_wav(ap)
            mock_separate.return_value = (vocals_wav, None)
            mock_synth.side_effect = lambda **kw: _write_wav(kw["output_path"])
            mock_mix.side_effect = lambda **kw: _write_wav(kw["output_path"])
            mock_lipsync.return_value = fake_video
            mock_compose.return_value = None

            result = translate_video(
                input_path=str(fake_video),
                output_path=str(output),
                target_language="es",
                lip_sync=True,
            )

        mock_extract.assert_called_once()
        mock_separate.assert_called_once()
        assert result == str(output.resolve())

    def test_pipeline_without_lip_sync(self, tmp_path: Path, fake_video: Path) -> None:
        output = tmp_path / "output.mp4"
        vocals_wav = tmp_path / "vocals.wav"
        _write_wav(vocals_wav)

        segments = [TranscriptionSegment(text="Hi", start=0.0, end=0.5)]
        translated = [TranscriptionSegment(text="Hola", start=0.0, end=0.5)]

        with (
            patch("dubalot.pipeline.extract_audio") as mock_extract,
            patch("dubalot.pipeline.separate_audio_stems") as mock_separate,
            patch("dubalot.pipeline.transcribe", return_value=segments),
            patch("dubalot.pipeline.translate_segments", return_value=translated),
            patch("dubalot.pipeline.synthesize_segments") as mock_synth,
            patch("dubalot.pipeline.mix_audio") as mock_mix,
            patch("dubalot.pipeline.apply_lip_sync") as mock_lipsync,
            patch("dubalot.pipeline._compose_final_video") as mock_compose,
        ):
            mock_extract.side_effect = lambda vp, ap: _write_wav(ap)
            mock_separate.return_value = (vocals_wav, None)
            mock_synth.side_effect = lambda **kw: _write_wav(kw["output_path"])
            mock_mix.side_effect = lambda **kw: _write_wav(kw["output_path"])
            mock_compose.return_value = None

            translate_video(
                input_path=str(fake_video),
                output_path=str(output),
                target_language="es",
                lip_sync=False,
            )

        mock_lipsync.assert_not_called()

    def test_raises_if_no_transcription(
        self, tmp_path: Path, fake_video: Path
    ) -> None:
        vocals_wav = tmp_path / "vocals.wav"
        _write_wav(vocals_wav)

        with (
            patch("dubalot.pipeline.extract_audio") as mock_extract,
            patch("dubalot.pipeline.separate_audio_stems") as mock_separate,
            patch("dubalot.pipeline.transcribe", return_value=[]),
        ):
            mock_extract.side_effect = lambda vp, ap: _write_wav(ap)
            mock_separate.return_value = (vocals_wav, None)

            with pytest.raises(RuntimeError, match="no transcription segments"):
                translate_video(
                    input_path=str(fake_video),
                    output_path=str(tmp_path / "out.mp4"),
                    target_language="es",
                )

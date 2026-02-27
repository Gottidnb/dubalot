"""Tests for dubalot.synthesizer."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf

from dubalot.synthesizer import _fit_to_duration, synthesize_segments
from dubalot.transcriber import TranscriptionSegment


def _make_mock_tts_module():
    """Return a mock TTS module (TTS.api) and the TTS class instance."""
    mock_tts_instance = MagicMock()

    mock_tts_cls = MagicMock(return_value=mock_tts_instance)
    mock_api_module = MagicMock()
    mock_api_module.TTS = mock_tts_cls

    mock_tts_top = MagicMock()
    mock_tts_top.api = mock_api_module

    return mock_tts_top, mock_api_module, mock_tts_cls, mock_tts_instance


class TestFitToDuration:
    def test_no_stretch_within_tolerance(self) -> None:
        audio = np.ones(16000, dtype=np.float32)
        result = _fit_to_duration(audio, target_duration=1.0, sample_rate=16000)
        assert np.array_equal(result, audio)

    def test_stretches_audio(self) -> None:
        audio = np.ones(8000, dtype=np.float32)  # 0.5 s at 16 kHz
        result = _fit_to_duration(audio, target_duration=1.0, sample_rate=16000)
        assert len(result) == 16000

    def test_compresses_audio(self) -> None:
        audio = np.ones(32000, dtype=np.float32)  # 2 s at 16 kHz
        result = _fit_to_duration(audio, target_duration=1.0, sample_rate=16000)
        assert len(result) == 16000

    def test_zero_duration_audio_returned_unchanged(self) -> None:
        audio = np.array([], dtype=np.float32)
        result = _fit_to_duration(audio, target_duration=1.0, sample_rate=16000)
        assert len(result) == 0

    def test_zero_target_duration_returned_unchanged(self) -> None:
        audio = np.ones(16000, dtype=np.float32)
        result = _fit_to_duration(audio, target_duration=0.0, sample_rate=16000)
        assert np.array_equal(result, audio)


class TestSynthesizeSegments:
    def _make_segments(self):
        return [
            TranscriptionSegment(text="Hola mundo", start=0.0, end=1.5),
            TranscriptionSegment(text="Adiós", start=2.0, end=3.0),
        ]

    def test_writes_output_file(self, tmp_path: Path, tmp_wav: Path) -> None:
        segments = self._make_segments()
        output = tmp_path / "speech.wav"

        mock_tts_top, mock_api, mock_cls, mock_inst = _make_mock_tts_module()

        def fake_tts_to_file(text, speaker_wav, language, file_path):
            sr = 24000
            sf.write(file_path, np.zeros(int(0.5 * sr), dtype=np.float32), sr)

        mock_inst.tts_to_file.side_effect = fake_tts_to_file

        with patch.dict(sys.modules, {"TTS": mock_tts_top, "TTS.api": mock_api}):
            result = synthesize_segments(
                segments=segments,
                voice_reference=tmp_wav,
                target_language="es",
                output_path=output,
            )

        assert result == output
        assert output.exists()
        audio, _ = sf.read(str(output))
        assert len(audio) > 0

    def test_empty_segments_writes_silence(self, tmp_path: Path, tmp_wav: Path) -> None:
        output = tmp_path / "silence.wav"
        mock_tts_top, mock_api, _, _ = _make_mock_tts_module()

        with patch.dict(sys.modules, {"TTS": mock_tts_top, "TTS.api": mock_api}):
            result = synthesize_segments(
                segments=[],
                voice_reference=tmp_wav,
                target_language="es",
                output_path=output,
            )
        assert result == output
        assert output.exists()

    def test_voice_reference_passed_to_tts(self, tmp_path: Path, tmp_wav: Path) -> None:
        segments = [TranscriptionSegment(text="Hi", start=0.0, end=1.0)]
        output = tmp_path / "out.wav"

        mock_tts_top, mock_api, mock_cls, mock_inst = _make_mock_tts_module()

        def fake_tts_to_file(text, speaker_wav, language, file_path):
            sf.write(file_path, np.zeros(24000, dtype=np.float32), 24000)

        mock_inst.tts_to_file.side_effect = fake_tts_to_file

        with patch.dict(sys.modules, {"TTS": mock_tts_top, "TTS.api": mock_api}):
            synthesize_segments(
                segments=segments,
                voice_reference=tmp_wav,
                target_language="es",
                output_path=output,
            )

        _, kwargs = mock_inst.tts_to_file.call_args
        assert kwargs["speaker_wav"] == str(tmp_wav)
        assert kwargs["language"] == "es"


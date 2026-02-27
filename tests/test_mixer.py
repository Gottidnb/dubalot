"""Tests for dubalot.mixer."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from dubalot.mixer import _match_length, _normalise, mix_audio


class TestMatchLength:
    def test_trims_longer_array(self) -> None:
        audio = np.ones(100)
        result = _match_length(audio, 50)
        assert len(result) == 50
        assert np.all(result == 1.0)

    def test_pads_shorter_array(self) -> None:
        audio = np.ones(30)
        result = _match_length(audio, 50)
        assert len(result) == 50
        assert np.all(result[:30] == 1.0)
        assert np.all(result[30:] == 0.0)

    def test_same_length_unchanged(self) -> None:
        audio = np.ones(50)
        result = _match_length(audio, 50)
        assert np.array_equal(result, audio)


class TestNormalise:
    def test_peak_is_target(self) -> None:
        audio = np.array([0.2, -0.4, 0.1])
        result = _normalise(audio, target_peak=0.8)
        assert np.max(np.abs(result)) == pytest.approx(0.8)

    def test_silent_audio_unchanged(self) -> None:
        audio = np.zeros(100)
        result = _normalise(audio, target_peak=1.0)
        assert np.all(result == 0.0)


class TestMixAudio:
    def _write_wav(self, path: Path, amplitude: float, duration: float = 1.0, sr: int = 16000) -> Path:
        n = int(duration * sr)
        audio = np.full(n, amplitude, dtype=np.float32)
        sf.write(str(path), audio, sr)
        return path

    def test_no_background_returns_speech_only(self, tmp_path: Path) -> None:
        speech = self._write_wav(tmp_path / "speech.wav", 0.5)
        output = tmp_path / "mixed.wav"

        result = mix_audio(speech, background_path=None, output_path=output)

        assert result == output
        assert output.exists()

    def test_with_background_produces_longer_or_equal_output(
        self, tmp_path: Path
    ) -> None:
        speech = self._write_wav(tmp_path / "speech.wav", 0.5, duration=2.0)
        bg = self._write_wav(tmp_path / "bg.wav", 0.3, duration=3.0)
        output = tmp_path / "mixed.wav"

        mix_audio(speech, background_path=bg, output_path=output)

        mixed, sr = sf.read(str(output))
        speech_audio, _ = sf.read(str(speech))
        # Output must be exactly as long as speech
        assert len(mixed) == len(speech_audio)

    def test_mixed_output_does_not_clip(self, tmp_path: Path) -> None:
        speech = self._write_wav(tmp_path / "speech.wav", 0.9)
        bg = self._write_wav(tmp_path / "bg.wav", 0.9)
        output = tmp_path / "mixed.wav"

        mix_audio(speech, background_path=bg, output_path=output)

        mixed, _ = sf.read(str(output))
        assert np.max(np.abs(mixed)) <= 1.0 + 1e-6

    def test_missing_background_file_treated_as_none(
        self, tmp_path: Path
    ) -> None:
        speech = self._write_wav(tmp_path / "speech.wav", 0.5)
        output = tmp_path / "mixed.wav"

        # Pass a path that doesn't exist
        mix_audio(speech, background_path=tmp_path / "nope.wav", output_path=output)

        assert output.exists()

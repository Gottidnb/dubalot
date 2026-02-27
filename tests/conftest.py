"""Shared fixtures for dubalot tests."""

from __future__ import annotations

import struct
import wave
from pathlib import Path

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# WAV helpers
# ---------------------------------------------------------------------------


def make_wav(path: Path, duration: float = 1.0, sample_rate: int = 16000) -> Path:
    """Write a short sine-wave WAV file to *path* and return it."""
    n_samples = int(duration * sample_rate)
    t = np.linspace(0, duration, n_samples, endpoint=False)
    audio = (np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)

    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(audio.tobytes())

    return path


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_wav(tmp_path: Path) -> Path:
    """Return path to a temporary 1-second sine-wave WAV file."""
    return make_wav(tmp_path / "audio.wav", duration=1.0)


@pytest.fixture()
def tmp_wav_2s(tmp_path: Path) -> Path:
    """Return path to a temporary 2-second sine-wave WAV file."""
    return make_wav(tmp_path / "audio_2s.wav", duration=2.0)

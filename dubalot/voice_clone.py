"""
voice_clone.py – Clone a speaker's voice and synthesise speech in any language.

Uses Coqui TTS (XTTS v2) which supports zero-shot, multilingual voice cloning:
given a short reference audio sample from the speaker it can generate speech in
the same voice in over 16 languages.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Optional


class VoiceCloner:
    """Clone a speaker's voice and synthesise speech.

    Parameters
    ----------
    model_name:
        Coqui TTS model to use.  Defaults to ``tts_models/multilingual/multi-dataset/xtts_v2``
        which supports voice cloning in 16+ languages.
    device:
        Torch device string (``"cpu"``, ``"cuda"``, …).  When ``None`` the
        class auto-selects GPU when available, otherwise CPU.
    """

    DEFAULT_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.device = device or self._default_device()
        self._tts = None  # lazy-loaded on first use

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def synthesise(
        self,
        text: str,
        reference_audio: str | os.PathLike,
        output_path: str | os.PathLike,
        language: str = "en",
    ) -> str:
        """Synthesise *text* in the cloned voice and write a WAV file.

        Parameters
        ----------
        text:
            The translated text to be spoken.
        reference_audio:
            Path to a short audio clip (≥ 6 s recommended) of the original
            speaker used as the voice reference for cloning.
        output_path:
            Destination path for the synthesised WAV file.
        language:
            BCP-47 language code for the target language (e.g. ``"en"``,
            ``"es"``, ``"fr"``).

        Returns
        -------
        str
            Absolute path to the written WAV file.
        """
        reference_audio = str(reference_audio)
        output_path = str(output_path)

        if not os.path.isfile(reference_audio):
            raise FileNotFoundError(
                f"Reference audio file not found: {reference_audio}"
            )
        if not text.strip():
            raise ValueError("text must not be empty")

        tts = self._load_tts()
        tts.tts_to_file(
            text=text,
            speaker_wav=reference_audio,
            language=language,
            file_path=output_path,
        )
        return os.path.abspath(output_path)

    def extract_reference_audio(
        self,
        video_path: str | os.PathLike,
        output_path: Optional[str | os.PathLike] = None,
        duration: float = 30.0,
    ) -> str:
        """Extract a reference audio clip from *video_path*.

        Parameters
        ----------
        video_path:
            Path to the source video file.
        output_path:
            Destination WAV path.  When ``None`` a temporary file is created.
        duration:
            Maximum length of the reference clip in seconds.

        Returns
        -------
        str
            Path to the extracted WAV file.
        """
        video_path = str(video_path)
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        import ffmpeg  # imported here to keep module loadable without ffmpeg

        if output_path is None:
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            output_path = tmp.name
            tmp.close()

        output_path = str(output_path)
        (
            ffmpeg.input(video_path, t=duration)
            .output(
                output_path,
                format="wav",
                acodec="pcm_s16le",
                ac=1,
                ar="22050",
            )
            .overwrite_output()
            .run(quiet=True)
        )
        return os.path.abspath(output_path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _default_device() -> str:
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def _load_tts(self):
        """Lazy-load the TTS model on first use."""
        if self._tts is None:
            from TTS.api import TTS  # noqa: PLC0415

            self._tts = TTS(model_name=self.model_name).to(self.device)
        return self._tts

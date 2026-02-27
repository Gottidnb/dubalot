"""
pipeline.py – End-to-end dubbing pipeline.

Flow
----
1. Extract audio from the input video.
2. Transcribe the audio with OpenAI Whisper (auto-detects source language).
3. Translate the transcript to the target language with deep-translator.
4. Synthesise dubbed speech using the cloned voice (Coqui TTS / XTTS v2).
5. Sync the dubbed audio to the video lip movements (Wav2Lip or ffmpeg).
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Optional

from dubalot.voice_clone import VoiceCloner
from dubalot.lip_sync import LipSyncer


class DubalotPipeline:
    """Orchestrate the full dubbing pipeline.

    Parameters
    ----------
    target_language:
        Target language for dubbing, as a BCP-47 code (e.g. ``"en"``,
        ``"es"``, ``"fr"``).
    voice_cloner:
        A pre-configured :class:`~dubalot.voice_clone.VoiceCloner` instance.
        When ``None`` a default instance is created automatically.
    lip_syncer:
        A pre-configured :class:`~dubalot.lip_sync.LipSyncer` instance.
        When ``None`` a default instance (ffmpeg fallback) is created.
    whisper_model:
        Whisper model size to use for transcription.  Valid values are
        ``"tiny"``, ``"base"``, ``"small"``, ``"medium"``, ``"large"``.
        Defaults to ``"base"``.
    """

    def __init__(
        self,
        target_language: str = "en",
        voice_cloner: Optional[VoiceCloner] = None,
        lip_syncer: Optional[LipSyncer] = None,
        whisper_model: str = "base",
    ) -> None:
        self.target_language = target_language
        self.voice_cloner = voice_cloner or VoiceCloner()
        self.lip_syncer = lip_syncer or LipSyncer()
        self.whisper_model = whisper_model
        self._whisper = None  # lazy-loaded

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        video_path: str | os.PathLike,
        output_path: str | os.PathLike,
    ) -> str:
        """Dub *video_path* into :attr:`target_language`.

        Parameters
        ----------
        video_path:
            Path to the source video that should be dubbed.
        output_path:
            Destination path for the dubbed MP4 video.

        Returns
        -------
        str
            Absolute path to the dubbed video file.
        """
        video_path = str(video_path)
        output_path = str(output_path)

        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        with tempfile.TemporaryDirectory() as tmpdir:
            # 1. Extract reference audio for voice cloning
            ref_audio = os.path.join(tmpdir, "reference.wav")
            self.voice_cloner.extract_reference_audio(video_path, ref_audio)

            # 2. Transcribe
            transcript, source_language = self._transcribe(ref_audio)

            # 3. Translate
            translated = self._translate(
                transcript, source_language, self.target_language
            )

            # 4. Synthesise dubbed speech (cloned voice)
            dubbed_audio = os.path.join(tmpdir, "dubbed.wav")
            self.voice_cloner.synthesise(
                text=translated,
                reference_audio=ref_audio,
                output_path=dubbed_audio,
                language=self.target_language,
            )

            # 5. Lip-sync
            self.lip_syncer.sync(video_path, dubbed_audio, output_path)

        return os.path.abspath(output_path)

    def transcribe(self, audio_path: str | os.PathLike) -> tuple[str, str]:
        """Transcribe *audio_path* and return ``(text, detected_language)``."""
        return self._transcribe(str(audio_path))

    def translate(self, text: str, source: str, target: str) -> str:
        """Translate *text* from *source* language to *target* language."""
        return self._translate(text, source, target)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _transcribe(self, audio_path: str) -> tuple[str, str]:
        """Return ``(transcript, detected_language)`` using Whisper."""
        import whisper  # noqa: PLC0415

        model = self._load_whisper()
        result = model.transcribe(audio_path)
        return result["text"].strip(), result.get("language", "auto")

    def _translate(self, text: str, source: str, target: str) -> str:
        """Translate *text* from *source* to *target* using deep-translator."""
        if source == target:
            return text
        from deep_translator import GoogleTranslator  # noqa: PLC0415

        translator = GoogleTranslator(source=source, target=target)
        # deep-translator has a character limit per request; chunk if needed
        max_len = 4999
        if len(text) <= max_len:
            return translator.translate(text)

        chunks = [text[i : i + max_len] for i in range(0, len(text), max_len)]
        return " ".join(translator.translate(chunk) for chunk in chunks)

    def _load_whisper(self):
        if self._whisper is None:
            import whisper  # noqa: PLC0415

            self._whisper = whisper.load_model(self.whisper_model)
        return self._whisper

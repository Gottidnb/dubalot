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

import argparse
import os
import sys
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


def main(argv=None):
    """CLI entry point: ``dubalot -i video.mp4 -t es -o video_es.mp4``."""
    parser = argparse.ArgumentParser(
        prog="dubalot",
        description="Translate a foreign-language video to a target language with the same voice.",
    )
    parser.add_argument(
        "-i", "--input",
        required=True,
        metavar="INPUT",
        help="Path to the source video file (e.g. video.mp4).",
    )
    parser.add_argument(
        "-o", "--output",
        required=True,
        metavar="OUTPUT",
        help="Path for the dubbed output video (e.g. video_es.mp4).",
    )
    parser.add_argument(
        "-t", "--target-language",
        default="en",
        metavar="LANG",
        help="Target language BCP-47 code (e.g. en, es, fr). Default: en.",
    )
    parser.add_argument(
        "--whisper-model",
        default="base",
        metavar="MODEL",
        help="Whisper model size: tiny, base, small, medium, large. Default: base.",
    )
    parser.add_argument(
        "--wav2lip-checkpoint",
        default=None,
        metavar="PATH",
        help="Optional path to a Wav2Lip .pth checkpoint for lip animation.",
    )
    parser.add_argument(
        "--wav2lip-script",
        default=None,
        metavar="PATH",
        help="Optional path to Wav2Lip inference.py script.",
    )

    args = parser.parse_args(argv)

    if not os.path.isfile(args.input):
        parser.error(f"Input file not found: {args.input}")

    lip_syncer = LipSyncer(
        wav2lip_checkpoint=args.wav2lip_checkpoint,
        wav2lip_script=args.wav2lip_script,
    )
    pipeline = DubalotPipeline(
        target_language=args.target_language,
        lip_syncer=lip_syncer,
        whisper_model=args.whisper_model,
    )

    print(f"Dubbing '{args.input}' → '{args.output}' (target: {args.target_language})")
    output = pipeline.run(args.input, args.output)
    print(f"Done. Output saved to: {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

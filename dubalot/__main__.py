"""Command-line interface for dubalot.

Usage
-----
Translate a video file to English::

    python -m dubalot input.mp4 output.mp4 --target-language en

Translate to Spanish using a small Whisper model::

    python -m dubalot input.mp4 output.mp4 --target-language es --whisper-model small

Provide your own reference audio clip for voice cloning::

    python -m dubalot input.mp4 output.mp4 --target-language fr --reference-audio voice.wav
"""

from __future__ import annotations

import argparse
import sys

from dubalot.pipeline import translate_video


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dubalot",
        description="Translate foreign video to native language with the same voice.",
    )
    parser.add_argument("input_video", help="Path to the source video file.")
    parser.add_argument("output_video", help="Path for the dubbed output video.")
    parser.add_argument(
        "--target-language",
        required=True,
        metavar="LANG",
        help=(
            "Target language BCP-47 code, e.g. 'en', 'es', 'fr', 'de', 'ja'."
        ),
    )
    parser.add_argument(
        "--whisper-model",
        default="base",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size for transcription (default: base).",
    )
    parser.add_argument(
        "--tts-model",
        default="tts_models/multilingual/multi-dataset/xtts_v2",
        metavar="MODEL",
        help="Coqui TTS model identifier (default: xtts_v2).",
    )
    parser.add_argument(
        "--reference-audio",
        default=None,
        metavar="FILE",
        help=(
            "Optional path to a reference audio clip (~6 s) of the speaker "
            "for voice cloning. When omitted, the source video's own audio is used."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        output = translate_video(
            input_video=args.input_video,
            output_video=args.output_video,
            target_language=args.target_language,
            whisper_model=args.whisper_model,
            tts_model=args.tts_model,
            reference_audio=args.reference_audio,
        )
        print(f"Dubbed video saved to: {output}")
        return 0
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

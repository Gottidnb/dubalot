"""
cli.py – command-line interface for dubalot.

Usage:
    dubalot --input video.mp4 --target-language es --output translated.mp4
    dubalot --input video.mp4 --target-language fr --model large-v3 --device cuda
"""

from __future__ import annotations

import argparse
import logging
import sys

from .pipeline import translate_video


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dubalot",
        description=(
            "Translate a video to a target language while preserving the "
            "speaker's voice, background audio, and lip-sync."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-i", "--input",
        required=True,
        metavar="PATH",
        help="Path to the source video file.",
    )
    parser.add_argument(
        "-o", "--output",
        required=True,
        metavar="PATH",
        help="Destination path for the translated video.",
    )
    parser.add_argument(
        "-t", "--target-language",
        required=True,
        metavar="LANG",
        help=(
            "Target language code understood by Google Translate / XTTS v2 "
            "(e.g. 'es', 'fr', 'de', 'ja', 'zh-CN')."
        ),
    )
    parser.add_argument(
        "-s", "--source-language",
        default="auto",
        metavar="LANG",
        help=(
            "Source language code, or 'auto' to let Whisper detect it "
            "automatically."
        ),
    )
    parser.add_argument(
        "-m", "--model",
        default="base",
        choices=["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"],
        dest="whisper_model",
        help="Whisper model size.  Larger models are more accurate but slower.",
    )
    parser.add_argument(
        "--voice-reference",
        metavar="PATH",
        default=None,
        help=(
            "Path to a clean audio clip of the speaker (3–30 s WAV).  "
            "When omitted, the vocal stem extracted from the video is used."
        ),
    )
    parser.add_argument(
        "--no-lip-sync",
        action="store_true",
        help=(
            "Disable Wav2Lip lip synchronisation.  Useful when Wav2Lip is not "
            "installed or for faster processing."
        ),
    )
    parser.add_argument(
        "--wav2lip-dir",
        metavar="PATH",
        default=None,
        help=(
            "Path to the Wav2Lip repository checkout.  Overrides the "
            "WAV2LIP_DIR environment variable."
        ),
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="PyTorch device to use for Whisper and TTS inference.",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep the temporary working directory for debugging.",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose (DEBUG) logging.",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``dubalot`` command."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )

    try:
        out = translate_video(
            input_path=args.input,
            output_path=args.output,
            target_language=args.target_language,
            source_language=args.source_language,
            whisper_model=args.whisper_model,
            voice_reference=args.voice_reference,
            lip_sync=not args.no_lip_sync,
            wav2lip_dir=args.wav2lip_dir,
            keep_temp=args.keep_temp,
            device=args.device,
        )
        print(f"Translation complete: {out}")
        return 0
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

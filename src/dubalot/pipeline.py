"""
pipeline.py – orchestrate the full video translation pipeline.

Steps:
  1. Extract audio from the source video.
  2. Separate vocals (speech) from the background using Demucs.
  3. Transcribe the speech with Whisper.
  4. Translate the transcription to the target language.
  5. Synthesise translated speech, cloning the original speaker's voice.
  6. Mix translated speech with the background audio stem.
  7. Apply Wav2Lip lip sync (optional).
  8. Compose the final video with the mixed audio track.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from .extractor import extract_audio, separate_audio_stems
from .transcriber import transcribe
from .translator import translate_segments
from .synthesizer import synthesize_segments
from .mixer import mix_audio
from .lip_sync import apply_lip_sync

logger = logging.getLogger(__name__)


def translate_video(
    input_path: str,
    output_path: str,
    target_language: str,
    source_language: str = "auto",
    whisper_model: str = "base",
    voice_reference: Optional[str] = None,
    lip_sync: bool = True,
    wav2lip_dir: Optional[str] = None,
    keep_temp: bool = False,
    device: str = "cpu",
) -> str:
    """Translate the video at *input_path* and write the result to *output_path*.

    Args:
        input_path: Path to the source video file.
        output_path: Destination path for the translated video.
        target_language: BCP-47 language code for the output language
            (e.g. ``"es"``, ``"fr"``, ``"de"``, ``"ja"``).
        source_language: Language of the source video, or ``"auto"`` for
            automatic detection by Whisper.
        whisper_model: Whisper model size – ``"tiny"``, ``"base"``,
            ``"small"``, ``"medium"``, or ``"large-v3"``.
        voice_reference: Optional path to a clean audio clip of the speaker
            to use as the voice reference for TTS cloning.  When omitted the
            vocal stem extracted from the video is used.
        lip_sync: Whether to apply Wav2Lip lip synchronisation.  Requires the
            ``WAV2LIP_DIR`` environment variable to point to a Wav2Lip checkout
            with a downloaded checkpoint.
        wav2lip_dir: Explicit path to the Wav2Lip repository (overrides the
            ``WAV2LIP_DIR`` environment variable).
        keep_temp: Keep the temporary working directory for debugging.
        device: PyTorch device – ``"cpu"`` or ``"cuda"``.

    Returns:
        Absolute path to the translated output video.

    Raises:
        FileNotFoundError: If *input_path* does not exist.
        RuntimeError: If any required step fails fatally.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input video not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    tmp_dir_obj = tempfile.TemporaryDirectory()
    tmp_dir = Path(tmp_dir_obj.name)

    try:
        _run_pipeline(
            input_path=input_path,
            output_path=output_path,
            target_language=target_language,
            source_language=source_language,
            whisper_model=whisper_model,
            voice_reference=Path(voice_reference) if voice_reference else None,
            lip_sync=lip_sync,
            wav2lip_dir=Path(wav2lip_dir) if wav2lip_dir else None,
            tmp_dir=tmp_dir,
            device=device,
        )
    finally:
        if keep_temp:
            logger.info("Temporary files kept at: %s", tmp_dir)
        else:
            tmp_dir_obj.cleanup()

    return str(output_path.resolve())


# ---------------------------------------------------------------------------
# Internal pipeline
# ---------------------------------------------------------------------------


def _run_pipeline(
    input_path: Path,
    output_path: Path,
    target_language: str,
    source_language: str,
    whisper_model: str,
    voice_reference: Optional[Path],
    lip_sync: bool,
    wav2lip_dir: Optional[Path],
    tmp_dir: Path,
    device: str,
) -> None:
    # ------------------------------------------------------------------
    # Step 1 – Extract audio
    # ------------------------------------------------------------------
    logger.info("[1/7] Extracting audio …")
    raw_audio_path = tmp_dir / "audio.wav"
    extract_audio(input_path, raw_audio_path)

    # ------------------------------------------------------------------
    # Step 2 – Source separation (vocals + background)
    # ------------------------------------------------------------------
    logger.info("[2/7] Separating speech from background …")
    stems_dir = tmp_dir / "stems"
    stems_dir.mkdir()
    vocals_path, background_path = separate_audio_stems(raw_audio_path, stems_dir)

    # ------------------------------------------------------------------
    # Step 3 – Transcription
    # ------------------------------------------------------------------
    logger.info("[3/7] Transcribing speech …")
    segments = transcribe(
        audio_path=vocals_path,
        model_size=whisper_model,
        language=source_language,
        device=device,
        word_timestamps=True,
    )

    if not segments:
        raise RuntimeError(
            "Whisper produced no transcription segments.  "
            "Check that the video contains audible speech."
        )

    # ------------------------------------------------------------------
    # Step 4 – Translation
    # ------------------------------------------------------------------
    logger.info("[4/7] Translating to '%s' …", target_language)
    translated_segments = translate_segments(
        segments=segments,
        target_language=target_language,
        source_language=source_language,
    )

    # ------------------------------------------------------------------
    # Step 5 – TTS (voice cloning)
    # ------------------------------------------------------------------
    logger.info("[5/7] Synthesising translated speech …")
    ref_audio = voice_reference or vocals_path
    speech_path = tmp_dir / "speech.wav"
    synthesize_segments(
        segments=translated_segments,
        voice_reference=ref_audio,
        target_language=target_language,
        output_path=speech_path,
        device=device,
    )

    # ------------------------------------------------------------------
    # Step 6 – Mix speech with background
    # ------------------------------------------------------------------
    logger.info("[6/7] Mixing speech with background audio …")
    mixed_audio_path = tmp_dir / "mixed.wav"
    mix_audio(
        speech_path=speech_path,
        background_path=background_path,
        output_path=mixed_audio_path,
    )

    # ------------------------------------------------------------------
    # Step 7 – Lip sync + final video composition
    # ------------------------------------------------------------------
    if lip_sync:
        logger.info("[7/7] Applying lip sync …")
        synced_video_path = tmp_dir / "synced.mp4"
        apply_lip_sync(
            video_path=input_path,
            audio_path=mixed_audio_path,
            output_path=synced_video_path,
            wav2lip_dir=wav2lip_dir,
        )
        source_for_final = synced_video_path
    else:
        logger.info("[7/7] Composing final video (lip sync disabled) …")
        source_for_final = input_path

    _compose_final_video(source_for_final, mixed_audio_path, output_path)
    logger.info("Translation complete → %s", output_path)


def _compose_final_video(
    video_path: Path,
    audio_path: Path,
    output_path: Path,
) -> None:
    """Mux *audio_path* into *video_path* and write to *output_path*.

    Replaces the video's existing audio track with *audio_path*.  Uses ffmpeg
    directly for reliable, lossless remuxing.
    """
    if not shutil.which("ffmpeg"):
        raise RuntimeError(
            "ffmpeg is required but not found on PATH.  "
            "Install it with your system package manager or conda."
        )

    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(video_path),
        "-i", str(audio_path),
        "-c:v", "copy",          # keep original video stream unchanged
        "-c:a", "aac",           # encode audio to AAC for broad compatibility
        "-map", "0:v:0",         # video from first input
        "-map", "1:a:0",         # audio from second input
        "-shortest",             # trim to shortest stream
        str(output_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed to compose final video:\n{result.stderr}"
        )

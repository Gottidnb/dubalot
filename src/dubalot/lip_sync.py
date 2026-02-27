"""
lip_sync.py – synchronise lip movements with the translated audio.

This module wraps **Wav2Lip** (https://github.com/Rudrabha/Wav2Lip), a GAN
model that re-renders the lower face of a video to match arbitrary input audio.

Because Wav2Lip is not distributed on PyPI, it must be installed manually:

    git clone https://github.com/Rudrabha/Wav2Lip
    # Download pretrained weights – see the Wav2Lip README for links.
    # Place wav2lip_gan.pth in Wav2Lip/checkpoints/

The path to the cloned directory is read from the ``WAV2LIP_DIR`` environment
variable or can be passed explicitly via *wav2lip_dir*.

If Wav2Lip is not available the original video is returned unchanged (with a
logged warning) so that the rest of the pipeline continues to produce usable
output.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_DEFAULT_CHECKPOINT = "wav2lip_gan.pth"


def apply_lip_sync(
    video_path: Path,
    audio_path: Path,
    output_path: Path,
    wav2lip_dir: Optional[Path] = None,
    checkpoint: str = _DEFAULT_CHECKPOINT,
) -> Path:
    """Apply Wav2Lip to synchronise *video_path* lips with *audio_path*.

    Args:
        video_path: Original video (will be used for face frames).
        audio_path: Translated mixed audio that the lips should match.
        output_path: Destination path for the lip-synced video.
        wav2lip_dir: Path to the Wav2Lip repository checkout.  Defaults to the
            ``WAV2LIP_DIR`` environment variable.
        checkpoint: Filename of the Wav2Lip checkpoint inside
            ``<wav2lip_dir>/checkpoints/``.

    Returns:
        *output_path* if Wav2Lip was applied successfully, or *video_path* if
        Wav2Lip is unavailable (fallback – no lip sync).
    """
    video_path = Path(video_path)
    audio_path = Path(audio_path)
    output_path = Path(output_path)

    wav2lip_dir = _resolve_wav2lip_dir(wav2lip_dir)
    if wav2lip_dir is None:
        logger.warning(
            "Wav2Lip not found – skipping lip sync.  "
            "Set the WAV2LIP_DIR environment variable to the Wav2Lip repo path "
            "or pass wav2lip_dir explicitly to enable lip sync."
        )
        # Copy (or symlink) original video to output_path so callers always
        # get a file at the expected location.
        shutil.copy2(str(video_path), str(output_path))
        return output_path

    checkpoint_path = wav2lip_dir / "checkpoints" / checkpoint
    if not checkpoint_path.exists():
        logger.warning(
            "Wav2Lip checkpoint not found at %s – skipping lip sync.",
            checkpoint_path,
        )
        shutil.copy2(str(video_path), str(output_path))
        return output_path

    logger.info(
        "Applying Wav2Lip: video=%s, audio=%s", video_path, audio_path
    )

    cmd = [
        "python",
        str(wav2lip_dir / "inference.py"),
        "--checkpoint_path", str(checkpoint_path),
        "--face", str(video_path),
        "--audio", str(audio_path),
        "--outfile", str(output_path),
        "--nosmooth",
    ]

    result = subprocess.run(
        cmd,
        cwd=str(wav2lip_dir),
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        logger.error(
            "Wav2Lip inference failed (exit %d):\n%s",
            result.returncode,
            result.stderr,
        )
        logger.warning("Falling back to original video (no lip sync).")
        shutil.copy2(str(video_path), str(output_path))
        return output_path

    logger.info("Lip-synced video written to %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_wav2lip_dir(
    explicit: Optional[Path],
) -> Optional[Path]:
    """Return the Wav2Lip directory, or ``None`` if not found."""
    if explicit is not None:
        return Path(explicit)

    env_val = os.environ.get("WAV2LIP_DIR")
    if env_val:
        return Path(env_val)

    # Check if 'inference.py' is on PATH (unlikely but possible)
    if shutil.which("wav2lip_inference") is not None:
        return None  # handled via PATH – not our subprocess approach

    return None

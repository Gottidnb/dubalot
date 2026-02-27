"""
lip_sync.py – Synchronise generated dubbed audio with the speaker's lip
movements in the original video.

Strategy
--------
1. The dubbed audio (cloned voice) and the original video are passed to
   ``Wav2Lip`` (https://github.com/Rudrabha/Wav2Lip) via a subprocess call
   when the model checkpoint is available on disk.
2. When Wav2Lip is not installed a pure-``ffmpeg`` fallback is used:  the
   original video stream is combined with the new audio track.  This does
   *not* animate the lips but produces a correctly synchronised output video
   that can be used immediately while the full Wav2Lip setup is prepared.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional


class LipSyncer:
    """Synchronise dubbed audio with lip movements in the original video.

    Parameters
    ----------
    wav2lip_checkpoint:
        Optional path to a pre-trained ``Wav2Lip`` ``.pth`` checkpoint file.
        When provided the ``Wav2Lip`` model is used for high-quality lip
        animation.  When ``None`` (default) the class falls back to simple
        audio replacement via ``ffmpeg``.
    wav2lip_script:
        Optional path to the ``inference.py`` script inside a local
        ``Wav2Lip`` repository.  Required when *wav2lip_checkpoint* is set.
    face_det_checkpoint:
        Optional path to the ``s3fd`` face detection model used by Wav2Lip.
    """

    def __init__(
        self,
        wav2lip_checkpoint: Optional[str | os.PathLike] = None,
        wav2lip_script: Optional[str | os.PathLike] = None,
        face_det_checkpoint: Optional[str | os.PathLike] = None,
    ) -> None:
        self.wav2lip_checkpoint = (
            str(wav2lip_checkpoint) if wav2lip_checkpoint else None
        )
        self.wav2lip_script = str(wav2lip_script) if wav2lip_script else None
        self.face_det_checkpoint = (
            str(face_det_checkpoint) if face_det_checkpoint else None
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def sync(
        self,
        video_path: str | os.PathLike,
        audio_path: str | os.PathLike,
        output_path: str | os.PathLike,
    ) -> str:
        """Produce a lip-synced video from *video_path* and *audio_path*.

        Parameters
        ----------
        video_path:
            Path to the original video file (used for the face/lip track).
        audio_path:
            Path to the dubbed audio WAV file to sync to the video.
        output_path:
            Destination path for the output video (MP4).

        Returns
        -------
        str
            Absolute path to the written video file.
        """
        video_path = str(video_path)
        audio_path = str(audio_path)
        output_path = str(output_path)

        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        if not os.path.isfile(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        if self._wav2lip_available():
            return self._sync_wav2lip(video_path, audio_path, output_path)
        return self._sync_ffmpeg(video_path, audio_path, output_path)

    # ------------------------------------------------------------------
    # Wav2Lip-based sync
    # ------------------------------------------------------------------

    def _wav2lip_available(self) -> bool:
        return bool(
            self.wav2lip_checkpoint
            and os.path.isfile(self.wav2lip_checkpoint)
            and self.wav2lip_script
            and os.path.isfile(self.wav2lip_script)
        )

    def _sync_wav2lip(
        self, video_path: str, audio_path: str, output_path: str
    ) -> str:
        """Run Wav2Lip inference to animate lips to match the dubbed audio."""
        cmd = [
            "python",
            self.wav2lip_script,
            "--checkpoint_path",
            self.wav2lip_checkpoint,
            "--face",
            video_path,
            "--audio",
            audio_path,
            "--outfile",
            output_path,
        ]
        if self.face_det_checkpoint:
            cmd += ["--face_det_batch_size", "1"]

        result = subprocess.run(
            cmd, capture_output=True, text=True, check=False
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Wav2Lip inference failed:\n{result.stderr}"
            )
        return os.path.abspath(output_path)

    # ------------------------------------------------------------------
    # ffmpeg-based audio replacement fallback
    # ------------------------------------------------------------------

    @staticmethod
    def _sync_ffmpeg(
        video_path: str, audio_path: str, output_path: str
    ) -> str:
        """Replace the audio track in *video_path* with *audio_path*.

        This is a simple fallback that produces a correctly-timed dubbed video
        without lip animation.  It requires ``ffmpeg`` to be installed and
        available on ``$PATH``.
        """
        if not shutil.which("ffmpeg"):
            raise EnvironmentError(
                "ffmpeg not found on PATH.  Install ffmpeg or provide a "
                "Wav2Lip checkpoint for lip animation."
            )

        cmd = [
            "ffmpeg",
            "-y",
            "-i", video_path,
            "-i", audio_path,
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-c:v", "copy",
            "-shortest",
            output_path,
        ]
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=False
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"ffmpeg audio replacement failed:\n{result.stderr}"
            )
        return os.path.abspath(output_path)

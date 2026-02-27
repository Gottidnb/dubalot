"""
Video editing utilities for dubalot.

Provides functions to extract audio from video, replace audio in video,
and trim video segments - core operations needed for video dubbing.
"""

import os
import subprocess
import tempfile
from typing import Optional


def extract_audio(video_path: str, output_path: Optional[str] = None) -> str:
    """
    Extract audio track from a video file.

    Args:
        video_path: Path to the source video file.
        output_path: Path for the extracted audio file. If not provided,
                     a temporary .wav file is created.

    Returns:
        Path to the extracted audio file.

    Raises:
        FileNotFoundError: If video_path does not exist.
        RuntimeError: If audio extraction fails.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    if output_path is None:
        suffix = ".wav"
        fd, output_path = tempfile.mkstemp(suffix=suffix)
        os.close(fd)

    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "44100",
        "-ac", "2",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Failed to extract audio from {video_path}:\n{result.stderr}"
        )

    return output_path


def replace_audio(video_path: str, audio_path: str, output_path: str) -> str:
    """
    Replace the audio track of a video with a new audio file.

    Args:
        video_path: Path to the source video file.
        audio_path: Path to the new audio file.
        output_path: Path for the output video file.

    Returns:
        Path to the output video file.

    Raises:
        FileNotFoundError: If video_path or audio_path does not exist.
        RuntimeError: If audio replacement fails.
    """
    for path, label in ((video_path, "Video"), (audio_path, "Audio")):
        if not os.path.exists(path):
            raise FileNotFoundError(f"{label} file not found: {path}")

    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-i", audio_path,
        "-c:v", "copy",
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-shortest",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Failed to replace audio in {video_path}:\n{result.stderr}"
        )

    return output_path


def trim_video(
    video_path: str,
    output_path: str,
    start_time: float,
    end_time: Optional[float] = None,
) -> str:
    """
    Trim a video to the specified time range.

    Args:
        video_path: Path to the source video file.
        output_path: Path for the trimmed video file.
        start_time: Start time in seconds.
        end_time: End time in seconds. If not provided, trims to end of video.

    Returns:
        Path to the trimmed video file.

    Raises:
        FileNotFoundError: If video_path does not exist.
        ValueError: If start_time is negative or start_time >= end_time.
        RuntimeError: If trimming fails.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    if start_time < 0:
        raise ValueError(f"start_time must be non-negative, got {start_time}")

    if end_time is not None and end_time <= start_time:
        raise ValueError(
            f"end_time ({end_time}) must be greater than start_time ({start_time})"
        )

    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-ss", str(start_time),
    ]
    if end_time is not None:
        cmd += ["-to", str(end_time)]
    cmd += ["-c", "copy", output_path]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Failed to trim {video_path}:\n{result.stderr}"
        )

    return output_path

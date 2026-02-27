"""Video processing utilities.

Handles extracting audio from video files and merging a new audio track
back into the original video stream.
"""

from __future__ import annotations

import os


try:
    from moviepy.editor import VideoFileClip, AudioFileClip
except ImportError:
    VideoFileClip = None  # type: ignore[assignment,misc]
    AudioFileClip = None  # type: ignore[assignment,misc]


def extract_audio(video_path: str, output_audio_path: str) -> str:
    """Extract the audio track from *video_path* and save it as a WAV file.

    Parameters
    ----------
    video_path:
        Path to the source video file.
    output_audio_path:
        Destination path for the extracted audio (WAV format).

    Returns
    -------
    str
        The path to the extracted audio file (*output_audio_path*).
    """
    if VideoFileClip is None:
        raise ImportError("moviepy is not installed. Run: pip install moviepy")

    clip = VideoFileClip(video_path)
    if clip.audio is None:
        raise ValueError(f"Video file '{video_path}' has no audio track.")
    clip.audio.write_audiofile(output_audio_path, logger=None)
    clip.close()
    return output_audio_path


def merge_audio(video_path: str, audio_path: str, output_video_path: str) -> str:
    """Replace the audio track of *video_path* with *audio_path*.

    Parameters
    ----------
    video_path:
        Path to the original video file (video stream is kept).
    audio_path:
        Path to the new audio file that will replace the original track.
    output_video_path:
        Destination path for the resulting video file.

    Returns
    -------
    str
        The path to the final video file (*output_video_path*).
    """
    if VideoFileClip is None or AudioFileClip is None:
        raise ImportError("moviepy is not installed. Run: pip install moviepy")

    video_clip = VideoFileClip(video_path)
    audio_clip = AudioFileClip(audio_path)

    final_clip = video_clip.set_audio(audio_clip)
    final_clip.write_videofile(output_video_path, codec="libx264", audio_codec="aac", logger=None)

    video_clip.close()
    audio_clip.close()
    final_clip.close()

    return output_video_path

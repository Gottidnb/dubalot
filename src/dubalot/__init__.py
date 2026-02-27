"""
dubalot – video translation tool.

Translates a video to a target language while:
  • preserving the speaker's voice (with a natural accent) via voice cloning
  • keeping background audio intact
  • synchronising the translated speech with lip movements in the video
"""

from .pipeline import translate_video

__all__ = ["translate_video"]
__version__ = "0.1.0"

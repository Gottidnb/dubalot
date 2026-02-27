"""
Dubalot – translate foreign video to native language with the same voice.

Package exports:
    VoiceCloner  – clone a speaker's voice and synthesise speech
    LipSyncer    – sync generated audio to the speaker's lip movements
    DubalotPipeline – end-to-end dubbing pipeline
"""

from dubalot.voice_clone import VoiceCloner
from dubalot.lip_sync import LipSyncer
from dubalot.pipeline import DubalotPipeline

__all__ = ["VoiceCloner", "LipSyncer", "DubalotPipeline"]
__version__ = "0.1.0"

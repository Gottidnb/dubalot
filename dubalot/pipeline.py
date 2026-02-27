"""End-to-end video translation pipeline.

This module ties together all the individual steps:

1. Extract audio from the source video.
2. Transcribe the audio to text (with timestamps).
3. Translate each segment to the target language.
4. Synthesise new speech using the original speaker's voice.
5. Merge the translated audio back into the video.
"""

from __future__ import annotations

import os
import tempfile

from dubalot.transcriber import transcribe
from dubalot.translator import translate
from dubalot.synthesizer import synthesize
from dubalot.video_processor import extract_audio, merge_audio


def translate_video(
    input_video: str,
    output_video: str,
    target_language: str,
    whisper_model: str = "base",
    tts_model: str = "tts_models/multilingual/multi-dataset/xtts_v2",
    reference_audio: str | None = None,
) -> str:
    """Translate *input_video* and save the dubbed version to *output_video*.

    Parameters
    ----------
    input_video:
        Path to the source video file that should be dubbed.
    output_video:
        Destination path for the translated/dubbed video.
    target_language:
        BCP-47 language tag for the desired output language (e.g. ``"en"``,
        ``"es"``, ``"fr"``, ``"de"``, ``"ja"``).
    whisper_model:
        Whisper model size used for transcription.
    tts_model:
        Coqui TTS model used for speech synthesis.
    reference_audio:
        Optional path to an external reference audio clip for voice cloning.
        When *None*, the audio extracted from the source video is used.

    Returns
    -------
    str
        The path to the finished dubbed video (*output_video*).

    Raises
    ------
    FileNotFoundError
        If *input_video* does not exist.
    ValueError
        If the source video has no audio track.
    """
    if not os.path.isfile(input_video):
        raise FileNotFoundError(f"Input video not found: {input_video}")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Step 1 – extract audio
        extracted_audio = os.path.join(tmpdir, "original_audio.wav")
        extract_audio(input_video, extracted_audio)

        # Step 2 – transcribe
        segments = transcribe(extracted_audio, model_name=whisper_model)

        # Step 3 – translate
        translated_segments = translate(segments, target_language=target_language)

        # Step 4 – synthesise with voice cloning
        voice_ref = reference_audio or extracted_audio
        dubbed_audio = os.path.join(tmpdir, "dubbed_audio.wav")
        synthesize(translated_segments, reference_audio=voice_ref, output_path=dubbed_audio, model_name=tts_model)

        # Step 5 – merge audio back into video
        merge_audio(input_video, dubbed_audio, output_video)

    return output_video

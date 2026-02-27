from setuptools import setup, find_packages

setup(
    name="dubalot",
    version="0.1.0",
    description="Translate foreign video to native language with the same voice",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "openai-whisper>=20231117",
        "deep-translator>=1.11.4",
        "TTS>=0.22.0",
        "moviepy>=1.0.3",
        "ffmpeg-python>=0.2.0",
        "numpy>=1.24.0",
        "torch>=2.0.0",
        "torchaudio>=2.0.0",
        "opencv-python>=4.8.0",
        "Pillow>=10.0.0",
        "requests>=2.31.0",
    ],
    entry_points={
        "console_scripts": [
            "dubalot=dubalot.pipeline:main",
        ],
    },
)

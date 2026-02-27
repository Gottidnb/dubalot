from setuptools import setup, find_packages

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="dubalot",
    version="0.1.0",
    description="Translate foreign video to native language with the same voice",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Gottidnb",
    url="https://github.com/Gottidnb/dubalot",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "openai-whisper>=20231117",
        "deep-translator>=1.11.4",
        "TTS>=0.22.0",
        "moviepy>=1.0.3",
        "pydub>=0.25.1",
        "torch>=2.0.0",
        "tqdm>=4.65.0",
    ],
    entry_points={
        "console_scripts": [
            "dubalot=dubalot.__main__:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)

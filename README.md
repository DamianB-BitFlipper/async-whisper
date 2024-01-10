# Async Whisper API

Asynchronously transcribe audio files split into chunks in parallel and intelligently join results, yielding nearly identical transcriptions to full audio transcriptions but in a fraction of the time.

## Installation

Simply install the package via `pip`. Additionally, `ffmpeg` or `libav` must be installed on the system.

```bash
pip install async-whisper
```

To set up `ffmpeg`, follow the `pydub` [setup](https://github.com/jiaaro/pydub#getting-ffmpeg-set-up) instructions.

### Development Installation

The development installation requires software like `git` and `conda` to be already available on the system. Contributions are welcome. Please create a fork, make your changes there and then open a Pull Request onto the main GitHub repository.

```bash
# Clone the code
git clone git@github.com:DamianB-BitFlipper/async-whisper.git
cd async-whisper

# Install development dependencies
conda env create -f environment.yml
conda activate async-whisper

# Run a pip editable install
pip install -e .
```

## Usage

The most useful API interface is the `AsyncWhisper.transcribe_audio` async method. It accepts a `pydub.AudioSegment` audio object and transcribes it asynchronously. The [pydub](https://github.com/jiaaro/pydub) library enables easy audio loading and manipulations and is a dependency of this project already -- no additional installation is required.

```python
import asyncio
from pydub import AudioSegment
from async_whisper import AsyncWhisper

OPENAI_API_KEY = "<your-openai-api-key>"
AUDIO_FILEPATH = "/path/to/audio/file.mp3"


async def main():
  audio_data = AudioSegment.from_mp3(AUDIO_FILEPATH)
  whisper_client = AsyncWhisper(OPENAI_API_KEY)
  transcription = await whisper_client.transcribe_audio(audio_data)
  print(f"Transcription: {transcription}")

if __name__ == "__main__":
    asyncio.run(main())
```

It's as simple as that! 

More information on how to fine-tune the configuration of the `AsyncWhisper` class can be found in the [DOCUMENTATION.md](https://github.com/DamianB-BitFlipper/async-whisper/blob/main/DOCUMENTATION.md) file.

## How It Works

The high-level of how this library works is conceptually simple. The audio recording is split into chunks of 60 seconds. However, adjacent audio chunks share overlapped portions of 10 seconds. That is, a 3-minute audio recording (180s) will be split into 4 chunks whose boundaries are: `[0 - 60, 50 - 110, 100 - 160, 150 - 180]` seconds.

Each audio chunk is asynchronously transcribe with OpenAI's Whisper in parallel.

Then using the overlapping portions, adjacent transcribed chunks are intelligently joined. The joining processes works by minimizing the [Levenshtein distance](https://en.wikipedia.org/wiki/Levenshtein_distance) between the overlapping segments of audio. Since Whisper transcribes the same audio data in these overlapping sections, it will yield similar, though not always identical, transcription outputs. Minimizing the `Levenstein distance` is a good heuristic to merge two similar, but not necessarily identical, strings.

## Performance Results

In the performance evaluation, `async-whisper` is evaluated against simply transcribing the entire audio altogether using Whisper (baseline). Execution time as well as the output similarity of `async-whisper` to the baseline using the Levenshtein distance are presented. 

Similarity scores of ≥ 97% are essentially semantically identical transcriptions with differences only in punctuation.

#### Short audio (≈1.5 minutes in duration)

| Baseline   | `async-whisper` | Similarity Score |
|:----------:|:---------------:|:----------------:|
| **10.99s** | **10.02s**      | **100%**         |


#### Medium audio (≈3 minutes in duration)

| Baseline   | `async-whisper` | Similarity Score |
|:----------:|:---------------:|:----------------:|
| **32.49s** | **7.71s**       | **97%**          |


#### Long audio (≈7.5 minutes in duration)

| Baseline   | `async-whisper` | Similarity Score |
|:----------:|:---------------:|:----------------:|
| **56.31s** | **11.03s**      | **97%**          |

import asyncio
import logging
import os
import time
from pathlib import Path

from pydub import AudioSegment
from thefuzz import fuzz

from async_whisper import AsyncWhisper

# Configure the logging module
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

AUDIO_FILES_DIR = Path("./private_audio_files")


async def main() -> None:
    if "OPENAI_API_KEY" not in os.environ:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    whisper_client = AsyncWhisper(os.environ["OPENAI_API_KEY"])

    glob = "*.mp3"
    for audio_filepath in AUDIO_FILES_DIR.glob(glob):
        audio_segment = AudioSegment.from_mp3(audio_filepath)

        entire_start_time = time.time()
        # Temporarily disable the timeout
        retry_timeout = whisper_client.retry_timeout
        whisper_client.retry_timeout = None

        # Transcribe the audio segment
        entire_transcription = await whisper_client._safe_transcribe_audio_segment(
            audio_segment,
            uid=0,
        )

        # Restore the timeout
        whisper_client.retry_timeout = retry_timeout
        entire_end_time = time.time()

        async_start_time = time.time()
        # Transcribe the audio segment asynchronously
        async_transcription = await whisper_client.transcribe_audio(audio_segment)

        async_end_time = time.time()

        logging.info(
            f"Audio file: {audio_filepath} -- {fuzz.ratio(entire_transcription, async_transcription)}"
        )
        logging.info(
            f"  Entire time: {entire_end_time - entire_start_time} -- Segments time: {async_end_time - async_start_time}"
        )


if __name__ == "__main__":
    asyncio.run(main())

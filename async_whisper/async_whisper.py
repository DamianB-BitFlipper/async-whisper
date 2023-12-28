import asyncio
import io
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import openai
from aiolimiter import AsyncLimiter
from pydub import AudioSegment
from thefuzz import fuzz


client = openai.AsyncOpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)


@dataclass
class _StitchMeta:
    overlap_len: int
    fuzz_ratio: int


@dataclass
class _AudioChunk:
    segment: AudioSegment
    segment_length_ms: int
    transcription: str | None = None

    @property
    def transcription_words(self) -> list[str]:
        if self.transcription is None:
            raise ValueError("Transcription is not set")
        return self.transcription.split()


AUDIO_INTERVIEWS_DIR = Path("../private_audio_files")

# Allow a maximum of 100 requests per minute
ASYNC_RATE_LIMIT_RPM = 100

# Timeout and retry after 15 seconds for segment transcription
TRANSCRIBE_SEGMENT_TIMEOUT = 15

# Each segment is 60 seconds long
SEGMENT_LENGTH_MS = 60_000

# Have a 10 second overlap between each segment
OVERLAP_LENGTH_MS = 10_000

# When stitching together transcription segments, have
# a `STITCH_WIGGLE` of words wiggle room
STITCH_WIGGLE = 15

# How many words in a row must be identical before we start
# picking from the following segment during overlap resolution
RESOLVE_OVERLAP_THRESHOLD = 4


class AsyncWhisper:
    def __init__(
        self,
        audio_chunk_ms: int = SEGMENT_LENGTH_MS,
        overlap_ms: int = OVERLAP_LENGTH_MS,
        rate_limit_rpm: int = ASYNC_RATE_LIMIT_RPM,
        *,
        sitch_wiggle: int = STITCH_WIGGLE,
        resolve_overlap_threshold: int = RESOLVE_OVERLAP_THRESHOLD,
    ):
        # Save the values to the instance
        self.audio_chunk_ms = audio_chunk_ms
        self.overlap_ms = overlap_ms
        self.rate_limit_rpm = rate_limit_rpm
        self.stitch_wiggle = sitch_wiggle
        self.resolve_overlap_threshold = resolve_overlap_threshold

        # Create an `AsyncLimiter` to limit the rate of requests
        self.rate_limiter = AsyncLimiter(self.rate_limit_rpm, 60)

    @staticmethod
    async def transcribe_audio_segment(
        audio_segment: AudioSegment,
        *,
        uid: int,
        timeout: int | None,
        language: str,
        prompt: str,
    ) -> str:
        # Load the `audio_segment` into a buffer
        buffer = io.BytesIO()
        audio_segment.export(buffer, format="mp3")

        # Trick OpenAI into thinking the `buffer` is an mp3 file
        buffer.name = "audio_segment.mp3"

        start_time = time.time()
        # Retry the request until it succeeds
        while True:
            try:
                transcript = await asyncio.wait_for(
                    client.audio.transcriptions.create(
                        file=buffer,
                        model="whisper-1",
                        language=language,
                        prompt=prompt,
                    ),
                    timeout=timeout,
                )
                break
            except asyncio.TimeoutError:
                # Sanity check
                assert timeout is not None

                # Backoff the timeout for the next request
                timeout *= 2

                print("Timeout error, retrying...", file=sys.stderr)
            except (
                openai.APIConnectionError,
                openai.APIStatusError,
                openai.RateLimitError,
            ) as e:
                print(
                    f"An error occurred processing audio segment: {e}, retrying in 5 seconds...",
                    file=sys.stderr,
                )
                await asyncio.sleep(5)

        print(f"{uid:3}: Transcribed in {time.time() - start_time} seconds")

        return transcript.text


async def safe_transcribe_audio_segment(
    audio_segment: AudioSegment,
    *,
    uid: int,
    timeout: int | None,
    language: str = "en",
    prompt: str = "",
) -> str:
    async with ASYNC_RATE_LIMITER:
        return await transcribe_audio_segment(
            audio_segment,
            uid=uid,
            timeout=timeout,
            language=language,
            prompt=prompt,
        )


def stitch_audio_segments(
    before_words: list[str], after_words: list[str], approx_overlap_len: int
) -> StitchMeta:
    """Stitch the `before_words` with the `after_words`.

    The overlap of number of words is approximately `approx_overlap_len`.
    This function gives some wiggle room on top of that.
    """
    # Limit the `max_overlap_len` to prevent index overflowing
    max_overlap_len = min(
        approx_overlap_len + STITCH_WIGGLE,
        len(before_words),
        len(after_words),
    )

    # Go back word-by-word in `before_words` and forward word-by-word in `after_words`
    # measuring the fuzz ratio of each word in the `overlap_len` region of words
    fuzz_ratios = []
    for overlap_len in range(1, max_overlap_len + 1):
        before_words_overlap = before_words[-overlap_len:]
        after_words_overlap = after_words[:overlap_len]
        total_fuzz_ratio = 0

        # Sanity check
        assert len(before_words_overlap) == len(after_words_overlap)

        # Compute the fuzz ratio of each word in the overlapping region
        for word1, word2 in zip(before_words_overlap, after_words_overlap):
            total_fuzz_ratio += fuzz.ratio(word1, word2)

        # Create and append a `StitchMeta`
        stitch_meta = StitchMeta(
            overlap_len=overlap_len,
            fuzz_ratio=total_fuzz_ratio,
        )
        fuzz_ratios.append(stitch_meta)

    # Return the stitch meta of the greatest fuzz ratio
    return max(fuzz_ratios, key=lambda e: e.fuzz_ratio)


def resolve_overlap(
    overlap1: list[str], overlap2: list[str], streak_threshold: int
) -> list[str]:
    """Pick differing words from `overlap1` first, then from `overlap2`
    after there has been a segment of identical words.
    """
    # Sanity check
    assert len(overlap1) == len(overlap2)

    # Create a list of words to return
    resolved_overlap = []

    # Keep track if we are picking from `overlap2` yet
    picking_from_overlap2 = False

    # Iterate over the words in `overlap1` and `overlap2`
    identical_streak = 0
    for word1, word2 in zip(overlap1, overlap2):
        if word1 == word2:
            identical_streak += 1
        else:
            identical_streak = 0

        # If we have reached the `streak_threshold`, start picking from `overlap2`
        if identical_streak >= streak_threshold and not picking_from_overlap2:
            picking_from_overlap2 = True

        # Append the respective word to `resolved_overlap`
        if not picking_from_overlap2:
            resolved_overlap.append(word1)
        else:
            resolved_overlap.append(word2)

    return resolved_overlap

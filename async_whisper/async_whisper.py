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
class StitchMeta:
    overlap_len: int
    fuzz_ratio: int


@dataclass
class AudioChunk:
    segment: AudioSegment
    segment_length: int
    transcription: str | None = None

    @property
    def transcription_words(self) -> list[str]:
        if self.transcription is None:
            raise ValueError("Transcription is not set")
        return self.transcription.split()


AUDIO_INTERVIEWS_DIR = Path("../private_audio_files")

# Allow a maximum of 100 requests per minute
ASYNC_RATE_LIMITER = AsyncLimiter(100, 60)

# Timeout and retry after 15 seconds for segment transcription
TRANSCRIBE_SEGMENT_TIMEOUT = 15

# Have a 10 second overlap between each segment
OVERLAP_LENGTH = 10_000

# Each segment is 60 seconds long
SEGMENT_LENGTH = 60_000

# When stitching together transcription segments, have
# a `STITCH_WIGGLE` of words wiggle room
STITCH_WIGGLE = 15

# How many words in a row must be identical before we start
# picking from the following segment during overlap resolution
RESOLVE_OVERLAP_THRESHOLD = 4


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


async def main() -> None:
    glob = "*.mp3"
    for audio_filepath in AUDIO_INTERVIEWS_DIR.glob(glob):
        audio_segment = AudioSegment.from_mp3(audio_filepath)

        audio_chunks = []
        total_length = len(audio_segment)
        start = 0
        while True:
            # Make `SEGMENT_LENGTH` second segments
            end = min(start + SEGMENT_LENGTH, total_length)

            # Add the segment to the list
            audio_chunks.append(
                AudioChunk(
                    segment=audio_segment[start:end],
                    segment_length=end - start,
                )
            )

            # Break if we're at the end of the audio segment
            if end == total_length:
                break

            # Increment the start time
            start += SEGMENT_LENGTH - OVERLAP_LENGTH

        # ADDED
        entire_start_time = time.time()
        # Transcribe the audio segment
        entire_transcription = await safe_transcribe_audio_segment(
            audio_segment,
            uid=0,
            timeout=None,
        )
        entire_end_time = time.time()

        segments_start_time = time.time()
        # Transcribe each segment in `segments`
        transcription_tasks = [
            safe_transcribe_audio_segment(
                audio_chunk.segment,
                uid=audio_chunk_id,
                timeout=TRANSCRIBE_SEGMENT_TIMEOUT,
            )
            for audio_chunk_id, audio_chunk in enumerate(audio_chunks)
        ]
        transcriptions = await asyncio.gather(*transcription_tasks)
        segments_end_time = time.time()

        # Set the `transcription` attribute of each `AudioChunk`
        for audio_chunk, transcription in zip(audio_chunks, transcriptions):
            audio_chunk.transcription = transcription

        # Stitch the transcription segments together
        before_words = audio_chunks[0].transcription_words
        for i in range(1, len(audio_chunks)):
            prev_audio_chunk = audio_chunks[i - 1]
            prev_words = prev_audio_chunk.transcription_words

            current_audio_chunk = audio_chunks[i]
            current_words = current_audio_chunk.transcription_words

            # Approximate the overlap length by extrapolating the words spoken per second
            # from the `prev_audio_chunk` and the `current_audio_chunk`
            approx_overlap_len = int(
                (len(prev_words) + len(current_words))
                * (
                    OVERLAP_LENGTH
                    / (
                        prev_audio_chunk.segment_length
                        + current_audio_chunk.segment_length
                    )
                )
            )

            stitch_meta = stitch_audio_segments(
                before_words=before_words,
                after_words=current_words,
                approx_overlap_len=approx_overlap_len,
            )

            stitch_str1_words = before_words[: -stitch_meta.overlap_len]
            stitch_str2_words = current_words[stitch_meta.overlap_len :]
            stitch_overlap_words = resolve_overlap(
                overlap1=before_words[-stitch_meta.overlap_len :],
                overlap2=current_words[: stitch_meta.overlap_len],
                streak_threshold=RESOLVE_OVERLAP_THRESHOLD,
            )

            stitch_words = stitch_str1_words + stitch_overlap_words + stitch_str2_words

            # Update `before_words` for the next iteration
            before_words = stitch_words

        # The stitched transcript is the final `before_words`
        stitched_transcript_str = " ".join(before_words)

        print(
            f"Audio file: {audio_filepath} -- {fuzz.ratio(entire_transcription, stitched_transcript_str)}"
        )
        print(
            f"  Entire time: {entire_end_time - entire_start_time} -- Segments time: {segments_end_time - segments_start_time}"
        )


if __name__ == "__main__":
    asyncio.run(main())

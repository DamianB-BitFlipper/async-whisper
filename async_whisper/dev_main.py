import asyncio

from pydub import AudioSegment


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
                    segment_length_ms=end - start,
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

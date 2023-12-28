from dataclasses import dataclass

from thefuzz import fuzz


@dataclass
class _StitchMeta:
    overlap_len: int
    fuzz_ratio: int


@staticmethod
def stitch_audio_segments(
    before_words: list[str],
    after_words: list[str],
    approx_overlap_len: int,
    stitch_wiggle: int,
) -> _StitchMeta:
    """Stitch the `before_words` with the `after_words`.

    The overlap of number of words is approximately `approx_overlap_len`.
    This function gives some wiggle room on top of that.
    """
    # Limit the `max_overlap_len` to prevent index overflowing
    max_overlap_len = min(
        approx_overlap_len + stitch_wiggle,
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

        # Create and append a `_StitchMeta`
        stitch_meta = _StitchMeta(
            overlap_len=overlap_len,
            fuzz_ratio=total_fuzz_ratio,
        )
        fuzz_ratios.append(stitch_meta)

    # Return the stitch meta of the greatest fuzz ratio
    return max(fuzz_ratios, key=lambda e: e.fuzz_ratio)


@staticmethod
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

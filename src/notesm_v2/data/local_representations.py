from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np


LOCAL_REPRESENTATIONS = {"masked_mean_1", "span_mean_2", "span_mean_4"}


def span_count_for_representation(representation: str) -> int:
    if representation == "masked_mean_1":
        return 1
    if representation == "span_mean_2":
        return 2
    if representation == "span_mean_4":
        return 4
    raise ValueError(f"Unsupported local representation: {representation}")


def content_positions_from_ids(
    input_ids: Sequence[int],
    attention_mask: Sequence[int],
    excluded_token_ids: Iterable[int],
) -> np.ndarray:
    if len(input_ids) != len(attention_mask):
        raise ValueError("input_ids and attention_mask must have the same length")
    excluded = {int(token_id) for token_id in excluded_token_ids}
    positions = [
        position
        for position, (token_id, is_valid) in enumerate(zip(input_ids, attention_mask, strict=True))
        if int(is_valid) == 1 and int(token_id) not in excluded
    ]
    return np.asarray(positions, dtype=int)


def split_positions_into_spans(valid_positions: Sequence[int], span_count: int) -> list[np.ndarray]:
    positions = np.asarray(valid_positions, dtype=int)
    if span_count <= 0:
        raise ValueError("span_count must be positive")
    if len(positions) == 0:
        return []
    return [span.astype(int, copy=False) for span in np.array_split(positions, span_count) if len(span) > 0]


def evidence_spans_for_positions(valid_positions: Sequence[int], representation: str) -> list[tuple[np.ndarray, dict[str, int]]]:
    spans = split_positions_into_spans(valid_positions, span_count_for_representation(representation))
    rows: list[tuple[np.ndarray, dict[str, int]]] = []
    for within_position, positions in enumerate(spans):
        rows.append(
            (
                positions,
                {
                    "evidence_slot": int(within_position),
                    "within_segment_position": int(within_position),
                    "token_start_position": int(positions[0]),
                    "token_end_position": int(positions[-1]) + 1,
                    "token_count": int(len(positions)),
                },
            )
        )
    return rows


def assert_exact_token_coverage(valid_positions: Sequence[int], spans: Sequence[Sequence[int]]) -> None:
    expected = [int(position) for position in valid_positions]
    observed = [int(position) for span in spans for position in span]
    if observed != expected:
        raise ValueError("Evidence spans must cover each valid token exactly once in order")

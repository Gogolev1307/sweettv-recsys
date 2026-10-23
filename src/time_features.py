"""Half-open calendar interval splitting for numeric schedule/view features."""

from __future__ import annotations

import numpy as np
import pandas as pd


def clipped_intervals(frame, start, end, start_column="start_time", end_column="end_time"):
    lower, upper = pd.Timestamp(start).value // 10**9, pd.Timestamp(end).value // 10**9
    if lower >= upper:
        raise ValueError("Empty fitting/target window")
    starts = frame[start_column].to_numpy(dtype="datetime64[s]").astype(np.int64)
    ends = frame[end_column].to_numpy(dtype="datetime64[s]").astype(np.int64)
    positions = np.flatnonzero((starts < upper) & (ends > lower) & (ends > starts))
    return positions, np.maximum(starts[positions], lower), np.minimum(ends[positions], upper)


def split_buckets(starts, ends, bucket_seconds):
    """Return interval owner, epoch bucket, and positive seconds in each bucket."""
    if bucket_seconds <= 0 or np.any(ends <= starts):
        raise ValueError("Intervals and bucket size must be positive")
    first = starts // bucket_seconds
    lengths = (ends - 1) // bucket_seconds - first + 1
    owners = np.repeat(np.arange(len(starts)), lengths)
    buckets = (
        np.repeat(first, lengths) + np.arange(int(lengths.sum()))
        - np.repeat(np.cumsum(lengths) - lengths, lengths)
    )
    seconds = np.minimum(ends[owners], (buckets + 1) * bucket_seconds) - np.maximum(starts[owners], buckets * bucket_seconds)
    return owners, buckets, seconds


def hour_slots(channels, hour_buckets):
    """Channel indices x 48 slots; epoch day zero was Thursday (weekday 3)."""
    weekend = ((hour_buckets // 24 + 3) % 7 >= 5).astype(np.int64)
    return channels * 48 + weekend * 24 + hour_buckets % 24

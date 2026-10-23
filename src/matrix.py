"""Shared fitting-window seconds matrix and deterministic score-to-ranking conversion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import sparse


@dataclass
class InteractionMatrix:
    user_ids: np.ndarray
    item_ids: np.ndarray
    seconds: sparse.csr_matrix
    test_rows: np.ndarray
    candidate_columns: np.ndarray

    @classmethod
    def build(cls, watched_a: pd.DataFrame, test_users: Iterable[int], candidates: Iterable[int]):
        """Include cold users/unseen candidates as zero rows/columns, not observations."""
        test_users = np.asarray(list(test_users), dtype=np.uint64)
        candidates = np.asarray(sorted(candidates), dtype=np.int64)
        totals = watched_a.groupby(["user_id", "tv_show_id"], as_index=False, sort=True).overlap_seconds.sum()
        users = np.union1d(totals.user_id.to_numpy(dtype=np.uint64), test_users)
        items = np.union1d(totals.tv_show_id.to_numpy(dtype=np.int64), candidates)
        rows = np.searchsorted(users, totals.user_id.to_numpy(dtype=np.uint64))
        columns = np.searchsorted(items, totals.tv_show_id.to_numpy(dtype=np.int64))
        seconds = sparse.csr_matrix(
            (totals.overlap_seconds.to_numpy(dtype=np.float64), (rows, columns)),
            shape=(len(users), len(items)),
        )
        seconds.sort_indices()
        if (seconds.data <= 0).any() or users.dtype != np.dtype("uint64"):
            raise AssertionError("Invalid A-only seconds matrix")
        return cls(users, items, seconds, np.searchsorted(users, test_users), np.searchsorted(items, candidates))


def log_weights(seconds: sparse.csr_matrix) -> sparse.csr_matrix:
    result = seconds.astype(np.float64, copy=True)
    result.data = np.log1p(result.data / 60.0)
    return result


def aligned_day_matrix(day_counts: pd.DataFrame, matrix: InteractionMatrix) -> sparse.csr_matrix:
    """Put positive programme-day counts in exactly the seconds matrix id order."""
    users = day_counts.user_id.to_numpy(dtype=np.uint64)
    items = day_counts.tv_show_id.to_numpy(dtype=np.int64)
    rows, columns = np.searchsorted(matrix.user_ids, users), np.searchsorted(matrix.item_ids, items)
    if (rows >= len(matrix.user_ids)).any() or (columns >= len(matrix.item_ids)).any():
        raise ValueError("Day counts contain ids outside the fitting matrix")
    if not np.array_equal(matrix.user_ids[rows], users) or not np.array_equal(matrix.item_ids[columns], items):
        raise ValueError("Day counts are not aligned with fitting ids")
    result = sparse.csr_matrix(
        (day_counts.watched_days.to_numpy(dtype=np.float64), (rows, columns)), shape=matrix.seconds.shape
    )
    if (result.data <= 0).any():
        raise ValueError("Day counts must be positive")
    return result


def rank_scores(
    scores: np.ndarray,
    candidate_ids: np.ndarray,
    user_ids: Iterable[int],
    *,
    positive_only: bool = True,
    k: int = 5,
    secondary_scores: np.ndarray | None = None,
) -> dict[int, list[int]]:
    """Ascending-id ties; zero-evidence lists can be completed by the protocol.

    Repeated argmax is O(k * candidates) and deterministically takes the first
    maximum in the ascending-id columns, including at large tied boundaries.
    MF may rank finite negative scores too (positive_only=False).
    """
    users = tuple(map(int, user_ids))
    if scores.shape != (len(users), len(candidate_ids)):
        raise ValueError("Score dimensions do not match users/candidates")
    if len(candidate_ids) < k or np.any(candidate_ids[1:] <= candidate_ids[:-1]):
        raise ValueError("Candidates must be distinct and ascending, with at least k items")
    if not np.isfinite(scores).all():
        raise ValueError("Nonfinite recommendation scores")
    if secondary_scores is not None:
        if secondary_scores.shape != scores.shape or not np.isfinite(secondary_scores).all():
            raise ValueError("Invalid secondary scores")
    work = np.array(scores, dtype=np.float64, copy=True)
    rows = np.arange(len(users))
    choices, values = [], []
    for _ in range(k):
        columns = np.argmax(work, axis=1)
        if secondary_scores is not None:
            tied = work == work[rows, columns, None]
            columns = np.argmax(np.where(tied, secondary_scores, -np.inf), axis=1)
        choices.append(candidate_ids[columns])
        values.append(work[rows, columns].copy())
        work[rows, columns] = -np.inf
    choices, values = np.stack(choices, axis=1), np.stack(values, axis=1)
    return {
        user: [int(item) for item, value in zip(items, row_values) if not positive_only or value > 0]
        for user, items, row_values in zip(users, choices, values)
    }

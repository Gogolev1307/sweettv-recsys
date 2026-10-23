"""Watched-day habit rate corrected for fitting and target airing opportunities."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.time_features import clipped_intervals, split_buckets


def scheduled_day_counts(schedule, item_ids, start, end):
    """Count a date once per programme, across channels, repeats and midnight."""
    item_ids = np.asarray(item_ids, dtype=np.int64)
    if np.any(item_ids[1:] <= item_ids[:-1]):
        raise ValueError("Schedule-count ids must be distinct and ascending")
    frame = schedule.loc[schedule.tv_show_id.ne(0) & schedule.tv_show_id.isin(item_ids)]
    positions, left, right = clipped_intervals(frame, start, end)
    owners, days, _ = split_buckets(left, right, 86400)
    item_rows = np.searchsorted(item_ids, frame.tv_show_id.to_numpy(dtype=np.int64)[positions])[owners]
    unique = pd.DataFrame({"item_row": item_rows, "day": days}).drop_duplicates()
    return np.bincount(unique.item_row.to_numpy(dtype=np.int64), minlength=len(item_ids)).astype(np.int64)


class ExposureRecommender:
    def __init__(self, smoothing_days=3):
        if smoothing_days <= 0:
            raise ValueError("Exposure smoothing must be positive")
        self.smoothing_days = smoothing_days

    def fit(self, watched_days, item_ids, fit_schedule, start, end):
        self.item_ids = np.asarray(item_ids, dtype=np.int64)
        if watched_days.shape[1] != len(self.item_ids) or (watched_days.data < 0).any():
            raise ValueError("Watched-day matrix is not aligned with programme ids")
        self.fit_airing_days = scheduled_day_counts(fit_schedule, self.item_ids, start, end)
        maximum_watched_days = watched_days.max(axis=0).toarray().ravel()
        if np.any(maximum_watched_days > self.fit_airing_days):
            raise AssertionError("A user watched a programme on more days than it aired")
        self.rate = watched_days.multiply(1.0 / (self.fit_airing_days + self.smoothing_days)).tocsr()
        self.diagnostics = {
            "smoothing_days": self.smoothing_days,
            "fit_items_with_airings": int((self.fit_airing_days > 0).sum()),
            "max_fit_airing_days": int(self.fit_airing_days.max(initial=0)),
            "maximum_habit_rate": float(self.rate.data.max(initial=0)),
        }
        return self

    def score(self, user_rows, candidate_columns, target_schedule, start, end):
        target_days = scheduled_day_counts(target_schedule, self.item_ids[candidate_columns], start, end)
        self.diagnostics["target_candidates_with_airings"] = int((target_days > 0).sum())
        self.diagnostics["max_target_airing_days"] = int(target_days.max(initial=0))
        return self.rate[user_rows][:, candidate_columns].multiply(target_days).toarray()

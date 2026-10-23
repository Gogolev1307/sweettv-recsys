"""Channel/weekend/hour habit from ALL fitting-window raw viewing seconds."""

from __future__ import annotations

import numpy as np
from scipy import sparse

from src.time_features import clipped_intervals, hour_slots, split_buckets


class SlotRecommender:
    def __init__(self, chunk_size=100_000):
        if chunk_size <= 0:
            raise ValueError("Slot chunk size must be positive")
        self.chunk_size = chunk_size

    def fit(self, views, requested_users, start, end):
        positions, left, right = clipped_intervals(views, start, end, end_column="stop_time")
        users = views.user_id.to_numpy(dtype=np.uint64)[positions]
        channels = views.vsetv_id.to_numpy(dtype=np.int64)[positions]
        self.user_ids = np.union1d(users, np.asarray(list(requested_users), dtype=np.uint64))
        self.channels = np.unique(channels)
        shape = (len(self.user_ids), len(self.channels) * 48)
        totals = sparse.csr_matrix(shape, dtype=np.float64)
        segments = 0
        for offset in range(0, len(positions), self.chunk_size):
            stop = offset + self.chunk_size
            owners, hours, seconds = split_buckets(left[offset:stop], right[offset:stop], 3600)
            rows = np.searchsorted(self.user_ids, users[offset:stop])[owners]
            channel_rows = np.searchsorted(self.channels, channels[offset:stop])[owners]
            totals += sparse.csr_matrix((seconds, (rows, hour_slots(channel_rows, hours))), shape=shape)
            segments += len(owners)
        row_seconds = np.asarray(totals.sum(axis=1)).ravel()
        inverse = np.divide(1.0, row_seconds, out=np.zeros_like(row_seconds), where=row_seconds != 0)
        self.profile = (sparse.diags(inverse) @ totals).tocsr()
        row_sums = np.asarray(self.profile.sum(axis=1)).ravel()
        error = float(np.max(np.abs(row_sums - (row_seconds > 0)), initial=0))
        if int(row_seconds.sum()) != int((right - left).sum()) or error > 1e-12:
            raise AssertionError("Slot splitting/normalization lost fitting viewing time")
        self.diagnostics = {
            "view_intervals": len(positions), "view_seconds": int(row_seconds.sum()),
            "hour_segments": segments, "profile_users": len(self.user_ids),
            "nonempty_profiles": int((row_seconds > 0).sum()), "channels": len(self.channels),
            "nonzero_slots": self.profile.nnz, "max_row_sum_error": error,
        }
        return self

    def target_exposure(self, schedule, candidate_ids, start, end):
        candidate_ids = np.asarray(candidate_ids, dtype=np.int64)
        if np.any(candidate_ids[1:] <= candidate_ids[:-1]):
            raise ValueError("Candidate ids must be distinct and ascending")
        frame = schedule.loc[schedule.tv_show_id.isin(candidate_ids) & schedule.channel_id.isin(self.channels)]
        positions, left, right = clipped_intervals(frame, start, end)
        items = frame.tv_show_id.to_numpy(dtype=np.int64)[positions]
        channels = frame.channel_id.to_numpy(dtype=np.int64)[positions]
        shape = (len(candidate_ids), len(self.channels) * 48)
        exposure = sparse.csr_matrix(shape, dtype=np.float64)
        segments = 0
        for offset in range(0, len(positions), self.chunk_size):
            stop = offset + self.chunk_size
            owners, hours, seconds = split_buckets(left[offset:stop], right[offset:stop], 3600)
            rows = np.searchsorted(candidate_ids, items[offset:stop])[owners]
            channel_rows = np.searchsorted(self.channels, channels[offset:stop])[owners]
            exposure += sparse.csr_matrix((seconds, (rows, hour_slots(channel_rows, hours))), shape=shape)
            segments += len(owners)
        self.diagnostics["target_hour_segments"] = segments
        self.diagnostics["target_airing_seconds_on_known_channels"] = int(exposure.sum())
        return exposure

    def score(self, users, candidate_ids, target_schedule, start, end):
        users = np.asarray(list(users), dtype=np.uint64)
        rows = np.searchsorted(self.user_ids, users)
        if (rows >= len(self.user_ids)).any() or not np.array_equal(self.user_ids[rows], users):
            raise ValueError("Prediction users were not included in the slot profile index")
        exposure = self.target_exposure(target_schedule, candidate_ids, start, end)
        return (self.profile[rows] @ exposure.T).toarray()

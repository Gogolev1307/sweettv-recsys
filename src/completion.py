"""Channel-aware completion frequency corrected by fit/target airing counts."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse

from src.join import completed_airings
from src.time_features import clipped_intervals


def airing_channel_counts(schedule, start, end, candidates=None):
    """Count unique labelled channel/start airings intersecting the window once."""
    frame = schedule.drop_duplicates(["channel_id", "start_time"], keep="last")
    frame = frame.loc[frame.tv_show_id.ne(0) & frame.duration.gt(0)]
    if candidates is not None:
        frame = frame.loc[frame.tv_show_id.isin(candidates)]
    positions, _, _ = clipped_intervals(frame, start, end)
    return frame.iloc[positions].groupby(["tv_show_id", "channel_id"]).size().rename("airings")


class CompletionRecommender:
    def fit(self, interactions, requested_users, candidate_ids, schedule, start, end):
        self.candidate_ids = np.asarray(candidate_ids, dtype=np.int64)
        if np.any(self.candidate_ids[1:] <= self.candidate_ids[:-1]):
            raise ValueError("Completion candidates must be distinct and ascending")
        self.user_ids = np.union1d(
            interactions.user_id.to_numpy(dtype=np.uint64), np.asarray(list(requested_users), dtype=np.uint64),
        )
        counts = airing_channel_counts(schedule, start, end, self.candidate_ids)
        self.pairs = counts.index
        self.nA = counts.to_numpy(dtype=np.float64)
        self.nT = None
        item_columns = np.searchsorted(self.candidate_ids, self.pairs.get_level_values("tv_show_id").to_numpy(dtype=np.int64))
        self.to_items = sparse.csr_matrix(
            (np.ones(len(self.pairs)), (np.arange(len(self.pairs)), item_columns)),
            shape=(len(self.pairs), len(self.candidate_ids)),
        )
        high80, high50 = completed_airings(interactions, 80), completed_airings(interactions, 50)
        self.popularity_freq = high80.groupby("tv_show_id").size().rename("completed_user_airings")

        def user_pair_counts(high):
            high = high.loc[high.tv_show_id.isin(self.candidate_ids)]
            grouped = high.groupby(["user_id", "tv_show_id", "channel_id"], as_index=False).size()
            columns = self.pairs.get_indexer(pd.MultiIndex.from_frame(grouped[["tv_show_id", "channel_id"]]))
            if (columns < 0).any():
                raise AssertionError("Completed fitting airing is absent from the fitting schedule")
            rows = np.searchsorted(self.user_ids, grouped.user_id.to_numpy(dtype=np.uint64))
            result = sparse.csr_matrix(
                (grouped["size"].to_numpy(dtype=np.float64), (rows, columns)),
                shape=(len(self.user_ids), len(self.pairs)),
            )
            if len(self.pairs) and np.any(result.max(axis=0).toarray().ravel() > self.nA):
                raise AssertionError("User completion frequency exceeds scheduled airing count")
            return result

        self.n80, self.n50 = user_pair_counts(high80), user_pair_counts(high50)
        if ((self.n50 - self.n80).data < 0).any():
            raise AssertionError("80% completions must be a subset of 50% completions")
        self.diagnostics = {
            "fit_users": len(self.user_ids), "fit_candidate_programme_channels": len(self.pairs),
            "fit_candidate_airings": int(self.nA.sum()), "completed_80_user_airings": len(high80),
            "completed_50_user_airings": len(high50), "n80_nonzero_user_pairs": self.n80.nnz,
            "n50_nonzero_user_pairs": self.n50.nnz,
        }
        return self

    def prepare_target(self, schedule, start, end):
        counts = airing_channel_counts(schedule, start, end, self.candidate_ids)
        self.nT = counts.reindex(self.pairs, fill_value=0).to_numpy(dtype=np.float64)
        self.diagnostics["target_airings_on_fitted_programme_channels"] = int(self.nT.sum())
        return self

    def scores(self, users, k):
        if k <= 0 or self.nT is None:
            raise ValueError("Positive smoothing k and prepared target availability are required")
        users = np.asarray(list(users), dtype=np.uint64)
        rows = np.searchsorted(self.user_ids, users)
        if (rows >= len(self.user_ids)).any() or not np.array_equal(self.user_ids[rows], users):
            raise ValueError("Completion prediction users are outside the fitting index")
        multiplier = self.nT / (self.nA + k)
        score80 = (self.n80[rows].multiply(multiplier) @ self.to_items).toarray()
        score50 = (self.n50[rows].multiply(multiplier) @ self.to_items).toarray()
        return score80, score50

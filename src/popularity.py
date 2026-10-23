"""Distinct-fitting-user popularity, with deterministic numeric-id tie breaks."""

from __future__ import annotations

from typing import Iterable

import pandas as pd


class PopularityRecommender:
    def fit(self, watched_a: pd.DataFrame, candidates: Iterable[int]) -> "PopularityRecommender":
        """Pass thresholded fitting-window interactions (A, or final weeks 11-30)."""
        candidate_ids = sorted(set(map(int, candidates)) - {0})
        if len(candidate_ids) < 5:
            raise ValueError("Popularity requires at least five candidates")
        counts = watched_a.groupby("tv_show_id").user_id.nunique()
        self.counts = counts.reindex(candidate_ids, fill_value=0).astype("int64")
        ranking = self.counts.rename("users").rename_axis("tv_show_id").reset_index()
        ranking = ranking.sort_values(["users", "tv_show_id"], ascending=[False, True])
        self.ranking = ranking.tv_show_id.astype("int64").tolist()
        self.history_users = set(map(int, watched_a.user_id.unique()))
        return self

    def predict(self, user_ids: Iterable[int], k: int = 5) -> dict[int, list[int]]:
        if not hasattr(self, "ranking"):
            raise ValueError("Fit the popularity model first")
        if k <= 0 or len(self.ranking) < k:
            raise ValueError("Not enough candidates for the requested k")
        return {int(user): self.ranking[:k] for user in user_ids}

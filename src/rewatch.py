"""Reference baseline: repeat programmes ranked by total A watched seconds."""

from __future__ import annotations

import numpy as np
from scipy import sparse

from src.matrix import rank_scores


class RewatchRecommender:
    def fit(self, seconds: sparse.csr_matrix) -> "RewatchRecommender":
        self.seconds = seconds
        return self

    def score(self, user_rows: np.ndarray, candidate_columns: np.ndarray) -> np.ndarray:
        return self.seconds[user_rows][:, candidate_columns].toarray()


class RewatchDaysRecommender:
    """Rank distinct watched dates, then A seconds, then ascending programme id."""

    def fit(self, days: sparse.csr_matrix, seconds: sparse.csr_matrix) -> "RewatchDaysRecommender":
        if days.shape != seconds.shape:
            raise ValueError("Days and seconds must have the same id order/shape")
        self.days, self.seconds = days, seconds
        return self

    def score(self, user_rows: np.ndarray, candidate_columns: np.ndarray) -> np.ndarray:
        return self.days[user_rows][:, candidate_columns].toarray()

    def predict(self, user_rows, candidate_columns, candidate_ids, user_ids):
        return rank_scores(
            self.score(user_rows, candidate_columns), candidate_ids, user_ids,
            secondary_scores=self.seconds[user_rows][:, candidate_columns].toarray(),
        )

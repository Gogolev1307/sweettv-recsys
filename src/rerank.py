"""Fit-only top-50 completion candidates, temporal features, and learned ranking."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.ensemble import HistGradientBoostingClassifier

from src.completion import CompletionRecommender, airing_channel_counts
from src.hybrid import minmax_normalize
from src.join import completed_airings, watched_interactions

FEATURE_NAMES = tuple(
    f"{stat}_{threshold}" for threshold in ("80", "50", "300")
    for stat in ("n", "s", "nt", "days_since")
) + ("popularity_freq", "target_airings", "user_n80_total", "user_n80_share", "base_score", "base_rank")
CONFIGURATIONS = (
    {"max_iter": 300, "learning_rate": 0.05, "max_leaf_nodes": 7},
    {"max_iter": 200, "learning_rate": 0.05, "max_leaf_nodes": 15},
    {"max_iter": 400, "learning_rate": 0.03, "max_leaf_nodes": 31},
)
PROMOTION_RULE = "TK gain >=1% relative and wins in >=190/200 seeded first-half user samples"


def check_temporal_split(fit_start, fit_end, label_start, label_end, label_ceiling):
    values = list(map(pd.Timestamp, (fit_start, fit_end, label_start, label_end, label_ceiling)))
    start, stop, target_start, target_end, ceiling = values
    if not start < stop == target_start < target_end <= ceiling:
        raise ValueError("Training labels must follow the fit cutoff and remain within the allowed label ceiling")


@dataclass
class CandidateRows:
    users: np.ndarray
    items: np.ndarray
    features: np.ndarray

    def labels(self, relevant_by_user):
        return np.fromiter(
            (int(int(item) in relevant_by_user.get(int(user), ())) for user, items in zip(self.users, self.items) for item in items),
            dtype=np.uint8, count=self.items.size,
        )


class RerankFeatures:
    """Only fitting interactions enter statistics; target input is numeric availability."""

    def __init__(self, k=80, candidates_per_user=50, batch_size=128):
        if k != 80 or candidates_per_user <= 0 or batch_size <= 0:
            raise ValueError("The experiment freezes k=80 and positive candidate/batch sizes")
        self.k, self.candidates_per_user, self.batch_size = k, candidates_per_user, batch_size

    def fit(self, interactions, users, candidate_ids, fit_schedule, fit_start, fit_end,
            target_schedule, target_start, target_end):
        self.fit_end = pd.Timestamp(fit_end)
        if self.fit_end > pd.Timestamp(target_start):
            raise ValueError("Fitting observations overlap the target window")
        if interactions.airing_start.ge(self.fit_end).any():
            raise ValueError("Future airings entered the fitting feature statistics")
        self.completion = CompletionRecommender().fit(
            interactions, users, candidate_ids, fit_schedule, pd.Timestamp(fit_start), self.fit_end,
        ).prepare_target(target_schedule, pd.Timestamp(target_start), pd.Timestamp(target_end))
        self.candidate_ids = self.completion.candidate_ids
        self.user_ids = self.completion.user_ids
        high = {"80": completed_airings(interactions, 80), "50": completed_airings(interactions, 50),
                "300": watched_interactions(interactions)}
        self.user_total80 = high["80"].groupby("user_id").size()
        counts300 = high["300"].loc[high["300"].tv_show_id.isin(self.candidate_ids)].groupby(
            ["user_id", "tv_show_id", "channel_id"], as_index=False
        ).size()
        columns = self.completion.pairs.get_indexer(pd.MultiIndex.from_frame(counts300[["tv_show_id", "channel_id"]]))
        if (columns < 0).any():
            raise AssertionError("Fitting 300s airing is absent from the fitting schedule")
        n300 = sparse.csr_matrix(
            (counts300["size"].to_numpy(dtype=np.float64),
             (np.searchsorted(self.user_ids, counts300.user_id.to_numpy(dtype=np.uint64)), columns)),
            shape=self.completion.n80.shape,
        )
        counts = {"80": self.completion.n80, "50": self.completion.n50, "300": n300}
        multiplier = self.completion.nT / (self.completion.nA + self.k)
        self.n, self.s, self.nt, self.recency = {}, {}, {}, {}
        for threshold, values in counts.items():
            self.n[threshold] = (values @ self.completion.to_items).tocsr()
            self.s[threshold] = (values.multiply(multiplier) @ self.completion.to_items).tocsr()
            active = values.copy()
            active.data[:] = 1.0
            self.nt[threshold] = (active.multiply(self.completion.nT) @ self.completion.to_items).tocsr()
            last = high[threshold].groupby(["user_id", "tv_show_id"]).airing_start.max()
            self.recency[threshold] = (self.fit_end - last).dt.total_seconds() / 86400.0
        self.frequency = self.completion.popularity_freq.reindex(self.candidate_ids, fill_value=0).to_numpy(dtype=np.float64)
        maximum = float(self.frequency.max(initial=0))
        self.normalized_frequency = np.divide(self.frequency, maximum, out=np.zeros_like(self.frequency), where=maximum != 0)
        target_counts = airing_channel_counts(target_schedule, pd.Timestamp(target_start), pd.Timestamp(target_end), self.candidate_ids)
        self.target_airings = target_counts.groupby(level="tv_show_id").sum().reindex(self.candidate_ids, fill_value=0).to_numpy(dtype=np.float64)
        return self

    @staticmethod
    def _query(matrix, rows, columns):
        return np.asarray(matrix[rows, columns]).ravel()

    def generate(self, users):
        users = np.asarray(list(users), dtype=np.uint64)
        global_rows = np.searchsorted(self.user_ids, users)
        if (global_rows >= len(self.user_ids)).any() or not np.array_equal(self.user_ids[global_rows], users):
            raise ValueError("Candidate users are outside the fitting feature index")
        width = min(self.candidates_per_user, len(self.candidate_ids))
        items = np.empty((len(users), width), dtype=np.int64)
        features = np.empty((len(users) * width, len(FEATURE_NAMES)), dtype=np.float64)
        for start in range(0, len(users), self.batch_size):
            stop = min(start + self.batch_size, len(users))
            rows = global_rows[start:stop]
            base = minmax_normalize(self.s["80"][rows].toarray())
            base += 0.5 * minmax_normalize(self.s["50"][rows].toarray())
            base += 0.1 * self.normalized_frequency
            # Candidate-id columns are ascending; stable sorting gives exact
            # descending-score/ascending-id ordering even at the top-50 boundary.
            top = np.argsort(-base, axis=1, kind="stable")[:, :width]
            items[start:stop] = self.candidate_ids[top]
            query_rows, query_columns = np.repeat(rows, width), top.ravel()
            query_users = np.repeat(users[start:stop], width)
            query_items = items[start:stop].ravel()
            query_index = pd.MultiIndex.from_arrays([query_users, query_items], names=["user_id", "tv_show_id"])
            out = features[start * width:stop * width]
            for offset, threshold in enumerate(("80", "50", "300")):
                out[:, offset * 4] = self._query(self.n[threshold], query_rows, query_columns)
                out[:, offset * 4 + 1] = self._query(self.s[threshold], query_rows, query_columns)
                out[:, offset * 4 + 2] = self._query(self.nt[threshold], query_rows, query_columns)
                out[:, offset * 4 + 3] = self.recency[threshold].reindex(query_index, fill_value=999).to_numpy(dtype=np.float64)
            out[:, 12] = self.frequency[query_columns]
            out[:, 13] = self.target_airings[query_columns]
            out[:, 14] = np.repeat(self.user_total80.reindex(users[start:stop], fill_value=0).to_numpy(dtype=np.float64), width)
            out[:, 15] = np.divide(out[:, 0], out[:, 14], out=np.zeros(len(out)), where=out[:, 14] != 0)
            out[:, 16] = base[np.arange(stop - start)[:, None], top].ravel()
            out[:, 17] = np.tile(np.arange(width), stop - start)
        if not np.isfinite(features).all() or np.any(features[:, (3, 7, 11)] < 0):
            raise AssertionError("Nonfinite features or future recency entered the candidate data")
        return CandidateRows(users, items, features)


def probability_rankings(rows: CandidateRows, probabilities, k=5):
    probabilities = np.asarray(probabilities).reshape(rows.items.shape)
    if not np.isfinite(probabilities).all():
        raise ValueError("Nonfinite re-ranker probabilities")
    return {
        int(user): [int(items[index]) for index in np.lexsort((items, -scores))[:k]]
        for user, items, scores in zip(rows.users, rows.items, probabilities)
    }


class LearnedReranker:
    def __init__(self, configuration):
        if configuration not in CONFIGURATIONS:
            raise ValueError("Re-ranker configuration is outside the requested three-point grid")
        self.configuration = dict(configuration)
        self.model = HistGradientBoostingClassifier(**configuration, random_state=42, early_stopping=False)

    def fit(self, features, labels):
        if (features.shape[1] != len(FEATURE_NAMES) or features.shape[0] != len(labels)
                or not np.isfinite(features).all() or set(np.unique(labels)) != {0, 1}):
            raise ValueError("Expected 18 features and both binary training classes")
        self.model.fit(features, labels)
        return self

    def predict(self, rows):
        return probability_rankings(rows, self.model.predict_proba(rows.features)[:, 1])


def user_partition_robustness(baseline_ap, rerank_ap, trials=200, seed=42):
    baseline_ap, rerank_ap = np.asarray(baseline_ap), np.asarray(rerank_ap)
    if len(baseline_ap) != len(rerank_ap) or len(baseline_ap) < 2:
        raise ValueError("Robustness requires paired AP values for the same evaluated users")
    rng = np.random.default_rng(seed)
    ratios, complementary = [], []
    for _ in range(trials):
        order = rng.permutation(len(baseline_ap))
        halves = (order[:len(order) // 2], order[len(order) // 2:])
        for destination, indices in zip((ratios, complementary), halves):
            baseline, score = float(baseline_ap[indices].mean()), float(rerank_ap[indices].mean())
            destination.append(score / baseline if baseline else (float("inf") if score else 1.0))
    ratios, complementary = np.array(ratios), np.array(complementary)
    return {
        "half_sample_wins": int((ratios > 1).sum()), "half_sample_ties": int((ratios == 1).sum()),
        "ratio_min": float(ratios.min()), "ratio_median": float(np.median(ratios)), "ratio_max": float(ratios.max()),
        "complementary_wins": int((complementary > 1).sum()),
        "both_halves_wins": int(((ratios > 1) & (complementary > 1)).sum()),
        "all_halves_ratio_min": float(min(ratios.min(), complementary.min())),
        "all_halves_ratio_median": float(np.median(np.concatenate([ratios, complementary]))),
        "all_halves_ratio_max": float(max(ratios.max(), complementary.max())),
        "trials": trials, "seed": seed,
    }

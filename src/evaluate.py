"""AP/MAP and the shared candidate, fallback and cohort validation contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Collection, Iterable, Mapping, Sequence

import pandas as pd


def average_precision_at_k(
    predictions: Iterable[int], relevant: Collection[int], k: int = 5
) -> float:
    """AP@k; repeats earn no further credit and still occupy their list rank."""
    if k <= 0:
        raise ValueError("k must be positive")
    truth = set(relevant)
    if not truth:
        return 0.0
    seen, hits, score = set(), 0, 0.0
    for rank, item in enumerate(predictions, start=1):
        if rank > k:
            break
        if item not in seen and item in truth:
            hits += 1
            score += hits / rank
        seen.add(item)
    return score / min(k, len(truth))


def mean_average_precision_at_k(
    predictions: Mapping[int, Sequence[int]],
    relevant_by_user: Mapping[int, Collection[int]],
    k: int = 5,
) -> float:
    """Mean over users with nonempty truth, never silently add zero-truth users."""
    scores = [
        average_precision_at_k(predictions.get(user, ()), relevant, k)
        for user, relevant in relevant_by_user.items() if relevant
    ]
    if not scores:
        raise ValueError("No users with relevant items to evaluate")
    return sum(scores) / len(scores)


@dataclass(frozen=True)
class ValidationProtocol:
    """Reuse this contract for every method; B truth is used only for scoring."""

    test_users: tuple[int, ...]
    candidates: frozenset[int]
    relevant_by_user: dict[int, frozenset[int]]
    k: int = 5

    @classmethod
    def from_interactions(
        cls, test_users: Iterable[int], schedule_b: pd.DataFrame, watched_b: pd.DataFrame
    ) -> "ValidationProtocol":
        users = tuple(map(int, test_users))
        if len(set(users)) != len(users):
            raise ValueError("Duplicate test users")
        candidates = frozenset(map(int, schedule_b.loc[schedule_b.tv_show_id != 0, "tv_show_id"]))
        truth_rows = watched_b.loc[watched_b.user_id.isin(users)]
        truth = {
            int(user): frozenset(map(int, items))
            for user, items in truth_rows.groupby("user_id", sort=False).tv_show_id
        }
        # Keep truth intact even for an airing carried in from A whose programme
        # never starts again in B. The candidate pool must not redefine relevance.
        if len(candidates) < 5:
            raise ValueError("Need at least five labelled candidate programmes")
        return cls(users, candidates, truth)

    def finalize_predictions(
        self,
        predictions: Mapping[int, Sequence[int]],
        popularity_ranking: Sequence[int],
        history_users: Collection[int],
    ) -> dict[int, list[int]]:
        """Candidate filtering, unique top five, popularity fill and cold start.

        Previously watched programmes remain eligible (series can be re-watched).
        The fallback ranking and history users must come exclusively from the
        fitting window: A for validation, weeks 11-30 for final submissions.
        """
        history = set(history_users)
        fallback = list(dict.fromkeys(int(item) for item in popularity_ranking if item in self.candidates))
        if len(fallback) < self.k:
            raise ValueError("Popularity fallback has fewer than five valid candidates")
        completed = {}
        for user in self.test_users:
            proposed = predictions.get(user, ()) if user in history else ()
            items, seen = [], set()
            for sequence in (proposed, fallback):
                for item in sequence:
                    item = int(item)
                    if item in self.candidates and item not in seen:
                        items.append(item)
                        seen.add(item)
                    if len(items) == self.k:
                        break
                if len(items) == self.k:
                    break
            completed[user] = items
        return completed

    def score(self, predictions: Mapping[int, Sequence[int]]) -> float:
        # Validate all 1,260 users, including those with no B truth.
        if set(predictions) != set(self.test_users):
            raise ValueError("Predictions must cover exactly the test-user cohort")
        for user, items in predictions.items():
            if len(items) != self.k or len(set(items)) != self.k or not set(items).issubset(self.candidates):
                raise ValueError(f"Invalid top-{self.k} recommendations for {user}")
        return mean_average_precision_at_k(predictions, self.relevant_by_user, self.k)

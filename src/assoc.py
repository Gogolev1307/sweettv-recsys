"""Sparse user-basket association rules; max-confidence consequent scoring."""

from __future__ import annotations

import numpy as np
from scipy import sparse


class AssociationRecommender:
    def __init__(self, minimum_support=20, minimum_lift=1.0):
        self.minimum_support, self.minimum_lift = minimum_support, minimum_lift

    def fit(self, seconds: sparse.csr_matrix) -> "AssociationRecommender":
        self.baskets = seconds.copy()
        self.baskets.data = np.ones_like(self.baskets.data, dtype=np.float64)
        support = np.asarray(self.baskets.sum(axis=0)).ravel()
        self.basket_count = int((np.diff(self.baskets.indptr) > 0).sum())
        pairs = (self.baskets.T @ self.baskets).tocoo()
        valid = (pairs.row != pairs.col) & (pairs.data >= self.minimum_support)
        rows, columns, counts = pairs.row[valid], pairs.col[valid], pairs.data[valid]
        confidence = counts / support[rows]
        lift = counts * self.basket_count / (support[rows] * support[columns])
        keep = lift > self.minimum_lift
        self.rules = sparse.csr_matrix(
            (confidence[keep], (rows[keep], columns[keep])), shape=(seconds.shape[1], seconds.shape[1])
        )
        self.rule_count = self.rules.nnz
        return self

    def score(self, user_rows, candidate_columns):
        candidates = self.rules[:, candidate_columns].tocsr()
        scores = np.zeros((len(user_rows), len(candidate_columns)), dtype=np.float64)
        # Loop over target users, never over item pairs; rule discovery is X.T@X.
        for index, user in enumerate(user_rows):
            start, end = self.baskets.indptr[user:user + 2]
            antecedents = self.baskets.indices[start:end]
            if len(antecedents):
                scores[index] = candidates[antecedents].max(axis=0).toarray().ravel()
        return scores

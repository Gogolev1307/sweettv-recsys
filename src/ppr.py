"""Batched bipartite personalised PageRank with restart-based dangling repair."""

from __future__ import annotations

import numpy as np
from scipy import sparse

from src.matrix import log_weights


def row_normalize(matrix: sparse.csr_matrix) -> tuple[sparse.csr_matrix, np.ndarray]:
    totals = np.asarray(matrix.sum(axis=1)).ravel()
    inverse = np.divide(1.0, totals, out=np.zeros_like(totals), where=totals != 0)
    return (sparse.diags(inverse) @ matrix).tocsr(), totals == 0


class PersonalizedPageRank:
    def __init__(self, tolerance=1e-6, max_iterations=500, batch_size=128):
        self.tolerance, self.max_iterations, self.batch_size = tolerance, max_iterations, batch_size

    def fit(self, seconds: sparse.csr_matrix) -> "PersonalizedPageRank":
        weights = log_weights(seconds)
        self.user_to_item, self.dangling_users = row_normalize(weights)
        self.item_to_user, self.dangling_items = row_normalize(weights.T.tocsr())
        self.to_users = self.item_to_user.T.tocsr()
        self.to_items = self.user_to_item.T.tocsr()
        errors = []
        for block, dangling in ((self.user_to_item, self.dangling_users), (self.item_to_user, self.dangling_items)):
            sums = np.asarray(block.sum(axis=1)).ravel()
            errors.append(float(np.max(np.abs(sums - (~dangling).astype(float)), initial=0)))
        self.diagnostics = {
            "user_row_sum_error": errors[0], "item_row_sum_error": errors[1],
            "dangling_users": int(self.dangling_users.sum()), "dangling_items": int(self.dangling_items.sum()),
        }
        return self

    def update(self, users: np.ndarray, items: np.ndarray, targets: np.ndarray, alpha: float):
        """alpha is the follow-link probability; restart probability is 1-alpha."""
        dangling_mass = users[self.dangling_users].sum(axis=0) + items[self.dangling_items].sum(axis=0)
        next_users = alpha * (self.to_users @ items)
        next_items = alpha * (self.to_items @ users)
        next_users[targets, np.arange(len(targets))] += 1.0 - alpha + alpha * dangling_mass
        return next_users, next_items

    def one_update_mass_check(self, target: int, alpha: float = 0.85) -> tuple[float, float]:
        users = np.zeros((self.user_to_item.shape[0], 1))
        items = np.zeros((self.user_to_item.shape[1], 1))
        users[target, 0] = 1.0
        before = float(users.sum() + items.sum())
        users, items = self.update(users, items, np.array([target]), alpha)
        return before, float(users.sum() + items.sum())

    def score(self, user_rows: np.ndarray, candidate_columns: np.ndarray, alpha: float):
        if not 0 < alpha < 1:
            raise ValueError("PPR alpha must be between zero and one")
        scores = np.empty((len(user_rows), len(candidate_columns)), dtype=np.float64)
        iterations, residuals, mass_errors = [], [], []
        non_converged = 0
        for start in range(0, len(user_rows), self.batch_size):
            targets = user_rows[start:start + self.batch_size]
            users = np.zeros((self.user_to_item.shape[0], len(targets)), dtype=np.float64)
            items = np.zeros((self.user_to_item.shape[1], len(targets)), dtype=np.float64)
            users[targets, np.arange(len(targets))] = 1.0  # p0 = restart
            first_converged = np.zeros(len(targets), dtype=np.int64)
            for iteration in range(1, self.max_iterations + 1):
                next_users, next_items = self.update(users, items, targets, alpha)
                # Freeze each column at its own stopping iteration while the
                # remaining targets continue in the same sparse-dense batch.
                inactive = first_converged != 0
                next_users[:, inactive] = users[:, inactive]
                next_items[:, inactive] = items[:, inactive]
                delta = np.abs(next_users - users).sum(axis=0) + np.abs(next_items - items).sum(axis=0)
                first_converged[(first_converged == 0) & (delta <= self.tolerance)] = iteration
                users, items = next_users, next_items
                if np.all(delta <= self.tolerance):
                    break
            non_converged += int((first_converged == 0).sum())
            first_converged[first_converged == 0] = self.max_iterations
            # Report the true fixed-point residual of the returned vector.
            next_users, next_items = self.update(users, items, targets, alpha)
            residuals.extend(np.abs(next_users - users).sum(axis=0) + np.abs(next_items - items).sum(axis=0))
            mass_errors.extend(np.abs(users.sum(axis=0) + items.sum(axis=0) - 1.0))
            iterations.extend(first_converged)
            scores[start:start + len(targets)] = items[candidate_columns].T
        diagnostics = {
            "mean_iterations": float(np.mean(iterations)), "max_iterations": int(np.max(iterations)),
            "max_final_l1_residual": float(np.max(residuals)), "non_converged_users": non_converged,
            "max_final_mass_error": float(np.max(mass_errors)),
        }
        if diagnostics["max_final_mass_error"] > 1e-10:
            raise AssertionError("PPR failed to conserve probability mass")
        return scores, diagnostics

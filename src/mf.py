"""Hu, Koren & Volinsky implicit ALS with exact NumPy normal-equation solves."""

from __future__ import annotations

import numpy as np
from scipy import sparse

from src.matrix import log_weights


def least_squares(observed: sparse.csr_matrix, fixed: np.ndarray, reg: float) -> np.ndarray:
    """observed stores c_ui - 1; unobserved entries have c=1 and p=0."""
    factors = fixed.shape[1]
    gram = fixed.T @ fixed + reg * np.eye(factors)
    result = np.zeros((observed.shape[0], factors), dtype=np.float64)
    for row in range(observed.shape[0]):
        start, end = observed.indptr[row:row + 2]
        if start == end:
            continue
        vectors = fixed[observed.indices[start:end]]
        extra = observed.data[start:end]
        lhs = gram + (vectors.T * extra) @ vectors
        rhs = vectors.T @ (1.0 + extra)  # p=1 for every observed pair
        result[row] = np.linalg.solve(lhs, rhs)
    return result


class ImplicitALS:
    def __init__(self, factors=16, reg=1.0, alpha=1.0, iterations=10, seed=42):
        if factors <= 0 or reg <= 0 or alpha <= 0 or iterations <= 0:
            raise ValueError("ALS factors, regularization, alpha and iterations must be positive")
        self.factors, self.reg, self.alpha = factors, reg, alpha
        self.iterations, self.seed = iterations, seed

    def fit(self, seconds: sparse.csr_matrix) -> "ImplicitALS":
        extra = log_weights(seconds)
        extra.data *= self.alpha
        transpose = extra.T.tocsr()
        rng = np.random.default_rng(self.seed)
        self.item_factors = rng.normal(0.0, 0.01, (seconds.shape[1], self.factors))
        for _ in range(self.iterations):
            self.user_factors = least_squares(extra, self.item_factors, self.reg)
            self.item_factors = least_squares(transpose, self.user_factors, self.reg)
        return self

    def score(self, user_rows: np.ndarray, candidate_columns: np.ndarray) -> np.ndarray:
        return self.user_factors[user_rows] @ self.item_factors[candidate_columns].T

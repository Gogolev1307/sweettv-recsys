"""Metadata cosine profiles with vocabulary/IDF fitted on fitting-period metadata."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from scipy import sparse

from src.matrix import log_weights

CONTENT_FIELDS = [
    "tv_show_category", "tv_show_genre_1", "tv_show_genre_2", "tv_show_genre_3",
    "director", "actors", "year_of_production",
]


def metadata_tokens(row: dict) -> list[str]:
    tokens = set()
    for field in CONTENT_FIELDS:
        value = row.get(field)
        if value is None or pd.isna(value) or not str(value).strip():
            continue
        text = " ".join(str(value).casefold().split())
        if field == "year_of_production":
            match = re.search(r"(?:18|19|20)\d{2}", text)
            if match:
                tokens.add(f"decade:{int(match.group()) // 10 * 10}")
        else:
            family = "genre" if field.startswith("tv_show_genre") else field
            values = text.split(",") if field in ("director", "actors") else [text]
            tokens.update(f"{family}:{value.strip()}" for value in values if value.strip())
    return sorted(tokens)


def l2_normalize(matrix: sparse.csr_matrix) -> sparse.csr_matrix:
    norms = np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel())
    inverse = np.divide(1.0, norms, out=np.zeros_like(norms), where=norms != 0)
    return (sparse.diags(inverse) @ matrix).tocsr()


class ContentRecommender:
    def fit(self, seconds, item_ids, metadata_a: pd.DataFrame, metadata_available: pd.DataFrame):
        """Fit IDF on metadata_a, then transform metadata_available.

        metadata_a is A for validation, or all weeks 11-30 for the final refit.
        Validation B-only metadata is transformed, never used to fit IDF.
        """
        documents_a = [metadata_tokens(row) for row in metadata_a.to_dict("records")]
        vocabulary = sorted({token for document in documents_a for token in document})
        self.vocabulary = {token: column for column, token in enumerate(vocabulary)}

        def counts(documents):
            rows, columns = [], []
            for index, document in enumerate(documents):
                present = [self.vocabulary[token] for token in document if token in self.vocabulary]
                rows.extend([index] * len(present))
                columns.extend(present)
            return sparse.csr_matrix(
                (np.ones(len(rows)), (rows, columns)), shape=(len(documents), len(vocabulary)), dtype=np.float64
            )

        fit_counts = counts(documents_a)
        document_frequency = np.asarray((fit_counts != 0).sum(axis=0)).ravel()
        self.idf = np.log((1.0 + len(documents_a)) / (1.0 + document_frequency)) + 1.0
        lookup = metadata_available.set_index("tv_show_id").to_dict("index")
        documents = [metadata_tokens(lookup.get(int(item), {})) for item in item_ids]
        self.item_vectors = l2_normalize(counts(documents).multiply(self.idf).tocsr())
        self.weights = log_weights(seconds)
        self.zero_vector_items = int((np.diff(self.item_vectors.indptr) == 0).sum())
        return self

    def score(self, user_rows, candidate_columns):
        profiles = l2_normalize((self.weights[user_rows] @ self.item_vectors).tocsr())
        return (profiles @ self.item_vectors[candidate_columns].T).toarray()

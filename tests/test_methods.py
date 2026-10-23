import unittest

import numpy as np
import pandas as pd
from scipy import sparse

from src.assoc import AssociationRecommender
from src.cbf import ContentRecommender, metadata_tokens
from src.hybrid import fuse_scores, minmax_normalize, normalize_components, weight_grid
from src.matrix import InteractionMatrix, rank_scores
from src.mf import ImplicitALS, least_squares
from src.ppr import PersonalizedPageRank


class MethodTests(unittest.TestCase):
    def test_seconds_matrix_aggregates_airings_and_preserves_uint64(self):
        user = 2**64 - 2
        watched = pd.DataFrame({
            "user_id": pd.Series([user, user, 7], dtype="uint64"),
            "tv_show_id": [2, 2, 1], "overlap_seconds": [600, 900, 300],
        })
        matrix = InteractionMatrix.build(watched, [user, 8], [1, 2, 3, 4, 5])
        self.assertEqual(int(matrix.user_ids[matrix.test_rows[0]]), user)
        self.assertEqual(matrix.seconds[matrix.test_rows[0], 1], 1500)
        self.assertEqual(matrix.seconds[matrix.test_rows[1]].nnz, 0)

    def test_rank_large_ties_and_sparse_evidence(self):
        scores = np.array([[2.] * 6, [1., 0, 0, 0, 0, 0]])
        self.assertEqual(rank_scores(scores, np.arange(1, 7), [1, 2]), {1: [1, 2, 3, 4, 5], 2: [1]})

    def test_als_normal_equation_matches_dense_confidence_objective(self):
        vectors = np.array([[1., 2.], [2., -1.], [0.5, 1.]])
        extra = sparse.csr_matrix([[2., 0., 4.]])
        confidence, preference = np.diag([3., 1., 5.]), np.array([1., 0., 1.])
        expected = np.linalg.solve(vectors.T @ confidence @ vectors + 10 * np.eye(2), vectors.T @ confidence @ preference)
        np.testing.assert_allclose(least_squares(extra, vectors, 10)[0], expected)
        seconds = sparse.csr_matrix([[600., 1200., 0], [0, 600., 0], [0, 0, 0]])
        first = ImplicitALS(factors=2, iterations=2).fit(seconds)
        second = ImplicitALS(factors=2, iterations=2).fit(seconds)
        np.testing.assert_array_equal(first.user_factors, second.user_factors)
        np.testing.assert_array_equal(first.item_factors[2], [0, 0])
        np.testing.assert_array_equal(first.user_factors[2], [0, 0])

    def test_ppr_against_dense_fixed_point_including_dangling_restart(self):
        seconds = sparse.csr_matrix([[600., 1200., 0, 0], [0, 600., 300., 0], [0, 0, 0, 0]])
        model = PersonalizedPageRank(tolerance=1e-11, batch_size=2).fit(seconds)
        scores, diagnostics = model.score(np.array([0, 1, 2]), np.arange(4), 0.7)
        base = np.block([[np.zeros((3, 3)), model.user_to_item.toarray()],
                         [model.item_to_user.toarray(), np.zeros((4, 4))]])
        for user in range(3):
            restart = np.eye(7)[user]
            transition = base.copy()
            transition[transition.sum(axis=1) == 0] = restart
            expected = 0.3 * np.linalg.solve(np.eye(7) - 0.7 * transition.T, restart)
            np.testing.assert_allclose(scores[user], expected[3:], atol=1e-10)
        self.assertEqual(diagnostics["non_converged_users"], 0)
        self.assertLess(diagnostics["max_final_l1_residual"], 1e-11)
        self.assertEqual(model.one_update_mass_check(2), (1.0, 1.0))

    def test_association_support_lift_and_max_not_sum(self):
        seconds = sparse.csr_matrix([[1., 1, 1, 0], [1, 1, 1, 0], [0, 0, 0, 1], [0, 0, 0, 1]])
        model = AssociationRecommender(minimum_support=2).fit(seconds)
        self.assertEqual(model.rule_count, 6)
        np.testing.assert_array_equal(model.score(np.array([0, 2]), np.arange(4)), [[1, 1, 1, 0], [0, 0, 0, 0]])

    def test_cbf_a_only_idf_unknown_metadata_and_cosine(self):
        a = pd.DataFrame([
            {"tv_show_id": 1, "tv_show_category": "Film", "actors": "Alice, Bob", "year_of_production": "1999-2001"},
            {"tv_show_id": 2, "tv_show_category": "Music"},
        ])
        available = pd.concat([a, pd.DataFrame([{"tv_show_id": 3, "tv_show_category": "New B category"}])], ignore_index=True)
        model = ContentRecommender().fit(sparse.csr_matrix([[600., 0, 0]]), np.array([1, 2, 3]), a, available)
        scores = model.score(np.array([0]), np.arange(3))
        np.testing.assert_allclose(scores, [[1, 0, 0]], atol=1e-12)
        self.assertIn("actors:alice", model.vocabulary)
        self.assertNotIn("tv_show_category:new b category", model.vocabulary)
        self.assertIn("decade:1990", metadata_tokens(a.iloc[0].to_dict()))

    def test_hybrid_candidate_minmax_constant_vectors_and_grid(self):
        raw = np.array([[2., 4., 6.], [3., 3., 3.]])
        np.testing.assert_array_equal(minmax_normalize(raw), [[0, .5, 1], [0, 0, 0]])
        components = normalize_components(np.array([0., 1., 2.]), raw, raw, raw)
        np.testing.assert_array_equal(fuse_scores(components, (1, 0, 0, 1)), [[0, 1, 2], [0, .5, 1]])
        self.assertEqual(len(list(weight_grid())), 255)


if __name__ == "__main__":
    unittest.main()

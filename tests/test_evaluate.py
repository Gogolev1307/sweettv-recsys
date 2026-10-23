import unittest

import pandas as pd

from src.evaluate import ValidationProtocol, average_precision_at_k, mean_average_precision_at_k


class EvaluationTests(unittest.TestCase):
    def test_hand_calculated_ap(self):
        # Hits at ranks 1, 2, 4: (1 + 1 + 3/4) / 3 = 11/12.
        self.assertAlmostEqual(average_precision_at_k(["A", "B", "D", "C", "E"], {"A", "B", "C"}), 11 / 12)

    def test_duplicate_gets_no_second_credit(self):
        self.assertAlmostEqual(average_precision_at_k(["A", "A", "B"], {"A", "B"}), (1 + 2 / 3) / 2)

    def test_map_denominator_and_cohort(self):
        predictions = {1: [1, 2, 3, 4, 5], 2: [9, 8, 7, 6, 5], 3: [1]}
        relevant = {1: set(range(1, 7)), 2: {8}, 3: set()}
        self.assertAlmostEqual(mean_average_precision_at_k(predictions, relevant), (1 + 0.5) / 2)

    def test_shared_fallback_and_cold_start(self):
        protocol = ValidationProtocol((1, 2), frozenset(range(1, 7)), {1: frozenset({1})})
        result = protocol.finalize_predictions({1: [2, 2, 99], 2: [6]}, [1, 2, 3, 4, 5, 6], {1})
        self.assertEqual(result, {1: [2, 1, 3, 4, 5], 2: [1, 2, 3, 4, 5]})
        self.assertEqual(protocol.score(result), 0.5)

    def test_uint64_cohort_and_truth_outside_candidates(self):
        user = 2**64 - 2
        schedule = pd.DataFrame({"tv_show_id": [0, 1, 2, 3, 4, 5]})
        watched = pd.DataFrame({"user_id": pd.Series([user, user], dtype="uint64"), "tv_show_id": [1, 9]})
        protocol = ValidationProtocol.from_interactions([user], schedule, watched)
        self.assertEqual(protocol.relevant_by_user, {user: frozenset({1, 9})})
        self.assertEqual(protocol.score({user: [1, 2, 3, 4, 5]}), 0.5)


if __name__ == "__main__":
    unittest.main()

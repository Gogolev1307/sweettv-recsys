import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.data import CACHE_DIR
from src.join import join_views_to_airings, watched_interactions
from src.submission import (
    METHODS, load_frozen_configurations, snapshot_submissions, validate_submission,
    verify_preserved_submissions, write_or_preserve_submission, write_submission,
)


class SubmissionTests(unittest.TestCase):
    def setUp(self):
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="submission_test_", dir=CACHE_DIR)
        self.path = Path(self.temporary.name) / "submission.csv"
        self.users = [2**64 - 2, 7]
        self.pool = set(range(1, 8))

    def tearDown(self):
        self.temporary.cleanup()

    def test_uint64_round_trip_template_order_and_cold_fallback(self):
        predictions = {7: [1, 2, 3, 4, 5], self.users[0]: [7, 6, 5, 4, 3]}
        write_submission(self.path, self.users, predictions)
        checks = validate_submission(self.path, self.users, self.pool, {7}, [1, 2, 3, 4, 5])
        self.assertEqual(checks["rows"], 2)
        self.assertEqual(checks["cold_start_fallback_users"], 1)
        self.assertEqual(self.path.read_text().splitlines()[1], "18446744073709551614,7 6 5 4 3")

    def test_reject_malformed_written_files(self):
        first = str(self.users[0])
        valid = f"user_id,tv_show_id\n{first},1 2 3 4 5\n7,1 2 3 4 5\n"
        invalid = {
            "float_user": valid.replace(first, "18446744073709551614.0"),
            "scientific_user": valid.replace(first, "1.8446744073709552e19"),
            "wrong_order": f"user_id,tv_show_id\n7,1 2 3 4 5\n{first},1 2 3 4 5\n",
            "duplicate_items": valid.replace("1 2 3 4 5", "1 2 3 4 4"),
            "zero_item": valid.replace("1 2 3 4 5", "0 2 3 4 5"),
            "unavailable_item": valid.replace("1 2 3 4 5", "1 2 3 4 8"),
            "wrong_cold_fallback": valid.replace("7,1 2 3 4 5", "7,2 1 3 4 5"),
            "wrong_spacing": valid.replace("1 2 3 4 5", "1  2 3 4 5"),
        }
        for label, text in invalid.items():
            with self.subTest(label=label):
                self.path.write_text(text)
                with self.assertRaises(ValueError):
                    validate_submission(self.path, self.users, self.pool, {7}, [1, 2, 3, 4, 5])

    def test_configuration_reader_uses_selected_flag_without_retuning(self):
        rows = [
            {"method": method, "hyperparameters": json.dumps({"seed": 42, "factors": 4} if method == "MF" else {}),
             "selected": True, "seed": 42, "evaluated_users": 1246, "map_at_5": .1}
            for method in METHODS
        ]
        rows.append({"method": "MF", "hyperparameters": '{"seed":42,"factors":32}', "selected": False,
                     "seed": 42, "evaluated_users": 1246, "map_at_5": 1.0})
        pd.DataFrame(rows).to_csv(self.path, index=False)
        configurations, digest = load_frozen_configurations(self.path)
        self.assertEqual(configurations["MF"]["factors"], 4)
        self.assertEqual(len(digest), 64)

    def test_existing_submissions_are_preserved_even_for_identical_predictions(self):
        path = self.path.with_name("submission_old.csv")
        predictions = {self.users[0]: [1, 2, 3, 4, 5], 7: [1, 2, 3, 4, 5]}
        self.assertTrue(write_or_preserve_submission(path, self.users, predictions))
        snapshot = snapshot_submissions(path.parent, "submission_hybrid_v2.csv")
        self.assertFalse(write_or_preserve_submission(path, self.users, predictions))
        verify_preserved_submissions(path.parent, snapshot)
        predictions[7] = [2, 1, 3, 4, 5]
        with self.assertRaises(ValueError):
            write_or_preserve_submission(path, self.users, predictions)
        verify_preserved_submissions(path.parent, snapshot)
        path.write_text(path.read_text() + "\n")
        with self.assertRaises(AssertionError):
            verify_preserved_submissions(path.parent, snapshot)

    def test_full_period_threshold_combines_boundary_pieces_before_filtering(self):
        boundary = pd.Timestamp("2020-05-18")
        start, end = boundary - pd.Timedelta(minutes=5), boundary + pd.Timedelta(minutes=5)
        views = pd.DataFrame({
            "user_id": pd.Series([2**64 - 2], dtype="uint64"), "vsetv_id": [7],
            "start_time": [boundary - pd.Timedelta(minutes=3)],
            "stop_time": [boundary + pd.Timedelta(minutes=3)],
        })
        schedule = pd.DataFrame({
            "airing_id": [1], "channel_id": [7], "tv_show_id": [99],
            "start_time": [start], "end_time": [end], "duration": [600],
        })
        a = join_views_to_airings(views, schedule, start, boundary)
        b = join_views_to_airings(views, schedule, boundary, end)
        full = join_views_to_airings(views, schedule, start, end)
        self.assertEqual(len(watched_interactions(a.interactions)), 0)
        self.assertEqual(len(watched_interactions(b.interactions)), 0)
        self.assertEqual(watched_interactions(full.interactions).overlap_seconds.tolist(), [360])


if __name__ == "__main__":
    unittest.main()

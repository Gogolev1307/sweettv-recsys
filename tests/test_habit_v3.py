import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from src.data import CACHE_DIR
from src.exposure import ExposureRecommender, scheduled_day_counts
from src.hybrid import (
    V2_SELECTION_RULE, V3_SELECTION_RULE, calibrated_truth_mean, days_weight_grid,
    fuse_habit_scores, habit_weight_grid, normalize_habit_components,
)
from src.slot import SlotRecommender
from src.submission import METHODS, load_frozen_configurations, load_frozen_v3_configuration


def airings(rows):
    frame = pd.DataFrame(rows, columns=["tv_show_id", "channel_id", "start_time", "duration"])
    frame["start_time"] = pd.to_datetime(frame.start_time)
    frame["end_time"] = frame.start_time + pd.to_timedelta(frame.duration, unit="s")
    return frame


class HabitV3Tests(unittest.TestCase):
    def test_slot_uses_all_fit_views_hour_splits_weekends_and_no_future_views(self):
        user = 2**64 - 2
        views = pd.DataFrame({
            "user_id": pd.Series([user, 7, user], dtype="uint64"), "vsetv_id": [7, 7, 99],
            "start_time": pd.to_datetime(["2020-03-13 23:30", "2020-03-14 00:30", "2020-03-21 12:00"]),
            "stop_time": pd.to_datetime(["2020-03-14 00:30", "2020-03-14 00:31", "2020-03-21 13:00"]),
        })
        # The 60-second view is deliberately below the labelled-airing threshold;
        # slot habits must include it without needing any labelled schedule join.
        model = SlotRecommender(chunk_size=1).fit(views, [user, 7, 8], pd.Timestamp("2020-03-13"), pd.Timestamp("2020-03-16"))
        self.assertEqual(model.diagnostics["view_seconds"], 3660)
        self.assertEqual(model.diagnostics["view_intervals"], 2)
        np.testing.assert_array_equal(model.channels, [7])
        row = int(np.searchsorted(model.user_ids, np.uint64(user)))
        profile = model.profile.getrow(row).toarray().ravel()
        self.assertEqual(set(np.flatnonzero(profile)), {23, 24})
        self.assertEqual(profile[23], .5)
        self.assertEqual(profile[24], .5)
        target = airings([
            (10, 7, "2020-03-21 00:00", 600), (11, 7, "2020-03-20 23:00", 1200),
            (12, 7, "2020-03-21 00:00", 1200), (13, 99, "2020-03-21 00:00", 3600),
            (14, 7, "2020-03-21 01:00", 600),
        ])
        target["actors"] = ["ignored target metadata"] * len(target)
        scores = model.score([user, 7, 8], np.arange(10, 15), target, pd.Timestamp("2020-03-20"), pd.Timestamp("2020-03-23"))
        np.testing.assert_allclose(scores, [[300, 600, 600, 0, 0], [600, 0, 1200, 0, 0], [0, 0, 0, 0, 0]])
        self.assertLess(model.diagnostics["max_row_sum_error"], 1e-12)

    def test_exposure_counts_unique_airing_dates_and_uses_smoothed_rate(self):
        fitting = []
        for day in range(9, 14):
            for channel in (7, 8):
                fitting.append((10, channel, f"2020-03-{day:02d} 12:00", 600))
        fitting.extend([(20, 7, "2020-03-12 23:55", 600), (30, 7, "2020-03-10 12:00", 300)])
        fitting = airings(fitting)
        ids = np.array([10, 20, 30, 40, 50])
        fit_start, fit_end = pd.Timestamp("2020-03-09"), pd.Timestamp("2020-03-14")
        np.testing.assert_array_equal(scheduled_day_counts(fitting, ids, fit_start, fit_end), [5, 2, 1, 0, 0])
        target = airings([
            (10, 7, "2020-03-16 12:00", 600), (10, 8, "2020-03-16 15:00", 600),
            (10, 7, "2020-03-17 12:00", 600), (10, 7, "2020-03-18 12:00", 600),
            (20, 7, "2020-03-17 23:55", 600), (30, 7, "2020-03-16 14:00", 300),
            (40, 7, "2020-03-16 15:00", 300),
        ])
        days = sparse.csr_matrix([[2., 1, 0, 0, 0], [0, 0, 0, 0, 0]])
        model = ExposureRecommender().fit(days, ids, fitting, fit_start, fit_end)
        scores = model.score(np.array([0, 1]), np.arange(5), target, pd.Timestamp("2020-03-16"), pd.Timestamp("2020-03-19"))
        np.testing.assert_allclose(scores, [[.75, .4, 0, 0, 0], [0, 0, 0, 0, 0]])
        with self.assertRaises(AssertionError):
            ExposureRecommender().fit(sparse.csr_matrix([[0., 0, 0, 1, 0]]), ids, fitting, fit_start, fit_end)

    def test_v3_grid_normalization_and_selection_ignore_t1_t4(self):
        grid = list(habit_weight_grid())
        self.assertEqual(len(grid), 1535)
        self.assertTrue(all(any(weights.values()) for weights in grid))
        self.assertTrue(all(weights["w_days"] in (0, .5, 1, 2) and weights["w_pop"] in (0, .1, .3) for weights in grid))
        left = {"T1": 1, "T2": .1, "T3": .3, "T4": 1}
        right = {"T1": 0, "T2": .3, "T3": .3, "T4": 0}
        self.assertAlmostEqual(calibrated_truth_mean(left), .2)
        self.assertGreater(calibrated_truth_mean(right), calibrated_truth_mean(left))
        components = normalize_habit_components(
            np.array([[0., 2, 4]]), np.array([[10., 20, 30]]), np.array([[7., 7, 7]]),
            np.array([[0., .1, .2]]), np.array([[-1., 0, 1]]), np.array([0., 1, 2]),
        )
        weights = {"w_days": 1, "w_exp": .5, "w_slot": 2, "w_ppr": 0, "w_mf": .2, "w_pop": .1}
        np.testing.assert_allclose(fuse_habit_scores(components, weights), [[0, .9, 1.8]])

    def test_v3_freeze_checks_t2_t3_objective_component_chain_and_availability_schema(self):
        CACHE_DIR.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=CACHE_DIR, prefix="habit_v3_test_") as directory:
            root = Path(directory)
            component = {"MF": {"seed": 42, "factors": 4}, "PPR": {"alpha": .8}}
            pd.DataFrame([
                {"method": method, "selected": True, "seed": 42, "evaluated_users": 1246,
                 "hyperparameters": json.dumps(component.get(method, {}))} for method in METHODS
            ]).to_csv(root / "validation_methods.csv", index=False)
            _, legacy_digest = load_frozen_configurations(root / "validation_methods.csv")
            v2_weights = list(days_weight_grid())[0]
            v2_csv = root / "validation_methods_v2.csv"
            pd.DataFrame([{"method": "hybrid_v2", "selected": True, "hyperparameters": json.dumps(v2_weights),
                           "strict_mean_map_at_5": .4}]).to_csv(v2_csv, index=False)
            v2_manifest = {
                "seed": 42, "threshold_seconds": 300, "selection_rule": V2_SELECTION_RULE,
                "weights": v2_weights, "component_configurations": component,
                "validation_results_file": v2_csv.name, "validation_results_sha256": hashlib.sha256(v2_csv.read_bytes()).hexdigest(),
                "strict_mean_map_at_5": .4, "validation_scores": {"T1": .9, "T2": .4, "T3": .4, "T4": .4},
                "legacy_config_sha256": legacy_digest,
            }
            v2_path = root / "hybrid_v2_selected.json"
            v2_path.write_text(json.dumps(v2_manifest))
            weights = list(habit_weight_grid())[0]
            v3_csv = root / "validation_methods_v3.csv"
            pd.DataFrame([{"method": "hybrid_v3", "selected": True, "hyperparameters": json.dumps(weights),
                           "selection_mean_T2_T3": .5}]).to_csv(v3_csv, index=False)
            manifest = {
                "seed": 42, "threshold_seconds": 300, "selection_rule": V3_SELECTION_RULE,
                "weights": weights, "component_configurations": component,
                "selection_mean_T2_T3": .5, "validation_scores": {"T1": .9, "T2": .3, "T3": .7, "T4": .1},
                "validation_results_file": v3_csv.name, "validation_results_sha256": hashlib.sha256(v3_csv.read_bytes()).hexdigest(),
                "legacy_config_sha256": legacy_digest, "v2_selection_manifest_sha256": hashlib.sha256(v2_path.read_bytes()).hexdigest(),
                "exposure_smoothing_days": 3, "target_schedule_fields": ["channel_id", "start_time", "duration", "tv_show_id"],
            }
            path = root / "hybrid_v3_selected.json"
            path.write_text(json.dumps(manifest))
            loaded, _ = load_frozen_v3_configuration(path)
            self.assertEqual(loaded["weights"], weights)
            manifest["target_schedule_fields"].append("actors")
            path.write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                load_frozen_v3_configuration(path)


if __name__ == "__main__":
    unittest.main()

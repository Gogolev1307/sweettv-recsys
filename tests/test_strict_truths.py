import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from src.data import CACHE_DIR
from src.hybrid import (
    V2_SELECTION_RULE, days_weight_grid, fuse_days_scores, normalize_days_components, strict_truth_mean,
)
from src.join import JoinResult, join_views_to_airings, programme_day_counts
from src.rewatch import RewatchDaysRecommender
from src.submission import METHODS, load_frozen_configurations, load_frozen_v2_configuration
from src.truths import validation_truths


class StrictTruthTests(unittest.TestCase):
    def test_actual_viewing_dates_midnight_and_per_airing_eligibility(self):
        user = 2**64 - 2
        dates = pd.to_datetime([
            "2020-05-17 23:50", "2020-05-18 12:00", "2020-05-18 23:55",
            "2020-05-19 23:55", "2020-05-19 10:00",
        ])
        schedule = pd.DataFrame({
            "airing_id": range(5), "channel_id": [7] * 5, "tv_show_id": [10, 10, 20, 30, 40],
            "start_time": dates, "duration": [2400, 600, 600, 600, 600],
        })
        schedule["end_time"] = schedule.start_time + pd.to_timedelta(schedule.duration, unit="s")
        views = pd.DataFrame({
            "user_id": pd.Series([user] * 5, dtype="uint64"), "vsetv_id": [7] * 5,
            "start_time": pd.to_datetime([
                "2020-05-18 00:05", "2020-05-18 12:00", "2020-05-18 23:57",
                "2020-05-19 23:55", "2020-05-19 10:00",
            ]),
            "stop_time": pd.to_datetime([
                "2020-05-18 00:15", "2020-05-18 12:05", "2020-05-19 00:03",
                "2020-05-20 00:00", "2020-05-19 10:04",
            ]),
        })
        result = join_views_to_airings(
            views, schedule, pd.Timestamp("2020-05-18"), pd.Timestamp("2020-05-20"),
            chunk_size=1, collect_days=True,
        )
        counts = programme_day_counts(result).set_index("tv_show_id").watched_days.to_dict()
        self.assertEqual(counts, {10: 1, 20: 2, 30: 1})
        self.assertEqual(set(map(int, result.daily_interactions.user_id)), {user})
        split = result.daily_interactions.loc[result.daily_interactions.tv_show_id.eq(20)]
        self.assertEqual(sorted(split.overlap_seconds), [180, 180])
        midnight = result.daily_interactions.loc[result.daily_interactions.tv_show_id.eq(30), "watch_day"]
        self.assertEqual(pd.to_datetime(midnight, unit="D").dt.strftime("%Y-%m-%d").tolist(), ["2020-05-19"])
        self.assertEqual(result.daily_interactions.overlap_seconds.sum(), result.interactions.overlap_seconds.sum())

    def test_four_truths_use_explicit_thresholds_days_topten_ties_and_cohorts(self):
        user = 2**64 - 2
        records, daily = [], []

        def add(target, item, seconds, day):
            airing = len(records)
            records.append({"user_id": target, "airing_id": airing, "tv_show_id": item, "overlap_seconds": seconds})
            daily.append({"user_id": target, "airing_id": airing, "tv_show_id": item,
                          "overlap_seconds": seconds, "watch_day": day})

        add(user, 10, 500, 0)
        for day in range(20):
            add(user, 20, 200, day)
        for day in range(5):
            add(user, 30, 300, day)
            add(user, 40, 300, 0)
        for item in range(100, 112):
            add(user, item, 1, 0)
        add(7, 50, 1, 0)
        interactions, days = pd.DataFrame(records), pd.DataFrame(daily)
        interactions["user_id"] = pd.Series([r["user_id"] for r in records], dtype="uint64")
        days["user_id"] = pd.Series([r["user_id"] for r in daily], dtype="uint64")
        protocols, stats = validation_truths([user, 7, 8], [10, 20, 30, 40, 50, *range(100, 112)],
                                            JoinResult(interactions, {}, days))
        self.assertEqual(protocols["T1"].relevant_by_user, {user: frozenset({10, 30, 40})})
        self.assertEqual(protocols["T2"].relevant_by_user, {user: frozenset({20})})
        self.assertEqual(protocols["T3"].relevant_by_user, {user: frozenset({30})})
        self.assertEqual(protocols["T4"].relevant_by_user[user], frozenset({10, 20, 30, 40, *range(100, 106)}))
        self.assertEqual(protocols["T4"].relevant_by_user[7], frozenset({50}))
        self.assertEqual(stats["T1"]["evaluated_users"], 1)
        self.assertEqual(stats["T1"]["median_relevant_all_users"], 0)
        self.assertEqual(stats["T1"]["median_relevant_evaluated_users"], 3)
        self.assertEqual(stats["T4"]["median_relevant_all_users"], 1)
        for values in stats.values():
            self.assertEqual(values["evaluated_users"] + values["no_relevant_users"], 3)

    def test_days_reference_uses_seconds_then_id_ties(self):
        days = sparse.csr_matrix([[2., 2, 3, 1, 2, 0]])
        seconds = sparse.csr_matrix([[3000., 6000, 900, 8000, 6000, 0]])
        model = RewatchDaysRecommender().fit(days, seconds)
        self.assertEqual(model.predict(np.array([0]), np.arange(6), np.arange(1, 7), [7]), {7: [3, 2, 5, 1, 4]})
        np.testing.assert_array_equal(model.score(np.array([0]), np.arange(6)), [[2, 2, 3, 1, 2, 0]])

    def test_v2_grid_formula_and_objective_exclude_t1(self):
        grid = list(days_weight_grid())
        self.assertEqual(len(grid), 24)
        self.assertTrue(all(weights["rewatch_days"] == 1 for weights in grid))
        left = {"T1": 1., "T2": .1, "T3": .2, "T4": .3}
        right = {"T1": 0., "T2": .3, "T3": .3, "T4": .3}
        self.assertGreater(strict_truth_mean(right), strict_truth_mean(left))
        self.assertAlmostEqual(strict_truth_mean(left), .2)
        components = normalize_days_components(
            np.array([[0., 2., 4.]]), np.array([[0., 1., 2.]]),
            np.array([0., 1., 2.]), np.array([[-1., 0., 1.]]),
        )
        result = fuse_days_scores(components, {"rewatch_days": 1., "w_ppr": .5, "w_pop": .3, "w_mf": .2})
        np.testing.assert_allclose(result, [[0., 1., 2.]])

    def test_v2_manifest_reads_frozen_selected_row_and_checks_source_integrity(self):
        CACHE_DIR.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=CACHE_DIR, prefix="truth_v2_test_") as directory:
            root = Path(directory)
            component = {"MF": {"seed": 42, "factors": 4}, "PPR": {"alpha": .8}}
            legacy = pd.DataFrame([
                {"method": name, "selected": True, "seed": 42, "evaluated_users": 1246,
                 "hyperparameters": json.dumps(component.get(name, {}))}
                for name in METHODS
            ])
            legacy.to_csv(root / "validation_methods.csv", index=False)
            _, legacy_digest = load_frozen_configurations(root / "validation_methods.csv")
            weights = list(days_weight_grid())[0]
            frame = pd.DataFrame([
                {"method": "hybrid_v2", "selected": True, "hyperparameters": json.dumps(weights), "strict_mean_map_at_5": .4},
                {"method": "hybrid_v2", "selected": False, "hyperparameters": json.dumps(list(days_weight_grid())[-1]),
                 "strict_mean_map_at_5": .5},
            ])
            csv_path = root / "validation_methods_v2.csv"
            frame.to_csv(csv_path, index=False)
            manifest = {
                "seed": 42, "threshold_seconds": 300, "selection_rule": V2_SELECTION_RULE,
                "weights": weights, "component_configurations": component,
                "validation_results_file": csv_path.name,
                "validation_results_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
                "strict_mean_map_at_5": .4, "validation_scores": {"T1": .9, "T2": .4, "T3": .4, "T4": .4},
                "legacy_config_sha256": legacy_digest,
            }
            path = root / "hybrid_v2_selected.json"
            path.write_text(json.dumps(manifest))
            loaded, _ = load_frozen_v2_configuration(path)
            self.assertEqual(loaded["weights"], weights)
            csv_path.write_text(csv_path.read_text() + "\n")
            with self.assertRaises(ValueError):
                load_frozen_v2_configuration(path)


if __name__ == "__main__":
    unittest.main()

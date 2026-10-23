import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.completion import CompletionRecommender, airing_channel_counts
from src.data import CACHE_DIR
from src.hybrid import (
    V2_SELECTION_RULE, V3_SELECTION_RULE, V4_SELECTION_RULE, competition_truth_score,
    completion_weight_grid, days_weight_grid, fuse_completion_scores, habit_weight_grid,
)
from src.join import JoinResult, completed_airings, join_views_to_airings
from src.submission import METHODS, load_frozen_configurations, load_frozen_v4_configuration
from src.truths import COMPETITION_GROUND_TRUTH_QUOTE, competition_truth


def interactions(records):
    frame = pd.DataFrame(records, columns=["user_id", "airing_id", "tv_show_id", "channel_id", "overlap_seconds", "programme_duration"])
    frame["user_id"] = pd.Series([row[0] for row in records], dtype="uint64")
    return frame


def schedule(rows):
    frame = pd.DataFrame(rows, columns=["tv_show_id", "channel_id", "start_time", "duration"])
    frame["start_time"] = pd.to_datetime(frame.start_time)
    frame["end_time"] = frame.start_time + pd.to_timedelta(frame.duration, unit="s")
    return frame


class CompletionV4Tests(unittest.TestCase):
    def test_tk_counts_airings_not_days_topfive_ties_and_small_truth_cohort(self):
        user = 2**64 - 2
        rows = []
        for item, repeats in ((10, 2), (20, 2), (30, 1), (40, 1), (50, 1), (60, 1), (70, 1)):
            for _ in range(repeats):
                rows.append((user, len(rows), item, 7, 800, 1000))
        rows.extend([(7, 100, 900, 7, 80, 100), (8, 101, 10, 7, 799, 1000), (user, 102, 0, 7, 1000, 1000)])
        frame = interactions(rows)
        protocol, stats = competition_truth([user, 7, 8], [10, 20, 30, 40, 50, 60, 70, 900], JoinResult(frame, {}))
        self.assertEqual(protocol.relevant_by_user[user], frozenset({10, 20, 30, 40, 50}))
        self.assertEqual(protocol.relevant_by_user[7], frozenset({900}))
        self.assertNotIn(8, protocol.relevant_by_user)
        self.assertEqual(stats["evaluated_users"], 2)
        self.assertEqual(stats["no_relevant_users"], 1)
        self.assertEqual(stats["relevance_size_counts"], {"0": 1, "1": 1, "2": 0, "3": 0, "4": 0, "5": 1})
        self.assertEqual(len(completed_airings(frame, 80)), 10)

    def test_completion_aggregates_sessions_and_uses_exact_full_duration_boundary(self):
        origin = pd.Timestamp("2020-03-09")
        views = pd.DataFrame({
            "user_id": pd.Series([1, 1, 2], dtype="uint64"), "vsetv_id": [7, 7, 7],
            "start_time": [origin, origin + pd.Timedelta(seconds=400), origin],
            "stop_time": [origin + pd.Timedelta(seconds=400), origin + pd.Timedelta(seconds=800), origin + pd.Timedelta(seconds=799)],
        })
        airings = schedule([(10, 7, origin, 1000)])
        airings.insert(0, "airing_id", [0])
        full = join_views_to_airings(views, airings, origin, origin + pd.Timedelta(hours=1))
        self.assertEqual(completed_airings(full.interactions, 80).user_id.tolist(), [1])
        self.assertEqual(len(completed_airings(full.interactions, 50)), 2)
        clipped = join_views_to_airings(views, airings, origin, origin + pd.Timedelta(seconds=500))
        # All 500 clipped seconds were watched, but the denominator is still 1000.
        self.assertEqual(len(completed_airings(clipped.interactions, 80)), 0)
        self.assertEqual(len(completed_airings(clipped.interactions, 50)), 2)

    def test_channel_aware_rates_airing_counts_short_completions_and_frequency_popularity(self):
        user = 2**64 - 2
        fit = schedule([
            (10, 7, "2020-03-09 00:00", 1000), (10, 7, "2020-03-10 00:00", 1000),
            (10, 8, "2020-03-11 00:00", 1000), (20, 7, "2020-03-09 01:00", 100),
            (30, 7, "2020-03-09 02:00", 1000), (0, 7, "2020-03-09 03:00", 1000),
        ])
        frame = interactions([
            (user, 0, 10, 7, 800, 1000), (user, 1, 10, 7, 500, 1000),
            (user, 2, 10, 8, 900, 1000), (7, 0, 10, 7, 799, 1000),
            (user, 3, 20, 7, 80, 100), (user, 4, 30, 7, 499, 1000), (user, 5, 0, 7, 1000, 1000),
        ])
        targets = []
        for channel, count in ((7, 3), (8, 2), (9, 10)):
            for index in range(count):
                targets.append((10, channel, pd.Timestamp("2020-03-16") + pd.Timedelta(hours=index), 1000))
        targets.extend([(20, 7, "2020-03-17 00:00", 100), (20, 7, "2020-03-17 01:00", 100)])
        target = schedule(targets)
        target = pd.concat([target, target.iloc[:1]], ignore_index=True)  # Duplicate export row is not another airing.
        start, end = pd.Timestamp("2020-03-09"), pd.Timestamp("2020-03-12")
        target_start, target_end = pd.Timestamp("2020-03-16"), pd.Timestamp("2020-03-18")
        counts = airing_channel_counts(target, target_start, target_end)
        self.assertEqual(counts.loc[(10, 7)], 3)
        model = CompletionRecommender().fit(frame, [user, 7, 8], np.array([10, 20, 30, 40, 50]), fit, start, end)
        model.prepare_target(target, target_start, target_end)
        score80, score50 = model.scores([user, 7, 8], 20)
        np.testing.assert_allclose(score80[0], [3 / 22 + 2 / 21, 2 / 21, 0, 0, 0])
        np.testing.assert_allclose(score50[0], [6 / 22 + 2 / 21, 2 / 21, 0, 0, 0])
        np.testing.assert_allclose(score80[1:], np.zeros((2, 5)))
        self.assertAlmostEqual(score50[1, 0], 3 / 22)
        self.assertEqual(model.popularity_freq.to_dict(), {10: 2, 20: 1})
        self.assertEqual(model.diagnostics["completed_80_user_airings"], 3)
        self.assertEqual(model.diagnostics["completed_50_user_airings"], 5)

    def test_v4_grid_and_single_tk_objective(self):
        grid = list(completion_weight_grid())
        self.assertEqual(len(grid), 1024)
        self.assertEqual(len(list(completion_weight_grid(40))), 256)
        self.assertEqual({row["k"] for row in grid}, {20, 40, 80, 160})
        self.assertEqual({row["w50"] for row in grid}, {0, .15, .3, .5})
        self.assertGreater(competition_truth_score({"T1": 0, "T2": 0, "T3": 0, "T4": 0, "TK": .6}),
                           competition_truth_score({"T1": 1, "T2": 1, "T3": 1, "T4": 1, "TK": .5}))
        parts = {"score80": np.array([[0., .5, 1.]]), "score50": np.array([[1., 0., .5]]),
                 "slot": np.array([[0., 1., 0.]]), "ppr": np.array([[0., 0., 1.]]), "pop": np.array([1., .5, 0.])}
        scores = fuse_completion_scores(parts, {"k": 20, "w50": .5, "w_slot": .2, "w_ppr": .1, "w_pop": .2})
        np.testing.assert_allclose(scores, [[.7, .8, 1.35]])

    def test_v4_freezing_preserves_quote_no_300_cutoff_and_tk_objective(self):
        CACHE_DIR.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=CACHE_DIR, prefix="completion_v4_test_") as directory:
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
            v2 = {"seed": 42, "threshold_seconds": 300, "selection_rule": V2_SELECTION_RULE, "weights": v2_weights,
                  "component_configurations": component, "validation_results_file": v2_csv.name,
                  "validation_results_sha256": hashlib.sha256(v2_csv.read_bytes()).hexdigest(), "strict_mean_map_at_5": .4,
                  "validation_scores": {"T1": .1, "T2": .4, "T3": .4, "T4": .4}, "legacy_config_sha256": legacy_digest}
            v2_path = root / "hybrid_v2_selected.json"
            v2_path.write_text(json.dumps(v2))
            v3_weights = list(habit_weight_grid())[0]
            v3_csv = root / "validation_methods_v3.csv"
            pd.DataFrame([{"method": "hybrid_v3", "selected": True, "hyperparameters": json.dumps(v3_weights),
                           "selection_mean_T2_T3": .5}]).to_csv(v3_csv, index=False)
            v3 = {"seed": 42, "threshold_seconds": 300, "selection_rule": V3_SELECTION_RULE, "weights": v3_weights,
                  "component_configurations": component, "validation_results_file": v3_csv.name,
                  "validation_results_sha256": hashlib.sha256(v3_csv.read_bytes()).hexdigest(), "selection_mean_T2_T3": .5,
                  "validation_scores": {"T1": .1, "T2": .3, "T3": .7, "T4": .1}, "legacy_config_sha256": legacy_digest,
                  "v2_selection_manifest_sha256": hashlib.sha256(v2_path.read_bytes()).hexdigest(), "exposure_smoothing_days": 3,
                  "target_schedule_fields": ["channel_id", "start_time", "duration", "tv_show_id"]}
            v3_path = root / "hybrid_v3_selected.json"
            v3_path.write_text(json.dumps(v3))
            configuration = list(completion_weight_grid())[0]
            v4_csv = root / "validation_methods_v4.csv"
            pd.DataFrame([{"method": "hybrid_v4", "selected": True, "hyperparameters": json.dumps(configuration),
                           "selection_map_TK": .6}]).to_csv(v4_csv, index=False)
            manifest = {"seed": 42, "history_threshold_seconds": 300, "selection_rule": V4_SELECTION_RULE,
                        "configuration": configuration, "score80_coefficient": 1, "screen_time_percentages": [80, 50],
                        "completion_requires_300_seconds": False, "competition_ground_truth_quote": COMPETITION_GROUND_TRUTH_QUOTE,
                        "target_schedule_fields": ["channel_id", "start_time", "duration", "tv_show_id"],
                        "best_standalone_score80_k": 20, "component_configurations": {"PPR": component["PPR"]},
                        "validation_results_file": v4_csv.name, "validation_results_sha256": hashlib.sha256(v4_csv.read_bytes()).hexdigest(),
                        "selection_map_TK": .6, "validation_scores": {"T1": 1, "T2": 1, "T3": 1, "T4": 1, "TK": .6},
                        "legacy_config_sha256": legacy_digest, "v3_selection_manifest_sha256": hashlib.sha256(v3_path.read_bytes()).hexdigest()}
            path = root / "hybrid_v4_selected.json"
            path.write_text(json.dumps(manifest))
            loaded, _ = load_frozen_v4_configuration(path)
            self.assertEqual(loaded["configuration"], configuration)
            manifest["completion_requires_300_seconds"] = True
            path.write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                load_frozen_v4_configuration(path)


if __name__ == "__main__":
    unittest.main()

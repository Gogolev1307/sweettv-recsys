import unittest

import numpy as np
import pandas as pd

from src.rerank import (
    FEATURE_NAMES, CandidateRows, RerankFeatures, check_temporal_split,
    probability_rankings, user_partition_robustness,
)


def schedule(records):
    frame = pd.DataFrame(records, columns=["tv_show_id", "channel_id", "start_time", "duration"])
    frame["start_time"] = pd.to_datetime(frame.start_time)
    frame["end_time"] = frame.start_time + pd.to_timedelta(frame.duration, unit="s")
    return frame


class RerankTests(unittest.TestCase):
    def test_threshold_features_channels_recency_and_out_of_pool_user_total(self):
        user = 2**64 - 2
        fitting = schedule([
            (10, 7, "2020-03-09 00:00", 1000), (10, 7, "2020-03-10 00:00", 1000),
            (10, 8, "2020-03-11 00:00", 1000), (20, 7, "2020-03-09 01:00", 100),
            (30, 7, "2020-03-09 02:00", 1000), (99, 7, "2020-03-09 03:00", 1000),
        ])
        records = [
            (user, 0, 10, 7, 800, 1000, "2020-03-09"), (user, 1, 10, 7, 500, 1000, "2020-03-10"),
            (user, 2, 10, 8, 900, 1000, "2020-03-11"), (7, 0, 10, 7, 799, 1000, "2020-03-09"),
            (user, 3, 20, 7, 80, 100, "2020-03-09 01:00"), (user, 4, 30, 7, 499, 1000, "2020-03-09 02:00"),
            (user, 5, 99, 7, 800, 1000, "2020-03-09 03:00"),
        ]
        interactions = pd.DataFrame(records, columns=["user_id", "airing_id", "tv_show_id", "channel_id", "overlap_seconds", "programme_duration", "airing_start"])
        interactions["user_id"] = pd.Series([record[0] for record in records], dtype="uint64")
        interactions["airing_start"] = pd.to_datetime(interactions.airing_start, format="mixed")
        target_records = []
        for channel, count in ((7, 3), (8, 2), (9, 10)):
            for hour in range(count):
                target_records.append((10, channel, pd.Timestamp("2020-03-16") + pd.Timedelta(hours=hour), 1000))
        target_records.extend([(20, 7, "2020-03-17 00:00", 100), (20, 7, "2020-03-17 01:00", 100),
                               (30, 7, "2020-03-17 02:00", 1000), (40, 7, "2020-03-17 03:00", 1000)])
        target = schedule(target_records)
        builder = RerankFeatures(batch_size=1).fit(
            interactions, [user, 7, 8], np.array([10, 20, 30, 40, 50]), fitting,
            pd.Timestamp("2020-03-09"), pd.Timestamp("2020-03-12"), target,
            pd.Timestamp("2020-03-16"), pd.Timestamp("2020-03-18"),
        )
        rows = builder.generate([user, 8, 7])
        self.assertEqual(int(rows.users[0]), user)
        self.assertEqual(len(FEATURE_NAMES), 18)
        position = int(np.flatnonzero(rows.items[0] == 10)[0])
        features = dict(zip(FEATURE_NAMES, rows.features[position]))
        self.assertEqual(features["n_80"], 2)
        self.assertAlmostEqual(features["s_80"], 3 / 82 + 2 / 81)
        self.assertEqual(features["nt_80"], 5)  # New target channel 9 does not count as a personal channel.
        self.assertEqual(features["days_since_80"], 1)
        self.assertEqual(features["n_50"], 3)
        self.assertAlmostEqual(features["s_50"], 6 / 82 + 2 / 81)
        self.assertEqual(features["n_300"], 3)
        self.assertEqual(features["target_airings"], 15)
        self.assertEqual(features["popularity_freq"], 2)
        self.assertEqual(features["user_n80_total"], 4)  # Includes programme 99 outside this target pool.
        self.assertEqual(features["user_n80_share"], .5)
        self.assertEqual(features["base_rank"], 0)
        self.assertAlmostEqual(features["base_score"], 1.6)
        cold = rows.features[5:10]
        np.testing.assert_array_equal(cold[:, (3, 7, 11)], np.full((5, 3), 999))
        np.testing.assert_array_equal(cold[:, 14:16], np.zeros((5, 2)))

    def test_top50_boundary_ties_and_empty_completion_features(self):
        origin = pd.Timestamp("2020-03-09")
        fitting = schedule([(1, 7, origin, 1000)])
        observed = pd.DataFrame({"user_id": pd.Series([1], dtype="uint64"), "airing_id": [0],
                                 "tv_show_id": [1], "channel_id": [7], "overlap_seconds": [300],
                                 "programme_duration": [1000], "airing_start": [origin]})
        builder = RerankFeatures().fit(observed, [1], np.arange(1, 61), fitting, origin, pd.Timestamp("2020-03-10"),
                                      fitting, pd.Timestamp("2020-03-10"), pd.Timestamp("2020-03-11"))
        rows = builder.generate([1])
        np.testing.assert_array_equal(rows.items[0], np.arange(1, 51))
        np.testing.assert_array_equal(rows.features[:, 17], np.arange(50))
        np.testing.assert_array_equal(rows.features[:, (3, 7)], np.full((50, 2), 999))

    def test_probability_ties_use_id_not_base_rank_and_labels_align(self):
        user = 2**64 - 2
        rows = CandidateRows(np.array([user], dtype=np.uint64), np.array([[50, 10, 40, 20, 30]]), np.zeros((5, 18)))
        self.assertEqual(probability_rankings(rows, np.full(5, .2)), {user: [10, 20, 30, 40, 50]})
        np.testing.assert_array_equal(rows.labels({user: {10, 40}}), [0, 1, 1, 0, 0])

    def test_training_windows_cannot_use_b_labels_in_validation(self):
        check_temporal_split("2020-03-09", "2020-04-06", "2020-04-06", "2020-05-18", "2020-05-18")
        with self.assertRaises(ValueError):
            check_temporal_split("2020-03-09", "2020-05-18", "2020-05-18", "2020-07-27", "2020-05-18")
        check_temporal_split("2020-03-09", "2020-06-15", "2020-06-15", "2020-07-27", "2020-07-27")

    def test_robustness_is_seeded_and_uses_paired_user_halves(self):
        baseline, better = np.full(10, .5), np.full(10, .55)
        first = user_partition_robustness(baseline, better)
        second = user_partition_robustness(baseline, better)
        self.assertEqual(first, second)
        self.assertEqual(first["half_sample_wins"], 200)
        self.assertEqual(first["complementary_wins"], 200)
        self.assertEqual(first["both_halves_wins"], 200)
        self.assertAlmostEqual(first["ratio_min"], 1.1)
        self.assertAlmostEqual(first["ratio_median"], 1.1)
        self.assertAlmostEqual(first["ratio_max"], 1.1)


if __name__ == "__main__":
    unittest.main()

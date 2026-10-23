import unittest

import pandas as pd

from src.join import join_views_to_airings, watched_interactions


class IntervalJoinTests(unittest.TestCase):
    def test_spanning_aggregation_unlabelled_uint64_and_window_clipping(self):
        user = 2**64 - 2
        origin = pd.Timestamp("2020-03-09")
        views = pd.DataFrame({
            "user_id": pd.Series([user, user], dtype="uint64"),
            "vsetv_id": [7, 7],
            "start_time": [origin + pd.Timedelta(minutes=5), origin],
            "stop_time": [origin + pd.Timedelta(minutes=35), origin + pd.Timedelta(minutes=5)],
        })
        schedule = pd.DataFrame({
            "airing_id": [0, 1, 2, 3, 4], "channel_id": [7, 7, 7, 7, 8],
            "tv_show_id": [101, 202, 0, 303, 404], "duration": [600] * 5,
            "start_time": [origin + pd.Timedelta(minutes=m) for m in [0, 10, 20, 30, 0]],
        })
        schedule["end_time"] = schedule.start_time + pd.to_timedelta(schedule.duration, unit="s")
        result = join_views_to_airings(views, schedule, origin, origin + pd.Timedelta(hours=1), chunk_size=1)
        overlaps = result.interactions.set_index("tv_show_id").overlap_seconds.to_dict()
        self.assertEqual(overlaps, {101: 600, 202: 600, 303: 300})
        self.assertEqual(result.interactions.user_id.tolist(), [user] * 3)
        self.assertEqual(result.diagnostics["unlabelled_seconds"], 600)
        self.assertEqual(result.diagnostics["covered_seconds"], 2100)
        self.assertEqual(set(watched_interactions(result.interactions, 600).tv_show_id), {101, 202})
        clipped = join_views_to_airings(views, schedule, origin, origin + pd.Timedelta(minutes=10))
        self.assertEqual(clipped.interactions.tv_show_id.tolist(), [101])
        self.assertEqual(clipped.interactions.overlap_seconds.tolist(), [600])

    def test_nested_airing_is_not_missed_or_double_counted_in_coverage(self):
        origin = pd.Timestamp("2020-03-09")
        views = pd.DataFrame({
            "user_id": pd.Series([1], dtype="uint64"), "vsetv_id": [7],
            "start_time": [origin + pd.Timedelta(minutes=25)],
            "stop_time": [origin + pd.Timedelta(minutes=35)],
        })
        schedule = pd.DataFrame({
            "airing_id": [0, 1, 2], "channel_id": [7] * 3,
            "tv_show_id": [1, 2, 0], "duration": [3600, 600, 600],
            "start_time": [origin + pd.Timedelta(minutes=m) for m in [0, 10, 30]],
        })
        schedule["end_time"] = schedule.start_time + pd.to_timedelta(schedule.duration, unit="s")
        result = join_views_to_airings(views, schedule, origin, origin + pd.Timedelta(hours=1))
        self.assertEqual(result.interactions.tv_show_id.tolist(), [1])
        self.assertEqual(result.interactions.overlap_seconds.tolist(), [600])
        self.assertEqual(result.diagnostics["covered_seconds"], 600)
        self.assertEqual(result.diagnostics["additive_overlap_seconds"], 900)
        self.assertEqual(result.diagnostics["unlabelled_seconds"], 300)


if __name__ == "__main__":
    unittest.main()

"""Cached interval inputs shared by strict-truth validation and final refits."""

from src.data import (
    CACHE_DIR, DATA_DIR, TEST_END, TEST_FILE, TEST_START, TRAIN_FILES, VIEW_FILE, cached_result,
    load_test_airings, load_training_schedule, load_views,
)
from src.join import join_views_to_airings


def load_calendar_join(name, start, end, cache_dir=CACHE_DIR):
    sources = [DATA_DIR / filename for filename in (VIEW_FILE, *TRAIN_FILES)]

    def build():
        schedule, _ = load_training_schedule(DATA_DIR, cache_dir)
        return join_views_to_airings(
            load_views(DATA_DIR, cache_dir), schedule, start, end, collect_days=True,
        )

    return cached_result(
        cache_dir, f"calendar_join_{name}", sources,
        {"window_start": str(start), "window_end": str(end), "calendar_version": 1}, build,
    )


def load_overlap_join(start, end, cache_dir=CACHE_DIR):
    """Compact non-daily joins for additional internal temporal splits."""
    sources = [DATA_DIR / filename for filename in (VIEW_FILE, *TRAIN_FILES)]

    def build():
        schedule, _ = load_training_schedule(DATA_DIR, cache_dir)
        return join_views_to_airings(load_views(DATA_DIR, cache_dir), schedule, start, end)

    return cached_result(cache_dir, "internal_overlap_join", sources,
                         {"start": str(start), "end": str(end), "join_version": 1}, build)


def load_slot_profile(name, requested_users, start, end, cache_dir=CACHE_DIR):
    from src.slot import SlotRecommender

    return cached_result(
        cache_dir, f"slot_profile_{name}", [DATA_DIR / VIEW_FILE, DATA_DIR / "submission.csv"],
        {"start": str(start), "end": str(end), "users": list(map(int, requested_users)), "slot_version": 1},
        lambda: SlotRecommender().fit(load_views(DATA_DIR, cache_dir), requested_users, start, end),
    )


def load_final_target_airings(cache_dir=CACHE_DIR):
    """Numerical test availability, including known training-export boundary carry-in."""
    import pandas as pd

    def build():
        training, _ = load_training_schedule(DATA_DIR, cache_dir)
        test = load_test_airings(DATA_DIR, cache_dir)
        core = ["channel_id", "start_time", "duration", "tv_show_id", "end_time"]
        combined = pd.concat([training[core], test[core]], ignore_index=True)
        combined = combined.drop_duplicates(["channel_id", "start_time"], keep="last")
        return combined.loc[(combined.start_time < TEST_END) & (combined.end_time > TEST_START)].reset_index(drop=True)

    return cached_result(
        cache_dir, "final_target_availability", [DATA_DIR / name for name in (*TRAIN_FILES, TEST_FILE)],
        {"start": str(TEST_START), "end": str(TEST_END), "availability_version": 1}, build,
    )

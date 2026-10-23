"""Typed, read-only access to the competition data and fingerprinted caches."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Callable, TypeVar

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
CACHE_DIR = ROOT / "cache"
VIEW_FILE = "dataset11-30.csv"
TRAIN_FILES = ("export_arh_11-20-final.csv", "export_arh_21-30-final.csv")
TEST_FILE = "export_arh_31-42-final.csv"
VIEW_TIME_FORMAT = "%Y-%m-%d %H:%M:%S"
SCHEDULE_TIME_FORMAT = "%d.%m.%Y %H:%M:%S"
A_START = pd.Timestamp("2020-03-09")
B_START = pd.Timestamp("2020-05-18")
TEST_START = pd.Timestamp("2020-07-27")
TEST_END = pd.Timestamp("2020-10-19")
CACHE_VERSION = 1
T = TypeVar("T")

SCHEDULE_DTYPES = {
    "channel_id": "int32",
    "channel_title": "string",
    "duration": "int64",
    "tv_show_title": "string",
    "tv_show_id": "int64",
    "tv_show_category": "string",
    "tv_show_genre_1": "string",
    "tv_show_genre_2": "string",
    "tv_show_genre_3": "string",
    # Values such as '1996-' are not integers.
    "year_of_production": "string",
    "director": "string",
    "actors": "string",
}


def cached_result(
    cache_dir: Path | None,
    name: str,
    sources: list[Path],
    parameters: dict,
    build: Callable[[], T],
) -> T:
    """Cache locally generated objects; invalidate when inputs/settings change."""
    if cache_dir is None:
        return build()
    fingerprint = {
        "version": CACHE_VERSION,
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "sources": [
            (str(path.resolve()), path.stat().st_size, path.stat().st_mtime_ns)
            for path in sources
        ],
        "parameters": parameters,
    }
    digest = hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()[:20]
    cache_dir.mkdir(parents=True, exist_ok=True)
    destination = cache_dir / f"{name}_{digest}.pkl"
    if destination.exists():
        return pd.read_pickle(destination)
    result = build()
    # An interrupted run must not leave a seemingly valid partial cache.
    with tempfile.NamedTemporaryFile(dir=cache_dir, suffix=".tmp", delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        pd.to_pickle(result, temporary_path)
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)
    return result


def load_views(data_dir: Path = DATA_DIR, cache_dir: Path | None = CACHE_DIR) -> pd.DataFrame:
    source = data_dir / VIEW_FILE

    def build() -> pd.DataFrame:
        frame = pd.read_csv(
            source,
            dtype={"user_id": "uint64", "vsetv_id": "int32", "duraton": "int64"},
            parse_dates=["start_time", "stop_time"],
            date_format=VIEW_TIME_FORMAT,
        )
        for column in ("start_time", "stop_time"):
            frame[column] = pd.to_datetime(frame[column], format=VIEW_TIME_FORMAT, errors="raise")
        if frame[["user_id", "vsetv_id", "start_time", "stop_time", "duraton"]].isna().any().any():
            raise ValueError("Missing view-log fields")
        # 'duraton' is a viewing interval, never a programme's running time.
        # The actual endpoints are authoritative, including at DST boundaries.
        frame["view_seconds"] = (frame.stop_time - frame.start_time).dt.total_seconds().astype("int64")
        if (frame.view_seconds <= 0).any():
            raise ValueError("Nonpositive view interval")
        return frame

    return cached_result(cache_dir, "views", [source], {}, build)


def load_schedule(path: Path, *, include_metadata: bool = False) -> pd.DataFrame:
    """Load either schedule format, optionally retaining the unused text metadata."""
    columns = None if include_metadata else ["channel_id", "start_time", "duration", "tv_show_id"]
    dtypes = SCHEDULE_DTYPES if include_metadata else {
        column: SCHEDULE_DTYPES[column] for column in columns if column != "start_time"
    }
    frame = pd.read_csv(
        path,
        usecols=columns,
        dtype=dtypes,
        parse_dates=["start_time"],
        date_format=SCHEDULE_TIME_FORMAT,
    )
    frame["start_time"] = pd.to_datetime(frame.start_time, format=SCHEDULE_TIME_FORMAT, errors="raise")
    if frame[["channel_id", "start_time", "duration", "tv_show_id"]].isna().any().any():
        raise ValueError(f"Missing schedule fields: {path.name}")
    if (frame.tv_show_id < 0).any():
        raise ValueError(f"Negative programme id: {path.name}")
    frame["end_time"] = frame.start_time + pd.to_timedelta(frame.duration, unit="s")
    return frame


def schedule_summary(frame: pd.DataFrame) -> dict:
    return {
        "rows": len(frame),
        "start_min": frame.start_time.min(),
        "start_max": frame.start_time.max(),
        "channels": frame.channel_id.nunique(),
        "unlabelled_rows": int((frame.tv_show_id == 0).sum()),
        "nonpositive_duration_rows": int((frame.duration <= 0).sum()),
    }


def load_training_schedule(
    data_dir: Path = DATA_DIR, cache_dir: Path | None = CACHE_DIR
) -> tuple[pd.DataFrame, dict]:
    sources = [data_dir / filename for filename in TRAIN_FILES]

    def build() -> tuple[pd.DataFrame, dict]:
        frames = [load_schedule(path) for path in sources]
        summaries = {path.name: schedule_summary(frame) for path, frame in zip(sources, frames)}
        keys = ["channel_id", "start_time"]
        within_file_duplicates = sum(int(frame.duplicated(keys).sum()) for frame in frames)
        combined = pd.concat(frames, ignore_index=True)
        removed = combined.duplicated(keys, keep="last")
        overlap_start = max(frame.start_time.min().normalize() for frame in frames)
        overlap_end = min(frame.start_time.max().normalize() for frame in frames)
        checks = {
            "files": summaries,
            "calendar_overlap": (overlap_start, overlap_end),
            "duplicates_removed": int(removed.sum()),
            "cross_file_duplicate_keys": int(removed.sum()) - within_file_duplicates,
            "duplicate_dates": combined.loc[removed, "start_time"].dt.strftime("%Y-%m-%d").value_counts().to_dict(),
        }
        # Later rows win; in these files this retains the positive-duration row
        # rather than the zero-duration duplicate. Deduplicate before joining.
        schedule = combined.loc[~removed].copy()
        checks["nonpositive_after_dedup"] = int((schedule.duration <= 0).sum())
        schedule = schedule.loc[schedule.duration > 0].sort_values(keys).reset_index(drop=True)
        schedule.insert(0, "airing_id", np.arange(len(schedule), dtype=np.int64))
        if schedule.duplicated(keys).any():
            raise AssertionError("Schedule keys are not unique")
        return schedule, checks

    return cached_result(cache_dir, "training_schedule", sources, {"deduplicate": "keep_last"}, build)


def load_test_schedule_summary(data_dir: Path = DATA_DIR, cache_dir: Path | None = CACHE_DIR) -> dict:
    """Read test airings for input checks only; never pass them into validation."""
    source = data_dir / TEST_FILE
    return cached_result(
        cache_dir, "test_schedule_summary", [source], {}, lambda: schedule_summary(load_schedule(source))
    )


def load_submission(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    frame = pd.read_csv(data_dir / "submission.csv", dtype={"user_id": "uint64", "tv_show_id": "string"})
    if list(frame.columns) != ["user_id", "tv_show_id"]:
        raise ValueError("Unexpected submission header")
    if frame.user_id.isna().any() or frame.user_id.duplicated().any() or len(frame) != 1260:
        raise ValueError("Expected exactly 1,260 distinct, non-null test users")
    return frame


def airings_in_window(schedule: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Select schedule rows whose airing starts are inside the half-open window."""
    return schedule.loc[(schedule.start_time >= start) & (schedule.start_time < end)]


def _labelled_training_metadata(sources: list[Path]) -> pd.DataFrame:
    labelled = []
    for source in sources:
        for chunk in pd.read_csv(
            source, dtype=SCHEDULE_DTYPES, parse_dates=["start_time"],
            date_format=SCHEDULE_TIME_FORMAT, chunksize=100_000,
        ):
            chunk = chunk.loc[chunk.tv_show_id != 0].copy()
            chunk["start_time"] = pd.to_datetime(chunk.start_time, format=SCHEDULE_TIME_FORMAT, errors="raise")
            chunk["end_time"] = chunk.start_time + pd.to_timedelta(chunk.duration, unit="s")
            labelled.append(chunk)
    frame = pd.concat(labelled, ignore_index=True).drop_duplicates(["channel_id", "start_time"], keep="last")
    return frame.loc[frame.duration > 0].sort_values(["start_time", "channel_id"])


def load_programme_metadata(
    data_dir: Path = DATA_DIR, cache_dir: Path | None = CACHE_DIR
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A metadata snapshot, plus A-first/B-fill side information for candidates.

    Read in chunks and discard unlabelled airtime early. The latest A airing
    supplies each A snapshot; otherwise the earliest B airing supplies content.
    B-only metadata is available for transformation/display, not IDF fitting.
    """
    sources = [data_dir / name for name in TRAIN_FILES]

    def build():
        frame = _labelled_training_metadata(sources)
        a = frame.loc[(frame.start_time < B_START) & (frame.end_time > A_START)].drop_duplicates("tv_show_id", keep="last")
        b = airings_in_window(frame, B_START, TEST_START).drop_duplicates("tv_show_id", keep="first")
        available = pd.concat([a, b.loc[~b.tv_show_id.isin(a.tv_show_id)]], ignore_index=True)
        return a.reset_index(drop=True), available.reset_index(drop=True)

    return cached_result(cache_dir, "programme_metadata", sources, {"snapshot_version": 1}, build)


def load_all_training_metadata(data_dir: Path = DATA_DIR, cache_dir: Path | None = CACHE_DIR) -> pd.DataFrame:
    """Latest known content in weeks 11-30; never read test-week metadata."""
    sources = [data_dir / name for name in TRAIN_FILES]

    def build():
        frame = _labelled_training_metadata(sources)
        frame = frame.loc[(frame.start_time < TEST_START) & (frame.end_time > A_START)]
        return frame.drop_duplicates("tv_show_id", keep="last").reset_index(drop=True)

    return cached_result(cache_dir, "all_training_metadata", sources, {"snapshot_version": 1}, build)


def load_test_candidates(data_dir: Path = DATA_DIR, cache_dir: Path | None = CACHE_DIR) -> np.ndarray:
    """The final test schedule supplies item availability only: read just ids."""
    source = data_dir / TEST_FILE

    def build():
        items = pd.read_csv(source, usecols=["tv_show_id"], dtype={"tv_show_id": "int64"}).tv_show_id
        if items.isna().any() or (items < 0).any():
            raise ValueError("Invalid test-schedule programme ids")
        return np.sort(items.loc[items != 0].unique())

    return cached_result(cache_dir, "test_candidate_ids", [source], {"columns": ["tv_show_id"]}, build)


def load_fitting_metadata(start, end, data_dir: Path = DATA_DIR, cache_dir: Path | None = CACHE_DIR):
    """Content available in the fitting window; target metadata never supplies vectors."""
    if end > TEST_START:
        raise ValueError("Metadata fitting cannot extend into test weeks")
    filenames = TRAIN_FILES[:1] if end <= B_START else TRAIN_FILES
    sources = [data_dir / name for name in filenames]

    def build():
        frame = _labelled_training_metadata(sources)
        frame = frame.loc[(frame.start_time < end) & (frame.end_time > start)]
        return frame.drop_duplicates("tv_show_id", keep="last").reset_index(drop=True)

    return cached_result(cache_dir, "fitting_metadata", sources, {"start": str(start), "end": str(end)}, build)


def load_test_airings(data_dir: Path = DATA_DIR, cache_dir: Path | None = CACHE_DIR):
    """Read only programme/channel/start/duration for target availability features."""
    source = data_dir / TEST_FILE

    def build():
        frame = load_schedule(source, include_metadata=False)
        frame = frame.drop_duplicates(["channel_id", "start_time"], keep="last")
        return frame.loc[frame.duration > 0].reset_index(drop=True)

    return cached_result(cache_dir, "test_airings_core", [source], {"columns": "availability_only"}, build)

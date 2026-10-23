"""All-overlaps interval join, with exact user ids and bounded working memory."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class JoinResult:
    interactions: pd.DataFrame
    diagnostics: dict[str, int]
    daily_interactions: pd.DataFrame | None = None


def _seconds(series: pd.Series) -> np.ndarray:
    return series.to_numpy(dtype="datetime64[s]").astype(np.int64)


def _previous_maximum(ends: np.ndarray) -> np.ndarray:
    result = np.empty(len(ends), dtype=np.int64)
    result[0] = np.iinfo(np.int64).min
    result[1:] = np.maximum.accumulate(ends)[:-1]
    return result


def clock_alignment_counts(
    views: pd.DataFrame, schedule: pd.DataFrame, offsets: tuple[int, ...] = (-3, -2, -1, 0, 1, 2, 3)
) -> dict[int, int]:
    """Count exact start coincidences for view_start - schedule_start in hours."""
    counts = dict.fromkeys(offsets, 0)
    schedules = schedule.groupby("channel_id", sort=False).indices
    view_starts = _seconds(views.start_time)
    schedule_starts = _seconds(schedule.start_time)
    for channel, positions in views.groupby("vsetv_id", sort=False).indices.items():
        if channel not in schedules:
            continue
        starts = np.sort(schedule_starts[schedules[channel]])
        for offset in offsets:
            targets = view_starts[positions] - offset * 3600
            indices = np.searchsorted(starts, targets)
            matches = (indices < len(starts)) & (starts[np.minimum(indices, len(starts) - 1)] == targets)
            counts[offset] += int(matches.sum())
    return counts


def join_views_to_airings(
    views: pd.DataFrame,
    schedule: pd.DataFrame,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    *,
    chunk_size: int = 100_000,
    collect_days: bool = False,
) -> JoinResult:
    """Sum overlaps per (user, airing), retaining every intersecting airing.

    Both view and airing intervals are half-open. Viewing is clipped to the
    requested window BEFORE aggregation, so no seconds from B enter A. Distinct
    view rows contribute additively; simultaneous sessions can make the summed
    fraction exceed one. Id 0 contributes only to coverage diagnostics.
    """
    if window_end <= window_start or chunk_size <= 0:
        raise ValueError("Invalid window or chunk size")
    lower, upper = window_start.value // 10**9, window_end.value // 10**9
    view_starts, view_ends = _seconds(views.start_time), _seconds(views.stop_time)
    in_window = (view_starts < upper) & (view_ends > lower) & (view_ends > view_starts)
    users = views.user_id.to_numpy(dtype=np.uint64)
    schedule_starts, schedule_ends = _seconds(schedule.start_time), _seconds(schedule.end_time)
    airing_ids = schedule.airing_id.to_numpy(dtype=np.int64)
    item_ids = schedule.tv_show_id.to_numpy(dtype=np.int64)
    schedule_groups = schedule.groupby("channel_id", sort=False).indices
    diagnostics = dict.fromkeys(
        [
            "views", "view_seconds", "matched_views", "multi_airing_views", "airing_pairs",
            "labelled_pairs", "unlabelled_pairs", "covered_seconds", "labelled_seconds",
            "unlabelled_seconds", "additive_overlap_seconds", "additive_unlabelled_seconds",
        ],
        0,
    )
    diagnostics["views"] = int(in_window.sum())
    diagnostics["view_seconds"] = int(
        (np.minimum(view_ends[in_window], upper) - np.maximum(view_starts[in_window], lower)).sum()
    )
    pieces, daily_pieces = [], []
    negative_infinity = np.iinfo(np.int64).min

    for channel, positions in views.groupby("vsetv_id", sort=True).indices.items():
        positions = positions[in_window[positions]]
        if len(positions) == 0 or channel not in schedule_groups:
            continue
        schedule_positions = schedule_groups[channel]
        active = (schedule_starts[schedule_positions] < upper) & (schedule_ends[schedule_positions] > lower)
        schedule_positions = schedule_positions[active]
        if len(schedule_positions) == 0:
            continue
        order = np.argsort(schedule_starts[schedule_positions], kind="stable")
        schedule_positions = schedule_positions[order]
        starts, ends = schedule_starts[schedule_positions], schedule_ends[schedule_positions]
        items, airings = item_ids[schedule_positions], airing_ids[schedule_positions]
        prefix_ends = np.maximum.accumulate(ends)
        previous_all = _previous_maximum(ends)
        previous_labelled = _previous_maximum(np.where(items != 0, ends, negative_infinity))
        previous_unlabelled = _previous_maximum(np.where(items == 0, ends, negative_infinity))

        for offset in range(0, len(positions), chunk_size):
            batch = positions[offset:offset + chunk_size]
            vstart, vend = np.maximum(view_starts[batch], lower), np.minimum(view_ends[batch], upper)
            # Prefix maxima are essential: an earlier long airing can overlap a
            # view even when a later short airing has already ended.
            first = np.searchsorted(prefix_ends, vstart, side="right")
            last = np.searchsorted(starts, vend, side="left")
            lengths = np.maximum(last - first, 0)
            pair_count = int(lengths.sum())
            if pair_count == 0:
                continue
            owners = np.repeat(np.arange(len(batch)), lengths)
            local_airings = (
                np.repeat(first, lengths) + np.arange(pair_count)
                - np.repeat(np.cumsum(lengths) - lengths, lengths)
            )
            left = np.maximum(vstart[owners], starts[local_airings])
            right = np.minimum(vend[owners], ends[local_airings])
            valid = right > left
            owners, local_airings, left, right = (
                array[valid] for array in (owners, local_airings, left, right)
            )
            overlaps = right - left
            labelled = items[local_airings] != 0
            per_view = np.bincount(owners, minlength=len(batch))
            diagnostics["matched_views"] += int((per_view > 0).sum())
            diagnostics["multi_airing_views"] += int((per_view > 1).sum())
            diagnostics["airing_pairs"] += len(owners)
            diagnostics["labelled_pairs"] += int(labelled.sum())
            diagnostics["unlabelled_pairs"] += int((~labelled).sum())
            diagnostics["additive_overlap_seconds"] += int(overlaps.sum())
            diagnostics["additive_unlabelled_seconds"] += int(overlaps[~labelled].sum())
            # Count time coverage as a union, not twice when schedule rows overlap.
            for key, previous, selected in (
                ("covered_seconds", previous_all, np.ones(len(owners), dtype=bool)),
                ("labelled_seconds", previous_labelled, labelled),
                ("unlabelled_seconds", previous_unlabelled, ~labelled),
            ):
                contributions = np.maximum(right - np.maximum(left, previous[local_airings]), 0)
                diagnostics[key] += int(contributions[selected].sum())
            if labelled.any():
                piece = pd.DataFrame({
                    "user_id": users[batch[owners[labelled]]],
                    "airing_id": airings[local_airings[labelled]],
                    "overlap_seconds": overlaps[labelled],
                })
                pieces.append(piece.groupby(["user_id", "airing_id"], as_index=False, sort=False).overlap_seconds.sum())
                if collect_days:
                    # Attribute positive viewing seconds to actual calendar dates,
                    # not the airing's start date. A midnight stop is exclusive.
                    labelled_left, labelled_right = left[labelled], right[labelled]
                    first_day = labelled_left // 86400
                    day_lengths = (labelled_right - 1) // 86400 - first_day + 1
                    day_owners = np.repeat(np.arange(len(first_day)), day_lengths)
                    days = (
                        np.repeat(first_day, day_lengths) + np.arange(int(day_lengths.sum()))
                        - np.repeat(np.cumsum(day_lengths) - day_lengths, day_lengths)
                    )
                    day_seconds = (
                        np.minimum(labelled_right[day_owners], (days + 1) * 86400)
                        - np.maximum(labelled_left[day_owners], days * 86400)
                    )
                    daily_piece = pd.DataFrame({
                        "user_id": users[batch[owners[labelled]]][day_owners],
                        "airing_id": airings[local_airings[labelled]][day_owners],
                        "watch_day": days.astype(np.int32),
                        "overlap_seconds": day_seconds,
                    })
                    daily_pieces.append(daily_piece.groupby(
                        ["user_id", "airing_id", "watch_day"], as_index=False, sort=False
                    ).overlap_seconds.sum())

    if pieces:
        interactions = pd.concat(pieces, ignore_index=True)
        interactions = interactions.groupby(["user_id", "airing_id"], as_index=False, sort=False).overlap_seconds.sum()
    else:
        interactions = pd.DataFrame({
            "user_id": pd.Series(dtype="uint64"),
            "airing_id": pd.Series(dtype="int64"),
            "overlap_seconds": pd.Series(dtype="int64"),
        })
    metadata = schedule[["airing_id", "channel_id", "tv_show_id", "start_time", "duration"]].rename(
        columns={"start_time": "airing_start", "duration": "programme_duration"}
    )
    interactions = interactions.merge(metadata, on="airing_id", how="left", validate="many_to_one", sort=False)
    interactions["overlap_fraction"] = interactions.overlap_seconds / interactions.programme_duration
    if interactions.user_id.dtype != np.dtype("uint64") or (interactions.tv_show_id == 0).any():
        raise AssertionError("Invalid item interactions")
    if diagnostics["covered_seconds"] > diagnostics["view_seconds"]:
        raise AssertionError("Union coverage exceeds viewing time")
    daily = None
    if collect_days:
        if daily_pieces:
            daily = pd.concat(daily_pieces, ignore_index=True).groupby(
                ["user_id", "airing_id", "watch_day"], as_index=False, sort=False
            ).overlap_seconds.sum()
        else:
            daily = pd.DataFrame({
                "user_id": pd.Series(dtype="uint64"), "airing_id": pd.Series(dtype="int64"),
                "watch_day": pd.Series(dtype="int32"), "overlap_seconds": pd.Series(dtype="int64"),
            })
        daily = daily.merge(schedule[["airing_id", "tv_show_id"]], on="airing_id", how="left", validate="many_to_one")
        if int(daily.overlap_seconds.sum()) != int(interactions.overlap_seconds.sum()):
            raise AssertionError("Calendar-day splitting lost or duplicated overlap seconds")
        diagnostics["daily_user_airing_rows"] = len(daily)
    return JoinResult(interactions, diagnostics, daily)


def watched_interactions(interactions: pd.DataFrame, threshold_seconds: int = 300) -> pd.DataFrame:
    if threshold_seconds <= 0:
        raise ValueError("The watch threshold must be positive")
    return interactions.loc[interactions.overlap_seconds >= threshold_seconds]


def completed_airings(interactions: pd.DataFrame, percentage: int = 80) -> pd.DataFrame:
    """Qualify an aggregated user/airing by its FULL scheduled running time.

    No 300-second cutoff is applied. Integer cross multiplication keeps exactly
    80%/50% boundary cases exact; each aggregated user/airing counts only once.
    """
    if not isinstance(percentage, int) or not 0 < percentage <= 100:
        raise ValueError("Completion percentage must be an integer from 1 to 100")
    if interactions.duplicated(["user_id", "airing_id"]).any():
        raise ValueError("Completion requires one aggregated record per user/airing")
    return interactions.loc[
        interactions.tv_show_id.ne(0) & interactions.programme_duration.gt(0)
        & (interactions.overlap_seconds * 100 >= interactions.programme_duration * percentage)
    ]


def programme_day_counts(result: JoinResult, threshold_seconds: int = 300) -> pd.DataFrame:
    """Distinct viewing dates of programmes with qualifying airings in this window.

    First threshold each complete user/airing total. Every date with positive
    overlap on such an airing counts; a date need not separately reach 300s.
    Multiple eligible airings on the same date still count as one programme day.
    """
    if result.daily_interactions is None:
        raise ValueError("Calendar counts require join_views_to_airings(..., collect_days=True)")
    eligible = watched_interactions(result.interactions, threshold_seconds)[["user_id", "airing_id"]]
    daily = result.daily_interactions.merge(eligible, on=["user_id", "airing_id"], how="inner", validate="many_to_one")
    return daily.groupby(["user_id", "tv_show_id"], as_index=False, sort=False).watch_day.nunique().rename(
        columns={"watch_day": "watched_days"}
    )


def history_users_across_windows(results: list[JoinResult], threshold_seconds: int = 300) -> set[int]:
    """Threshold full-period totals, including airings straddling A/B."""
    combined = pd.concat(
        [result.interactions[["user_id", "airing_id", "overlap_seconds"]] for result in results],
        ignore_index=True,
    )
    totals = combined.groupby(["user_id", "airing_id"], as_index=False, sort=False).overlap_seconds.sum()
    return set(map(int, watched_interactions(totals, threshold_seconds).user_id.unique()))

"""Four B-only relevance definitions, isolated from A-only recommendation fitting."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluate import ValidationProtocol
from src.join import JoinResult, completed_airings, programme_day_counts, watched_interactions

COMPETITION_GROUND_TRUTH_QUOTE = (
    "all entries with tv_show_id = 0 were removed from TV program schedule files; "
    "for each user and each TV show user screen time (time user watched the show divided by total duration of the show) was calculated. "
    "Only TV shows with high screen time (>= 0.8) were taken into account; "
    "for each user high-screen time TV shows frequency was calculated; "
    "for each user strictly 5 most frequent TV shows were selected DESC"
)
TK_DEFINITION = "top 5 programmes by count of labelled airings with screen time >= 0.8; ascending-id count ties"

TRUTH_DEFINITIONS = {
    "T1": "any airing with overlap >= 300 seconds",
    "T2": "total labelled overlap seconds in B >= 3600 (all positive airings)",
    "T3": "qualifying airings (>= 300 seconds) on >= 5 distinct actual viewing dates",
    "T4": "top 10 programmes by total labelled B overlap seconds; ascending-id ties (all positive airings)",
}
ALL_TRUTH_DEFINITIONS = {**TRUTH_DEFINITIONS, "TK": TK_DEFINITION}


def validation_truths(test_users, candidates, joined_b: JoinResult, *, include_tk=False):
    users = tuple(map(int, test_users))
    rows = joined_b.interactions.loc[joined_b.interactions.user_id.isin(users)]
    totals = rows.groupby(["user_id", "tv_show_id"], as_index=False).overlap_seconds.sum()
    days = programme_day_counts(joined_b)
    days = days.loc[days.user_id.isin(users)]
    top_ten = totals.sort_values(
        ["user_id", "overlap_seconds", "tv_show_id"], ascending=[True, False, True]
    ).groupby("user_id", sort=False).head(10)
    truth_rows = {
        "T1": watched_interactions(rows),
        "T2": totals.loc[totals.overlap_seconds >= 3600],
        "T3": days.loc[days.watched_days >= 5],
        "T4": top_ten,
    }
    protocols, accounting = {}, {}
    for name, frame in truth_rows.items():
        truth = {
            int(user): frozenset(map(int, items))
            for user, items in frame.groupby("user_id", sort=False).tv_show_id
        }
        counts = np.array([len(truth.get(user, ())) for user in users])
        evaluated = int((counts > 0).sum())
        if evaluated == 0:
            raise ValueError(f"No evaluated users for {name}")
        protocols[name] = ValidationProtocol(users, frozenset(map(int, candidates)), truth)
        accounting[name] = {
            "evaluated_users": evaluated, "no_relevant_users": len(users) - evaluated,
            "median_relevant_all_users": float(np.median(counts)),
            "median_relevant_evaluated_users": float(np.median(counts[counts > 0])),
        }
        if evaluated + accounting[name]["no_relevant_users"] != len(users):
            raise AssertionError("Truth cohort accounting does not close")
    if include_tk:
        protocols["TK"], accounting["TK"] = competition_truth(users, candidates, joined_b)
    return protocols, accounting


def competition_truth(test_users, candidates, joined_b: JoinResult):
    users = tuple(map(int, test_users))
    rows = joined_b.interactions.loc[joined_b.interactions.user_id.isin(users)]
    high = completed_airings(rows, 80)
    frequency = high.groupby(["user_id", "tv_show_id"], as_index=False).size().rename(columns={"size": "n80"})
    top_five = frequency.sort_values(
        ["user_id", "n80", "tv_show_id"], ascending=[True, False, True]
    ).groupby("user_id", sort=False).head(5)
    truth = {
        int(user): frozenset(map(int, items))
        for user, items in top_five.groupby("user_id", sort=False).tv_show_id
    }
    sizes = np.array([len(truth.get(user, ())) for user in users])
    evaluated = int((sizes > 0).sum())
    if evaluated == 0:
        raise ValueError("No users with competition-rule truth")
    stats = {
        "evaluated_users": evaluated, "no_relevant_users": len(users) - evaluated,
        "median_relevant_all_users": float(np.median(sizes)),
        "median_relevant_evaluated_users": float(np.median(sizes[sizes > 0])),
        "relevance_size_counts": {str(size): int((sizes == size).sum()) for size in range(6)},
        "high_completion_user_airings": len(high),
    }
    if (sizes > 5).any() or sum(stats["relevance_size_counts"].values()) != len(users):
        raise AssertionError("TK must contain min(5, qualifying programmes) items per evaluated user")
    return ValidationProtocol(users, frozenset(map(int, candidates)), truth), stats

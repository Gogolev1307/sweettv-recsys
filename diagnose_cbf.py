"""Reproduce CBF scores and independently audit feature/profile/ranking alignment."""

from __future__ import annotations

import os

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[variable] = "1"

import sys
import time
from contextlib import redirect_stdout

import numpy as np
import pandas as pd

from run_validation import Tee
from src.cbf import ContentRecommender, metadata_tokens
from src.data import (
    A_START, B_START, CACHE_DIR, DATA_DIR, ROOT, TEST_START, TRAIN_FILES, VIEW_FILE,
    airings_in_window, cached_result, load_programme_metadata, load_submission,
    load_training_schedule, load_views,
)
from src.evaluate import ValidationProtocol
from src.join import join_views_to_airings, watched_interactions
from src.matrix import InteractionMatrix, rank_scores
from src.popularity import PopularityRecommender


def run():
    started = time.perf_counter()
    np.random.seed(42)
    sources = [DATA_DIR / name for name in (VIEW_FILE, *TRAIN_FILES)]
    schedule, _ = load_training_schedule()
    joins = []
    for name, lower, upper in (("A", A_START, B_START), ("B", B_START, TEST_START)):
        joins.append(cached_result(
            CACHE_DIR, f"join_{name}", sources,
            {"window_start": str(lower), "window_end": str(upper), "join_version": 1},
            lambda lower=lower, upper=upper: join_views_to_airings(load_views(), schedule, lower, upper),
        ))
    a, b = (watched_interactions(result.interactions, 300) for result in joins)
    users = tuple(map(int, load_submission().user_id))
    protocol = ValidationProtocol.from_interactions(users, airings_in_window(schedule, B_START, TEST_START), b)
    matrix = cached_result(
        CACHE_DIR, "method_matrix", sources + [DATA_DIR / "submission.csv"],
        {"threshold": 300, "matrix_version": 1},
        lambda: InteractionMatrix.build(a, users, protocol.candidates),
    )
    metadata_a, available = load_programme_metadata()
    lookup = available.set_index("tv_show_id").to_dict("index")
    model = ContentRecommender().fit(matrix.seconds, matrix.item_ids, metadata_a, available)
    counts = {int(user): int(count) for user, count in a.groupby("user_id").size().items()}
    user = min(users, key=lambda value: (-counts.get(value, 0), value))
    row = int(np.searchsorted(matrix.user_ids, np.uint64(user)))
    history = matrix.seconds.getrow(row)
    candidate_ids = matrix.item_ids[matrix.candidate_columns]
    scores = model.score(np.array([row]), matrix.candidate_columns)[0]
    order = np.lexsort((candidate_ids, -scores))
    ranks = np.empty(len(order), dtype=int)
    ranks[order] = np.arange(1, len(order) + 1)
    history_order = np.lexsort((matrix.item_ids[history.indices], -history.data))
    previous = pd.read_csv(ROOT / "results" / "validation_methods.csv")
    before = float(previous.loc[previous.method.eq("CBF") & previous.selected, "map_at_5"].iloc[0])
    print("CBF DIAGNOSTIC - unchanged scoring implementation; seed=42; threshold=300s")
    print(f"CBF MAP@5 before (saved): {before:.9f}")
    print(f"Heaviest test user: {user}; A airings={counts[user]:,}; A programmes={history.nnz:,}")
    print(f"Candidates={len(candidate_ids):,}; median cosine={np.median(scores):.9f}; max cosine={scores.max():.9f}")
    print("Five most-watched A programmes: id | seconds | cosine | candidate rank | title")
    for index in history_order[:5]:
        item = int(matrix.item_ids[history.indices[index]])
        position = int(np.searchsorted(candidate_ids, item))
        title = lookup.get(item, {}).get("tv_show_title", "")
        if position < len(candidate_ids) and candidate_ids[position] == item:
            print(f"{item} | {int(history.data[index]):,} | {scores[position]:.9f} | "
                  f"{ranks[position]:,}/{len(candidate_ids):,} | {title}")
        else:
            print(f"{item} | {int(history.data[index]):,} | not a candidate | {title}")

    # Reconstruct each programme independently from its own id-keyed metadata,
    # without relying on metadata DataFrame order or the model's document order.
    mismatches, maximum_error = 0, 0.0
    for feature_row, item in enumerate(matrix.item_ids):
        columns = sorted(
            model.vocabulary[token] for token in metadata_tokens(lookup.get(int(item), {}))
            if token in model.vocabulary
        )
        expected = np.zeros(len(model.vocabulary))
        expected[columns] = model.idf[columns]
        norm = np.linalg.norm(expected)
        if norm:
            expected /= norm
        actual = model.item_vectors.getrow(feature_row).toarray().ravel()
        error = float(np.max(np.abs(expected - actual)))
        maximum_error = max(maximum_error, error)
        mismatches += int(error > 1e-12)
    print(f"Independent id-to-token/IDF reconstruction: {len(matrix.item_ids):,} rows; "
          f"mismatches={mismatches}; max error={maximum_error:.3e}")
    if mismatches:
        raise AssertionError("Item feature row/id mismatch")
    sample_ids = [int(matrix.item_ids[history.indices[index]]) for index in history_order[:5]]
    sample_ids += [int(candidate_ids[index]) for index in order[:3]]
    print("Alignment samples: id -> matrix column / feature row -> candidate position (0-based) -> title")
    for item in sample_ids:
        feature_row = int(np.searchsorted(matrix.item_ids, item))
        position = int(np.searchsorted(candidate_ids, item))
        print(f"{item} -> {feature_row}/{feature_row} -> {position} -> {lookup[item].get('tv_show_title', '')}")

    mask_ok = np.array_equal(candidate_ids, np.array(sorted(protocol.candidates), dtype=np.int64))
    full_scores = model.score(np.array([row]), np.arange(len(matrix.item_ids)))[0]
    slice_error = float(np.max(np.abs(scores - full_scores[matrix.candidate_columns])))
    actual_history = {int(matrix.item_ids[index]): int(seconds) for index, seconds in zip(history.indices, history.data)}
    direct_history = {
        int(item): int(seconds) for item, seconds in
        a.loc[a.user_id.eq(user)].groupby("tv_show_id").overlap_seconds.sum().items()
    }
    profile = np.asarray(
        model.item_vectors[history.indices].multiply(np.log1p(history.data / 60)[:, None]).sum(axis=0)
    ).ravel()
    profile /= np.linalg.norm(profile)
    manual_scores = np.asarray(model.item_vectors[matrix.candidate_columns] @ profile).ravel()
    profile_error = float(np.max(np.abs(scores - manual_scores)))
    ranked = rank_scores(scores[None, :], candidate_ids, [user])[user]
    if not mask_ok or slice_error > 1e-12 or actual_history != direct_history or profile_error > 1e-12:
        raise AssertionError("History, profile or candidate-mask alignment failed")
    if ranked != list(map(int, candidate_ids[order[:5]])):
        raise AssertionError("Ranking direction or id tie-breaking failed")
    print(f"History id/seconds alignment: PASS ({history.nnz} programmes); independent profile cosine max error={profile_error:.3e}")
    print(f"Candidate mask equals sorted B ids: {mask_ok}; full-score slice max error={slice_error:.3e}")
    print("Descending score / ascending id tie ranking: PASS")
    print("CBF top five: id | cosine | watched in A | relevant in B | title")
    for position in order[:5]:
        item = int(candidate_ids[position])
        print(f"{item} | {scores[position]:.9f} | {item in actual_history} | "
              f"{item in protocol.relevant_by_user[user]} | {lookup[item]['tv_show_title']}")
    categories = pd.DataFrame({
        "category": [lookup[int(matrix.item_ids[index])].get("tv_show_category", "") for index in history.indices],
        "seconds": history.data, "log_weight": np.log1p(history.data / 60),
        "programmes": np.ones(history.nnz, dtype=int),
    }).groupby("category").sum().sort_values("log_weight", ascending=False)
    categories["seconds_share"] = categories.seconds / categories.seconds.sum()
    categories["log_weight_share"] = categories.log_weight / categories.log_weight.sum()
    print("History category contributions (top five by log weight):")
    print(categories.iloc[:5].to_string(float_format=lambda value: f"{value:.6f}"))
    print(f"Top-five history seconds share={history.data[history_order[:5]].sum() / history.data.sum():.6%}; "
          f"log-weight share={np.log1p(history.data[history_order[:5]] / 60).sum() / np.log1p(history.data / 60).sum():.6%}")
    print("Largest profile features:")
    inverse = {value: key for key, value in model.vocabulary.items()}
    for index in np.argsort(-profile, kind="stable")[:8]:
        print(f"{inverse[int(index)]} {profile[index]:.9f}")

    all_scores = model.score(matrix.test_rows, matrix.candidate_columns)
    popularity = PopularityRecommender().fit(a, protocol.candidates)
    predictions = protocol.finalize_predictions(
        rank_scores(all_scores, candidate_ids, users), popularity.ranking, popularity.history_users
    )
    current = protocol.score(predictions)
    slots = films = seen = zero_popularity = hits = no_hit_users = 0
    for target, relevant in protocol.relevant_by_user.items():
        target_row = int(np.searchsorted(matrix.user_ids, np.uint64(target)))
        target_history = set(map(int, matrix.item_ids[matrix.seconds.getrow(target_row).indices]))
        no_hit_users += int(not (set(predictions[target]) & relevant))
        for item in predictions[target]:
            slots += 1
            films += int(lookup.get(item, {}).get("tv_show_category") == "Фильмы")
            seen += int(item in target_history)
            zero_popularity += int(popularity.counts.loc[item] == 0)
            hits += int(item in relevant)
    print(f"Evaluated top-five slots={slots:,}: films={films:,}; watched in A={seen:,}; "
          f"zero A viewers={zero_popularity:,}; relevant in B={hits:,}")
    print(f"Evaluated users with no relevant CBF top-five hit: {no_hit_users:,}/{len(protocol.relevant_by_user):,}")
    print(f"CBF MAP@5 current (unchanged): {current:.9f}; evaluated users={len(protocol.relevant_by_user):,}")
    if abs(current - before) > 1e-9:
        raise AssertionError("CBF no longer reproduces its saved before score")
    print("Conclusion: no implementation defect found; log-weighted multi-interest centroid favours broad film metadata.")
    print("Saved: results/cbf_diagnostics.txt")
    print(f"Runtime: {time.perf_counter() - started:.3f} seconds")


if __name__ == "__main__":
    output = ROOT / "results" / "cbf_diagnostics.txt"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as report, redirect_stdout(Tee(sys.stdout, report)):
        run()

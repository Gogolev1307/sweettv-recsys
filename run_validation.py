"""Run the frozen A -> B popularity validation and write the complete report."""

from __future__ import annotations

import argparse
import platform
import random
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd
import scipy

from src.data import (
    A_START, B_START, CACHE_DIR, DATA_DIR, ROOT, TEST_END, TEST_FILE, TEST_START,
    TRAIN_FILES, VIEW_FILE, airings_in_window, cached_result, load_submission,
    load_test_schedule_summary, load_training_schedule, load_views,
)
from src.evaluate import ValidationProtocol
from src.join import (
    clock_alignment_counts, history_users_across_windows, join_views_to_airings,
    watched_interactions,
)
from src.popularity import PopularityRecommender

QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90, 0.99)


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
            stream.flush()
        return len(text)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def quantiles(series: pd.Series) -> str:
    return " ".join(f"p{round(q * 100):02d}={value:,.3f}" for q, value in series.quantile(QUANTILES).items())


def print_coverage(name, result):
    d = result.diagnostics
    print(f"{name}: {d['views']:,} view intervals; {d['matched_views']:,} matched ({d['matched_views'] / d['views']:.3%}); "
          f"{d['multi_airing_views']:,} overlap multiple airings")
    print(f"  View-airing records: {d['airing_pairs']:,} = {d['labelled_pairs']:,} labelled + {d['unlabelled_pairs']:,} unlabelled")
    print(f"  Viewing seconds: {d['view_seconds']:,}; covered (union): {d['covered_seconds']:,} "
          f"({d['covered_seconds'] / d['view_seconds']:.3%}); uncovered: {d['view_seconds'] - d['covered_seconds']:,}")
    print(f"  Labelled seconds (union): {d['labelled_seconds']:,}; id=0 seconds (union): {d['unlabelled_seconds']:,} "
          f"({d['unlabelled_seconds'] / d['view_seconds']:.3%} of viewing)")
    print(f"  Additive airing-overlap seconds: {d['additive_overlap_seconds']:,}; id=0: {d['additive_unlabelled_seconds']:,}; "
          f"labelled (user, airing) aggregates: {len(result.interactions):,}")


def run(args, started):
    random.seed(args.seed)
    np.random.seed(args.seed)
    data_dir, cache_dir = args.data_dir, None if args.no_cache else args.cache_dir
    print("sweet.tv popularity validation (MAP@5)")
    print(f"Python {platform.python_version()} | pandas {pd.__version__} | numpy {np.__version__} | scipy {scipy.__version__}")
    print(f"Random seed: {args.seed} (popularity itself is deterministic)")
    print(f"A: [{A_START}, {B_START}); B: [{B_START}, {TEST_START}); test: [{TEST_START}, {TEST_END})")
    print("Windows are half-open; view seconds are clipped at boundaries before thresholding.")
    print(f"Cache: {'disabled' if cache_dir is None else str(cache_dir.relative_to(ROOT) if cache_dir.is_relative_to(ROOT) else cache_dir)}; "
          "keys include source path/size/mtime, package versions and pipeline version")

    views = load_views(data_dir, cache_dir)
    schedule, checks = load_training_schedule(data_dir, cache_dir)
    test_summary = load_test_schedule_summary(data_dir, cache_dir)
    submission = load_submission(data_dir)
    test_users = tuple(map(int, submission.user_id))
    source_paths = [data_dir / filename for filename in (VIEW_FILE, *TRAIN_FILES)]
    print("\nDATA CHECKS")
    print(f"Views: {len(views):,}; users: {views.user_id.nunique():,}; user_id dtype: {views.user_id.dtype}; "
          f"test users: {len(test_users):,} ({submission.user_id.dtype})")
    print(f"View start range: {views.start_time.min()} .. {views.start_time.max()}; last stop: {views.stop_time.max()}")
    mismatches = views.duraton - views.view_seconds
    print(f"duraton != stop-start: {int((mismatches != 0).sum()):,}; differences in seconds: "
          f"{mismatches[mismatches != 0].value_counts().to_dict()}; using stop-start")
    for filename, summary in (*checks["files"].items(), (TEST_FILE, test_summary)):
        print(f"{filename}: {summary['rows']:,} rows; starts {summary['start_min']} .. {summary['start_max']}; "
              f"nonpositive durations: {summary['nonpositive_duration_rows']:,}")
    overlap_start, overlap_end = checks["calendar_overlap"]
    print(f"Training files' calendar overlap: {overlap_start.date()} .. {overlap_end.date()}; "
          f"cross-file duplicate keys: {checks['cross_file_duplicate_keys']:,}")
    print(f"Concatenate then deduplicate (channel_id, start_time), keep last: {checks['duplicates_removed']:,} removed; "
          f"dates: {checks['duplicate_dates']}")
    print(f"Deduplicated schedule: {len(schedule):,} unique airings; "
          f"remaining nonpositive durations removed: {checks['nonpositive_after_dedup']:,}")
    known = views.vsetv_id.isin(schedule.channel_id.unique())
    missing = sorted(map(int, views.loc[~known, "vsetv_id"].unique()))
    print(f"vsetv_id -> channel_id: {int(known.sum()):,}/{len(views):,} views ({known.mean():.3%}); "
          f"missing channels: {missing}")
    alignment = cached_result(
        cache_dir, "clock_alignment", source_paths, {"offset_hours": [0]},
        lambda: clock_alignment_counts(views, schedule, offsets=(0,)),
    )
    print(f"Same-channel exact view-start/programme-start coincidences at offset 0: {alignment[0]:,}")
    print("Times use the supplied same-clock convention; no timezone shift applied.")

    results = []
    for name, lower, upper in (("A", A_START, B_START), ("B", B_START, TEST_START)):
        result = cached_result(
            cache_dir, f"join_{name}", source_paths,
            {"window_start": str(lower), "window_end": str(upper), "join_version": 1},
            lambda lower=lower, upper=upper: join_views_to_airings(views, schedule, lower, upper),
        )
        results.append(result)
    joined_a, joined_b = results
    print("\nJOIN COVERAGE (before the watch threshold; id=0 never becomes an item)")
    print_coverage("A", joined_a)
    print_coverage("B", joined_b)

    print("\nTHRESHOLD STATISTICS (labelled airings; one aggregate per user/airing/window)")
    print(f"View interval seconds, weeks 11-30: {quantiles(views.view_seconds)}")
    for name, result in zip(("A", "B"), results):
        print(f"{name} per-airing overlap seconds: {quantiles(result.interactions.overlap_seconds)}")
        print(f"{name} overlap/programme-duration: {quantiles(result.interactions.overlap_fraction)}")
        print(f"{name} fractions > 1 from additive sessions: {int((result.interactions.overlap_fraction > 1).sum()):,}")

    schedule_b = airings_in_window(schedule, B_START, TEST_START)
    print("\nVALIDATION PROTOCOL")
    print("Fit: A labelled interactions only; truth: each test user's watched programme ids in B.")
    print("Candidates: labelled ids in schedule rows with airing starts inside B.")
    print("Re-watching allowed; five distinct candidates per user; A-popularity fill and cold-start fallback.")
    print("Test-weeks schedule is read only for input checks; B interactions are used only for truth/diagnostics.")
    all_test = set(test_users)
    metrics = {}
    # One join, re-thresholded consistently in BOTH A and B for each experiment.
    for threshold in sorted({60, 300, 900, args.threshold_seconds}):
        watched_a = watched_interactions(joined_a.interactions, threshold)
        watched_b = watched_interactions(joined_b.interactions, threshold)
        protocol = ValidationProtocol.from_interactions(test_users, schedule_b, watched_b)
        model = PopularityRecommender().fit(watched_a, protocol.candidates)
        predictions = protocol.finalize_predictions(model.predict(test_users), model.ranking, model.history_users)
        score = protocol.score(predictions)
        evaluated = len(protocol.relevant_by_user)
        no_truth = len(all_test - protocol.relevant_by_user.keys())
        if evaluated + no_truth != 1260:
            raise AssertionError("Test-user accounting does not close")
        metrics[threshold] = {
            "map": score,
            "evaluated": evaluated,
            "no_truth": no_truth,
            "cold_a": len(all_test - model.history_users),
            "a_interactions": len(watched_a),
            "b_interactions": len(watched_b),
        }
        if threshold == args.threshold_seconds:
            primary_protocol, primary_model = protocol, model

    primary = metrics[args.threshold_seconds]
    history_all = history_users_across_windows(results, args.threshold_seconds)
    print(f"\nPOPULARITY BASELINE (watch >= {args.threshold_seconds} seconds)")
    print(f"Candidate programmes: {len(primary_protocol.candidates):,}; ranking: distinct A viewers descending, id ascending for ties")
    outside_candidates = set().union(*primary_protocol.relevant_by_user.values()) - primary_protocol.candidates
    affected_users = sum(bool(items & outside_candidates) for items in primary_protocol.relevant_by_user.values())
    print(f"B-truth carry-in programmes outside B-start candidates: {len(outside_candidates):,} across {affected_users:,} users; truth retained")
    print(f"Watched user-airings: A={primary['a_interactions']:,}; B={primary['b_interactions']:,}")
    top_five = [(item, int(primary_model.counts.loc[item])) for item in primary_model.ranking[:5]]
    print(f"Top five (programme id, distinct A viewers): {top_five}")
    print(f"Validation MAP@5: {primary['map']:.9f}")
    print(f"All {len(test_users):,} users receive exactly five distinct valid ids; verified before scoring.")
    print("\nCOHORT ACCOUNTING")
    print(f"{len(test_users):,} test users = {primary['evaluated']:,} evaluated + {primary['no_truth']:,} with no relevant B programme (exact)")
    print(f"Test users with no usable A history: {primary['cold_a']:,}")
    print(f"Test users with no usable weeks 11-30 history: {len(all_test - history_all):,}")
    evaluated_cold = len(set(primary_protocol.relevant_by_user) - primary_model.history_users)
    print(f"Evaluated users with no usable A history: {evaluated_cold:,}")
    print("\nTHRESHOLD SENSITIVITY (same protocol; threshold applied independently in A and B)")
    for threshold in (60, 300, 900):
        metric = metrics[threshold]
        print(f"{threshold:>3} seconds: popularity MAP@5={metric['map']:.9f}; evaluated={metric['evaluated']:,}; "
              f"no B truth={metric['no_truth']:,}; no A history={metric['cold_a']:,}")
    print(f"\nResults file: {args.output.relative_to(ROOT) if args.output.is_relative_to(ROOT) else args.output}")
    print(f"Runtime: {time.perf_counter() - started:.3f} seconds")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threshold-seconds", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "validation_popularity.txt")
    parser.add_argument("--no-cache", action="store_true", help="Recompute without reading or writing intermediate caches")
    args = parser.parse_args()
    if args.threshold_seconds <= 0:
        parser.error("--threshold-seconds must be positive")
    args.output = args.output.resolve()
    if args.output.is_relative_to(args.data_dir.resolve()):
        parser.error("Results must not be written inside the raw-data directory")
    if args.cache_dir.resolve().is_relative_to(args.data_dir.resolve()):
        parser.error("Caches must not be written inside the raw-data directory")
    started = time.perf_counter()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as report, redirect_stdout(Tee(sys.stdout, report)):
        run(args, started)


if __name__ == "__main__":
    main()

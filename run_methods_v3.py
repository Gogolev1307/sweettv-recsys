"""Calibrated T2/T3 selection of slot/exposure/personal-habit score fusion."""

from __future__ import annotations

import os

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[variable] = "1"

import hashlib
import json
import platform
import random
import sys
import time
from contextlib import redirect_stdout

import numpy as np
import pandas as pd
import scipy

from run_validation import Tee
from src.assoc import AssociationRecommender
from src.calibration import CALIBRATION_STATEMENT, KAGGLE_FEEDBACK, KAGGLE_PUBLIC_LIFTS
from src.cbf import ContentRecommender
from src.data import (
    A_START, B_START, CACHE_DIR, DATA_DIR, ROOT, TEST_START, TRAIN_FILES, VIEW_FILE,
    airings_in_window, cached_result, load_fitting_metadata, load_submission, load_training_schedule,
)
from src.experiment import load_calendar_join, load_slot_profile
from src.exposure import ExposureRecommender
from src.hybrid import (
    COMPONENTS, V3_SELECTION_RULE, calibrated_truth_mean, fuse_days_scores, fuse_habit_scores,
    fuse_scores, habit_weight_grid, normalize_components, normalize_days_components, normalize_habit_components,
)
from src.join import programme_day_counts, watched_interactions
from src.matrix import InteractionMatrix, aligned_day_matrix, rank_scores
from src.mf import ImplicitALS
from src.popularity import PopularityRecommender
from src.ppr import PersonalizedPageRank
from src.rewatch import RewatchDaysRecommender, RewatchRecommender
from src.submission import load_frozen_configurations, load_frozen_v2_configuration
from src.truths import TRUTH_DEFINITIONS, validation_truths


def json_text(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def run(report, started):
    random.seed(42)
    np.random.seed(42)
    configurations, legacy_digest = load_frozen_configurations(ROOT / "results" / "validation_methods.csv")
    v2, v2_digest = load_frozen_v2_configuration(ROOT / "results" / "hybrid_v2_selected.json")
    previous = pd.read_csv(ROOT / "results" / "validation_methods_v2.csv").loc[lambda frame: frame.selected].set_index("method")
    print("sweet.tv calibrated habit validation v3 (MAP@5)")
    print(f"Python {platform.python_version()} | numpy {np.__version__} | scipy {scipy.__version__} | pandas {pd.__version__}")
    try:
        import sklearn
        print(f"scikit-learn {sklearn.__version__} installed; TF-IDF uses NumPy/SciPy")
    except ImportError:
        print("scikit-learn not installed; TF-IDF uses NumPy/SciPy")
    print("Seed: 42; BLAS/OpenMP threads: 1; labelled fitting-airing threshold: 300s")
    print(f"Fit A=[{A_START}, {B_START}); target/truth B=[{B_START}, {TEST_START})")
    print("Protocol: 6,376 candidates; re-watching allowed; five distinct ids; ascending-id score ties; same A-popularity fill/cold start.")
    print("Target schedule fields: channel_id, start_time, duration, tv_show_id only; no target content features or viewing in fitting.")
    print("CBF scope: A-only metadata for both history/candidate vectors; B-only unknown content has zero vectors.")
    print("Slot uses ALL clipped raw A viewing, including short views, unknown channels and unlabelled airtime; hourly splits include weekend changes.")
    print("Exposure = watched A days / (scheduled A airing days + 3) * scheduled B airing days; repeats/channels count once per calendar date.")
    print(f"Selection rule: {V3_SELECTION_RULE}.")
    print(CALIBRATION_STATEMENT)
    print(f"Frozen MF: {json_text(configurations['MF'])}; frozen PPR: {json_text(configurations['PPR'])}")
    print("Configuration ties retain the first ascending (w_days,w_exp,w_slot,w_ppr,w_mf,w_pop) grid point.")
    print("Hybrid-grid runtimes cover fusion/ranking/evaluation; component fits and shared normalization are timed separately.")

    prep = time.perf_counter()
    schedule, _ = load_training_schedule()
    users = tuple(map(int, load_submission().user_id))
    candidates = frozenset(map(int, airings_in_window(schedule, B_START, TEST_START).tv_show_id)) - {0}
    joined_a, joined_b = load_calendar_join("A", A_START, B_START), load_calendar_join("B", B_START, TEST_START)
    protocols, accounting = validation_truths(users, candidates, joined_b)
    if len(candidates) != 6376 or len(users) != 1260 or accounting["T1"]["evaluated_users"] != 1246:
        raise AssertionError("Validation candidate/cohort changed")
    watched = watched_interactions(joined_a.interactions)
    matrix = cached_result(
        CACHE_DIR, "method_matrix", [DATA_DIR / name for name in (VIEW_FILE, *TRAIN_FILES, "submission.csv")],
        {"threshold": 300, "matrix_version": 1}, lambda: InteractionMatrix.build(watched, users, candidates),
    )
    days = aligned_day_matrix(programme_day_counts(joined_a), matrix)
    candidate_ids = matrix.item_ids[matrix.candidate_columns]
    metadata = load_fitting_metadata(A_START, B_START)
    del joined_a, joined_b
    print(f"A matrix: {matrix.seconds.shape[0]:,} users x {matrix.seconds.shape[1]:,} programmes; {matrix.seconds.nnz:,} observed pairs")
    print(f"Shared input preparation: {time.perf_counter() - prep:.3f}s")
    for truth, stats in accounting.items():
        print(f"{truth}: 1,260 = {stats['evaluated_users']:,} evaluated + {stats['no_relevant_users']:,} no relevant; "
              f"median all/evaluated={stats['median_relevant_all_users']:g}/{stats['median_relevant_evaluated_users']:g}; "
              f"{TRUTH_DEFINITIONS[truth]}")

    records = []
    point_started = time.perf_counter()
    popularity = PopularityRecommender().fit(watched, candidates)
    pop_scores = popularity.counts.reindex(candidate_ids).to_numpy(dtype=np.float64)
    cold = set(users) - popularity.history_users
    print(f"A-cold test users: {len(cold)}; all methods use the same forced popularity fallback")

    def record(method, parameters, scores, point_started, *, positive_only=True, secondary=None, diagnostics=None, quiet=False):
        proposed = rank_scores(scores, candidate_ids, users, positive_only=positive_only, secondary_scores=secondary)
        predictions = protocols["T1"].finalize_predictions(proposed, popularity.ranking, popularity.history_users)
        if any(predictions[user] != popularity.ranking[:5] for user in cold):
            raise AssertionError("Method-specific cold-start behavior")
        metrics = {name: protocol.score(predictions) for name, protocol in protocols.items()}
        row = {
            "method": method, "configuration_id": f"config_{len(records):04d}", "hyperparameters": json_text(parameters),
            "selected": method != "hybrid_v3", "seed": 42, "selection_mean_T2_T3": calibrated_truth_mean(metrics),
            "runtime_seconds": time.perf_counter() - point_started, "diagnostics": json_text(diagnostics or {}),
        }
        for truth, value in metrics.items():
            row[f"map_at_5_{truth}"] = value
            for key, count in accounting[truth].items():
                row[f"{key}_{truth}"] = count
        if method in previous.index and method != "CBF":
            if any(abs(value - previous.loc[method, f"map_at_5_{truth}"]) > 1e-9 for truth, value in metrics.items()):
                raise AssertionError(f"Fixed method {method} changed under the existing truths")
        records.append(row)
        if not quiet:
            print(f"{method}: " + " ".join(f"{truth}={value:.9f}" for truth, value in metrics.items())
                  + f" mean_T2_T3={row['selection_mean_T2_T3']:.9f}; runtime={row['runtime_seconds']:.3f}s")
        return len(records) - 1

    print("\nFIXED METHODS AND NEW HABIT COMPONENTS")
    record("popularity", configurations["popularity"], np.broadcast_to(pop_scores, (len(users), len(candidates))), point_started)
    point_started = time.perf_counter()
    seconds_scores = RewatchRecommender().fit(matrix.seconds).score(matrix.test_rows, matrix.candidate_columns)
    record("rewatch", {}, seconds_scores, point_started)
    point_started = time.perf_counter()
    day_scores = RewatchDaysRecommender().fit(days, matrix.seconds).score(matrix.test_rows, matrix.candidate_columns)
    record("rewatch_days", {"tie_break": "seconds_then_id"}, day_scores, point_started, secondary=seconds_scores)
    point_started = time.perf_counter()
    mf = ImplicitALS(**configurations["MF"]).fit(matrix.seconds)
    mf_scores = mf.score(matrix.test_rows, matrix.candidate_columns)
    record("MF", configurations["MF"], mf_scores, point_started, positive_only=False)
    del mf
    point_started = time.perf_counter()
    parameters = configurations["PPR"]
    ppr = PersonalizedPageRank(**{key: value for key, value in parameters.items() if key != "alpha"}).fit(matrix.seconds)
    ppr_scores, ppr_diagnostics = ppr.score(matrix.test_rows, matrix.candidate_columns, parameters["alpha"])
    record("PPR", parameters, ppr_scores, point_started, diagnostics=ppr_diagnostics)
    if ppr_diagnostics["non_converged_users"]:
        raise AssertionError("PPR did not converge")
    print(f"PPR: non-converged=0; max residual={ppr_diagnostics['max_final_l1_residual']:.3e}; max mass error={ppr_diagnostics['max_final_mass_error']:.3e}")
    del ppr
    point_started = time.perf_counter()
    cbf = ContentRecommender().fit(matrix.seconds, matrix.item_ids, metadata, metadata)
    record("CBF", {**configurations["CBF"], "metadata_scope": "A_only"},
           cbf.score(matrix.test_rows, matrix.candidate_columns), point_started,
           diagnostics={"zero_vector_items": cbf.zero_vector_items, "target_content_fields_used": 0})
    print(f"CBF: A-only content snapshots={len(metadata):,}; zero-vector items={cbf.zero_vector_items:,}; target content fields used=0")
    del cbf, metadata
    point_started = time.perf_counter()
    association = AssociationRecommender(
        minimum_support=configurations["association"]["minimum_support"],
        minimum_lift=configurations["association"]["lift_strictly_greater_than"],
    ).fit(matrix.seconds)
    record("association", configurations["association"], association.score(matrix.test_rows, matrix.candidate_columns), point_started,
           diagnostics={"rules_kept": association.rule_count})
    del association
    point_started = time.perf_counter()
    slot = load_slot_profile("A", users, A_START, B_START)
    slot_scores = slot.score(users, candidate_ids, schedule, B_START, TEST_START)
    record("slot", {"slot": "channel_weekend_hour", "viewing": "all_fit_views"}, slot_scores, point_started, diagnostics=slot.diagnostics)
    print(f"Slot: view intervals={slot.diagnostics['view_intervals']:,}; all-view seconds={slot.diagnostics['view_seconds']:,}; "
          f"profiles={slot.diagnostics['nonempty_profiles']:,}; slots nnz={slot.diagnostics['nonzero_slots']:,}; "
          f"max profile row-sum error={slot.diagnostics['max_row_sum_error']:.3e}")
    del slot
    point_started = time.perf_counter()
    exposure = ExposureRecommender().fit(days, matrix.item_ids, schedule, A_START, B_START)
    exposure_scores = exposure.score(matrix.test_rows, matrix.candidate_columns, schedule, B_START, TEST_START)
    record("exposure", {"smoothing_days": 3}, exposure_scores, point_started, diagnostics=exposure.diagnostics)
    print(f"Exposure: max scheduled A/B days={exposure.diagnostics['max_fit_airing_days']}/{exposure.diagnostics['max_target_airing_days']}; "
          f"max smoothed habit rate={exposure.diagnostics['maximum_habit_rate']:.9f}")
    del exposure, schedule, watched
    point_started = time.perf_counter()
    components = normalize_components(pop_scores, seconds_scores, ppr_scores, mf_scores)
    record("hybrid_old", configurations["hybrid"],
           fuse_scores(components, tuple(configurations["hybrid"][name] for name in COMPONENTS)), point_started)
    del components, seconds_scores
    point_started = time.perf_counter()
    components = normalize_days_components(day_scores, ppr_scores, pop_scores, mf_scores)
    record("hybrid_v2", v2["weights"], fuse_days_scores(components, v2["weights"]), point_started)
    del components

    print("\nKAGGLE CALIBRATION (user-provided feedback; public lifts rounded as supplied)")
    print(f"Popularity Kaggle public/private: {KAGGLE_FEEDBACK['popularity']['public']:.5f}/{KAGGLE_FEEDBACK['popularity']['private']:.5f}")
    print("method         T1_lift   T2_lift   T3_lift   T4_lift   Kaggle_public_lift   public/private")
    baseline = records[0]
    for method in ("hybrid_old", "hybrid_v2"):
        row = next(row for row in records if row["method"] == method)
        feedback = KAGGLE_FEEDBACK[method]
        row.update({"kaggle_public": feedback["public"], "kaggle_private": feedback["private"],
                    "kaggle_public_lift_reported": KAGGLE_PUBLIC_LIFTS[method]})
        lifts = [row[f"map_at_5_{truth}"] / baseline[f"map_at_5_{truth}"] for truth in protocols]
        print(f"{method:<14} " + " ".join(f"{value:.6f}x" for value in lifts)
              + f" {KAGGLE_PUBLIC_LIFTS[method]:.2f}x                {feedback['public']:.5f}/{feedback['private']:.5f}")
    print(CALIBRATION_STATEMENT)

    normalization_started = time.perf_counter()
    components = normalize_habit_components(day_scores, exposure_scores, slot_scores, ppr_scores, mf_scores, pop_scores)
    normalization_seconds = time.perf_counter() - normalization_started
    del day_scores, exposure_scores, slot_scores, ppr_scores, mf_scores
    print(f"\nV3 shared normalization: {normalization_seconds:.3f}s")
    print("V3 GRID: 1,535 nonzero configurations; selecting mean(T2,T3)")
    best_index, best_weights = None, None
    for weights in habit_weight_grid():
        point_started = time.perf_counter()
        index = record("hybrid_v3", weights, fuse_habit_scores(components, weights), point_started, quiet=True)
        if best_index is None or records[index]["selection_mean_T2_T3"] > records[best_index]["selection_mean_T2_T3"]:
            best_index, best_weights = index, weights
    records[best_index]["selected"] = True
    del components
    for row in records:
        for truth in protocols:
            row[f"lift_over_popularity_{truth}"] = row[f"map_at_5_{truth}"] / baseline[f"map_at_5_{truth}"]
    frame = pd.DataFrame(records)
    if len(frame) != 1546 or frame.method.eq("hybrid_v3").sum() != 1535 or frame.selected.sum() != 12:
        raise AssertionError("Missing fixed methods/grid points")
    output = ROOT / "results" / "validation_methods_v3.csv"
    frame.to_csv(output, index=False, float_format="%.9f")
    selected = records[best_index]
    manifest = {
        "seed": 42, "threshold_seconds": 300, "selection_rule": V3_SELECTION_RULE,
        "weights": best_weights, "component_configurations": {name: configurations[name] for name in ("MF", "PPR")},
        "selection_mean_T2_T3": selected["selection_mean_T2_T3"],
        "validation_scores": {truth: selected[f"map_at_5_{truth}"] for truth in protocols},
        "truth_definitions": TRUTH_DEFINITIONS, "cohort_accounting": accounting,
        "validation_results_file": output.name, "validation_results_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "legacy_config_sha256": legacy_digest, "v2_selection_manifest_sha256": v2_digest,
        "validation_candidate_count": len(candidates), "exposure_smoothing_days": 3,
        "slot_definition": "channel_id, weekend, hour; all fit viewing; hourly splits; row-normalised seconds",
        "target_schedule_fields": ["channel_id", "start_time", "duration", "tv_show_id"],
        "calibration": {"statement": CALIBRATION_STATEMENT, "feedback": KAGGLE_FEEDBACK,
                        "reported_public_lifts": KAGGLE_PUBLIC_LIFTS},
    }
    (ROOT / "results" / "hybrid_v3_selected.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    top = frame.loc[frame.method.eq("hybrid_v3")].sort_values("selection_mean_T2_T3", ascending=False, kind="stable").head(10)
    print("TOP 10 V3 CONFIGURATIONS")
    for row in top.itertuples(index=False):
        print(f"{row.hyperparameters}: T1={row.map_at_5_T1:.9f} T2={row.map_at_5_T2:.9f} T3={row.map_at_5_T3:.9f} "
              f"T4={row.map_at_5_T4:.9f} mean_T2_T3={row.selection_mean_T2_T3:.9f}; runtime={row.runtime_seconds:.3f}s")
    print(f"Selected v3 weights: {json_text(best_weights)}; mean_T2_T3={selected['selection_mean_T2_T3']:.9f}")
    print("\nSELECTED/FIXED COMPARISON")
    print("method         T1          T2          T3          T4          mean_T2_T3 runtime_s")
    for row in frame.loc[frame.selected].itertuples(index=False):
        print(f"{row.method:<14} {row.map_at_5_T1:.9f} {row.map_at_5_T2:.9f} {row.map_at_5_T3:.9f} {row.map_at_5_T4:.9f} "
              f"{row.selection_mean_T2_T3:.9f} {row.runtime_seconds:.3f}")
    report.write("\nALL 1,546 CONFIGURATIONS\n" + frame.to_string(index=False, float_format=lambda value: f"{value:.9f}") + "\n\n")
    report.flush()
    print("Saved: results/validation_methods_v3.csv; results/validation_methods_v3.txt; results/hybrid_v3_selected.json")
    print(f"Total runtime: {time.perf_counter() - started:.3f} seconds")


if __name__ == "__main__":
    started = time.perf_counter()
    output = ROOT / "results" / "validation_methods_v3.txt"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as report, redirect_stdout(Tee(sys.stdout, report)):
        run(report, started)

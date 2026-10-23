"""Published competition-rule truth and channel-aware completion habit selection."""

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
from src.calibration import KAGGLE_FEEDBACK, KAGGLE_PUBLIC_LIFTS
from src.cbf import ContentRecommender
from src.completion import CompletionRecommender
from src.data import (
    A_START, B_START, CACHE_DIR, DATA_DIR, ROOT, TEST_START, TRAIN_FILES, VIEW_FILE,
    airings_in_window, cached_result, load_fitting_metadata, load_submission, load_training_schedule,
)
from src.experiment import load_calendar_join, load_slot_profile
from src.exposure import ExposureRecommender
from src.hybrid import (
    COMPLETION_K_VALUES, COMPONENTS, V4_SELECTION_RULE, competition_truth_score, completion_weight_grid,
    fuse_completion_scores, fuse_days_scores, fuse_habit_scores, fuse_scores, minmax_normalize,
    normalize_components, normalize_days_components, normalize_habit_components,
)
from src.join import programme_day_counts, watched_interactions
from src.matrix import InteractionMatrix, aligned_day_matrix, rank_scores
from src.mf import ImplicitALS
from src.popularity import PopularityRecommender
from src.ppr import PersonalizedPageRank
from src.rewatch import RewatchDaysRecommender, RewatchRecommender
from src.submission import load_frozen_configurations, load_frozen_v2_configuration, load_frozen_v3_configuration
from src.truths import ALL_TRUTH_DEFINITIONS, COMPETITION_GROUND_TRUTH_QUOTE, validation_truths


def json_text(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def run(report, started):
    random.seed(42)
    np.random.seed(42)
    configurations, legacy_digest = load_frozen_configurations(ROOT / "results" / "validation_methods.csv")
    v2, _ = load_frozen_v2_configuration(ROOT / "results" / "hybrid_v2_selected.json")
    v3, v3_digest = load_frozen_v3_configuration(ROOT / "results" / "hybrid_v3_selected.json")
    previous = pd.read_csv(ROOT / "results" / "validation_methods_v3.csv").loc[lambda frame: frame.selected].set_index("method")
    print("sweet.tv published-rule completion validation v4 (MAP@5)")
    print(f"Python {platform.python_version()} | numpy {np.__version__} | scipy {scipy.__version__} | pandas {pd.__version__}")
    try:
        import sklearn
        print(f"scikit-learn {sklearn.__version__} installed; TF-IDF uses NumPy/SciPy")
    except ImportError:
        print("scikit-learn not installed; TF-IDF uses NumPy/SciPy")
    print("Published competition definition (quote supplied by user):")
    print(f'"{COMPETITION_GROUND_TRUTH_QUOTE}"')
    print(f"Selection rule: {V4_SELECTION_RULE}.")
    print("TK: per-user/labelled-airing overlap divided by full airing duration; >=0.8 airings counted once; top 5 programme counts, id ties.")
    print("No 300s cutoff for TK or completion counts. Existing 300s A history still controls popularity fill and the shared cold-start fallback.")
    print(f"Fit A=[{A_START}, {B_START}); target/truth B=[{B_START}, {TEST_START})")
    print("Protocol: same 6,376 candidates, re-watching, five distinct ids, ascending-id score ties; truths are not candidate-filtered.")
    print("nA/nT count unique labelled channel/start airings intersecting their windows; full airing duration remains the screen-time denominator.")
    print("Target schedule fields: channel_id, start_time, duration, tv_show_id only; CBF uses A-only content metadata.")
    print("Seed: 42; BLAS/OpenMP threads: 1; grid rows time fusion/ranking/evaluation, with shared fits/normalization separate.")
    print(f"Frozen PPR: {json_text(configurations['PPR'])}; frozen MF (comparators): {json_text(configurations['MF'])}")

    prep = time.perf_counter()
    schedule, _ = load_training_schedule()
    users = tuple(map(int, load_submission().user_id))
    candidates = frozenset(map(int, airings_in_window(schedule, B_START, TEST_START).tv_show_id)) - {0}
    joined_a, joined_b = load_calendar_join("A", A_START, B_START), load_calendar_join("B", B_START, TEST_START)
    protocols, accounting = validation_truths(users, candidates, joined_b, include_tk=True)
    if len(users) != 1260 or len(candidates) != 6376 or accounting["T1"]["evaluated_users"] != 1246:
        raise AssertionError("Original protocol candidate/cohort changed")
    watched = watched_interactions(joined_a.interactions)
    matrix = cached_result(
        CACHE_DIR, "method_matrix", [DATA_DIR / name for name in (VIEW_FILE, *TRAIN_FILES, "submission.csv")],
        {"threshold": 300, "matrix_version": 1}, lambda: InteractionMatrix.build(watched, users, candidates),
    )
    candidate_ids = matrix.item_ids[matrix.candidate_columns]
    days = aligned_day_matrix(programme_day_counts(joined_a), matrix)
    completion_started = time.perf_counter()
    completion = CompletionRecommender().fit(joined_a.interactions, users, candidate_ids, schedule, A_START, B_START)
    completion.prepare_target(schedule, B_START, TEST_START)
    completion_seconds = time.perf_counter() - completion_started
    metadata = load_fitting_metadata(A_START, B_START)
    del joined_a, joined_b
    print(f"A matrix: {matrix.seconds.shape[0]:,} users x {matrix.seconds.shape[1]:,} programmes; {matrix.seconds.nnz:,} observed >=300s pairs")
    print(f"Shared input preparation: {time.perf_counter() - prep:.3f}s; completion-count preparation={completion_seconds:.3f}s")
    d = completion.diagnostics
    print(f"Completion A: >=80% user-airings={d['completed_80_user_airings']:,}; >=50%={d['completed_50_user_airings']:,}; "
          f"candidate programme/channel pairs={d['fit_candidate_programme_channels']:,}; nA={d['fit_candidate_airings']:,}; "
          f"nT on fitted pairs={d['target_airings_on_fitted_programme_channels']:,}")
    print("\nTRUTH COHORTS")
    for truth, stats in accounting.items():
        print(f"{truth}: 1,260 = {stats['evaluated_users']:,} evaluated + {stats['no_relevant_users']:,} no relevant; "
              f"median all/evaluated={stats['median_relevant_all_users']:g}/{stats['median_relevant_evaluated_users']:g}; "
              f"{ALL_TRUTH_DEFINITIONS[truth]}")
    print(f"TK relevant-set sizes including empty users: {json_text(accounting['TK']['relevance_size_counts'])}")
    print(f"TK qualifying user-airings among test users: {accounting['TK']['high_completion_user_airings']:,}")

    point_started = time.perf_counter()
    popularity = PopularityRecommender().fit(watched, candidates)
    pop_scores = popularity.counts.reindex(candidate_ids).to_numpy(dtype=np.float64)
    cold = set(users) - popularity.history_users
    records = []
    print(f"A-cold test users: {len(cold)}; all methods use the same forced distinct-A-viewer popularity fallback")

    def record(method, parameters, scores, point_started, *, positive_only=True, secondary=None, diagnostics=None, quiet=False):
        proposed = rank_scores(scores, candidate_ids, users, positive_only=positive_only, secondary_scores=secondary)
        predictions = protocols["T1"].finalize_predictions(proposed, popularity.ranking, popularity.history_users)
        if any(predictions[user] != popularity.ranking[:5] for user in cold):
            raise AssertionError("Method-specific cold-start fallback")
        metrics = {truth: protocol.score(predictions) for truth, protocol in protocols.items()}
        row = {
            "method": method, "configuration_id": f"config_{len(records):04d}", "hyperparameters": json_text(parameters),
            "selected": method not in ("score80", "hybrid_v4"), "seed": 42,
            "selection_map_TK": competition_truth_score(metrics), "runtime_seconds": time.perf_counter() - point_started,
            "diagnostics": json_text(diagnostics or {}),
        }
        for truth, value in metrics.items():
            row[f"map_at_5_{truth}"] = value
            for key in ("evaluated_users", "no_relevant_users", "median_relevant_all_users", "median_relevant_evaluated_users"):
                row[f"{key}_{truth}"] = accounting[truth][key]
        if method in previous.index:
            if any(abs(metrics[truth] - previous.loc[method, f"map_at_5_{truth}"]) > 1e-9 for truth in ("T1", "T2", "T3", "T4")):
                raise AssertionError(f"Existing method {method} no longer matches T1-T4")
        records.append(row)
        if not quiet:
            print(f"{method} {json_text(parameters)}: " + " ".join(f"{truth}={value:.9f}" for truth, value in metrics.items())
                  + f"; runtime={row['runtime_seconds']:.3f}s")
        return len(records) - 1

    print("\nFIXED METHODS (one prediction list per user, scored on T1-T4 and TK)")
    record("popularity", {}, np.broadcast_to(pop_scores, (len(users), len(candidates))), point_started)
    point_started = time.perf_counter()
    seconds_scores = RewatchRecommender().fit(matrix.seconds).score(matrix.test_rows, matrix.candidate_columns)
    record("rewatch", {}, seconds_scores, point_started)
    point_started = time.perf_counter()
    days_scores = RewatchDaysRecommender().fit(days, matrix.seconds).score(matrix.test_rows, matrix.candidate_columns)
    record("rewatch_days", {"tie_break": "seconds_then_id"}, days_scores, point_started, secondary=seconds_scores)
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
    record("CBF", {**configurations["CBF"], "metadata_scope": "A_only"}, cbf.score(matrix.test_rows, matrix.candidate_columns), point_started)
    del cbf, metadata
    point_started = time.perf_counter()
    association = AssociationRecommender(
        minimum_support=configurations["association"]["minimum_support"],
        minimum_lift=configurations["association"]["lift_strictly_greater_than"],
    ).fit(matrix.seconds)
    record("association", configurations["association"], association.score(matrix.test_rows, matrix.candidate_columns), point_started)
    del association
    point_started = time.perf_counter()
    slot = load_slot_profile("A", users, A_START, B_START)
    slot_scores = slot.score(users, candidate_ids, schedule, B_START, TEST_START)
    record("slot", {"slot": "channel_weekend_hour", "viewing": "all_fit_views"}, slot_scores, point_started, diagnostics=slot.diagnostics)
    del slot
    point_started = time.perf_counter()
    exposure = ExposureRecommender().fit(days, matrix.item_ids, schedule, A_START, B_START)
    exposure_scores = exposure.score(matrix.test_rows, matrix.candidate_columns, schedule, B_START, TEST_START)
    record("exposure", {"smoothing_days": 3}, exposure_scores, point_started)
    del exposure, schedule, watched, days
    point_started = time.perf_counter()
    parts = normalize_components(pop_scores, seconds_scores, ppr_scores, mf_scores)
    record("hybrid_old", configurations["hybrid"], fuse_scores(parts, tuple(configurations["hybrid"][name] for name in COMPONENTS)), point_started)
    del parts, seconds_scores
    point_started = time.perf_counter()
    parts = normalize_days_components(days_scores, ppr_scores, pop_scores, mf_scores)
    record("hybrid_v2", v2["weights"], fuse_days_scores(parts, v2["weights"]), point_started)
    del parts
    point_started = time.perf_counter()
    parts = normalize_habit_components(days_scores, exposure_scores, slot_scores, ppr_scores, mf_scores, pop_scores)
    record("hybrid_v3", v3["weights"], fuse_habit_scores(parts, v3["weights"]), point_started)
    del parts, days_scores, exposure_scores, mf_scores

    print("\nTK / KAGGLE CALIBRATION (Kaggle feedback supplied by user; public lifts rounded)")
    print(f"Popularity public Kaggle={KAGGLE_FEEDBACK['popularity']['public']:.5f}; validation TK={records[0]['map_at_5_TK']:.9f}")
    print("method         TK_MAP@5     TK_lift    Kaggle_public   Kaggle_lift")
    for method in ("hybrid_old", "hybrid_v2", "hybrid_v3"):
        row = next(row for row in records if row["method"] == method)
        row["kaggle_public"] = KAGGLE_FEEDBACK[method]["public"]
        row["kaggle_public_lift_reported"] = KAGGLE_PUBLIC_LIFTS[method]
        print(f"{method:<14} {row['map_at_5_TK']:.9f} {row['map_at_5_TK'] / records[0]['map_at_5_TK']:.6f}x "
              f"{KAGGLE_FEEDBACK[method]['public']:.5f}         {KAGGLE_PUBLIC_LIFTS[method]:.2f}x")

    point_started = time.perf_counter()
    freq_scores = completion.popularity_freq.reindex(candidate_ids, fill_value=0).to_numpy(dtype=np.float64)
    record("popularity_freq", {"score": "all_fit_users_completed_80_airings"},
           np.broadcast_to(freq_scores, (len(users), len(candidates))), point_started)
    prep = time.perf_counter()
    shared = {"slot": minmax_normalize(slot_scores), "ppr": minmax_normalize(ppr_scores)}
    maximum = float(freq_scores.max(initial=0))
    shared["pop"] = np.divide(freq_scores, maximum, out=np.zeros_like(freq_scores), where=maximum != 0)
    print(f"\nV4 shared normalization: {time.perf_counter() - prep:.3f}s; popularity_freq maximum={int(maximum):,}")
    del slot_scores, ppr_scores
    print("V4 GRID: 1,024 configurations (4 k values x 256 weight points); select TK MAP@5 only")
    print("score80 standalone k is also selected by TK; configuration ties retain the first ascending k/w50/w_slot/w_ppr/w_pop point.")
    best_index, best_parameters, best80_index = None, None, None
    for k in COMPLETION_K_VALUES:
        point_started = time.perf_counter()
        raw80, raw50 = completion.scores(users, k)
        index80 = record("score80", {"k": k}, raw80, point_started)
        if best80_index is None or records[index80]["selection_map_TK"] > records[best80_index]["selection_map_TK"]:
            best80_index = index80
        parts = {**shared, "score80": minmax_normalize(raw80), "score50": minmax_normalize(raw50)}
        del raw80, raw50
        for parameters in completion_weight_grid(k):
            point_started = time.perf_counter()
            index = record("hybrid_v4", parameters, fuse_completion_scores(parts, parameters), point_started, quiet=True)
            if best_index is None or records[index]["selection_map_TK"] > records[best_index]["selection_map_TK"]:
                best_index, best_parameters = index, parameters
        print(f"Completed k={k}: 256 grid points; best TK so far={records[best_index]['selection_map_TK']:.9f}")
        del parts
    records[best_index]["selected"] = True
    records[best80_index]["selected"] = True
    for row in records:
        for truth in protocols:
            row[f"lift_over_popularity_{truth}"] = row[f"map_at_5_{truth}"] / records[0][f"map_at_5_{truth}"]
    frame = pd.DataFrame(records)
    if len(frame) != 1041 or frame.method.eq("hybrid_v4").sum() != 1024 or frame.selected.sum() != 15:
        raise AssertionError("Missing fixed methods, standalone k values or completion grid points")
    output = ROOT / "results" / "validation_methods_v4.csv"
    frame.to_csv(output, index=False, float_format="%.9f")
    selected = records[best_index]
    manifest = {
        "seed": 42, "history_threshold_seconds": 300, "selection_rule": V4_SELECTION_RULE,
        "configuration": best_parameters, "score80_coefficient": 1,
        "component_configurations": {"PPR": configurations["PPR"]},
        "selection_map_TK": selected["selection_map_TK"],
        "validation_scores": {truth: selected[f"map_at_5_{truth}"] for truth in protocols},
        "best_standalone_score80_k": json.loads(records[best80_index]["hyperparameters"])["k"],
        "competition_ground_truth_quote": COMPETITION_GROUND_TRUTH_QUOTE,
        "truth_definitions": ALL_TRUTH_DEFINITIONS, "cohort_accounting": accounting,
        "screen_time_percentages": [80, 50], "completion_requires_300_seconds": False,
        "airing_count_scope": "unique channel/start airings intersecting each half-open window; full duration denominator",
        "target_schedule_fields": ["channel_id", "start_time", "duration", "tv_show_id"],
        "legacy_config_sha256": legacy_digest, "v3_selection_manifest_sha256": v3_digest,
        "validation_results_file": output.name, "validation_results_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "validation_candidate_count": len(candidates), "calibration_feedback": KAGGLE_FEEDBACK,
    }
    (ROOT / "results" / "hybrid_v4_selected.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    top = frame.loc[frame.method.eq("hybrid_v4")].sort_values("selection_map_TK", ascending=False, kind="stable").head(10)
    print("\nTOP 10 V4 CONFIGURATIONS")
    for row in top.itertuples(index=False):
        print(f"{row.hyperparameters}: T1={row.map_at_5_T1:.9f} T2={row.map_at_5_T2:.9f} T3={row.map_at_5_T3:.9f} "
              f"T4={row.map_at_5_T4:.9f} TK={row.map_at_5_TK:.9f}; runtime={row.runtime_seconds:.3f}s")
    print(f"Selected v4: {json_text(best_parameters)}; TK={selected['selection_map_TK']:.9f}")
    print(f"Best standalone score80: {records[best80_index]['hyperparameters']}; TK={records[best80_index]['selection_map_TK']:.9f}")
    print("\nSELECTED/FIXED COMPARISON")
    print("method          T1          T2          T3          T4          TK          runtime_s")
    for row in frame.loc[frame.selected].itertuples(index=False):
        print(f"{row.method:<15} {row.map_at_5_T1:.9f} {row.map_at_5_T2:.9f} {row.map_at_5_T3:.9f} "
              f"{row.map_at_5_T4:.9f} {row.map_at_5_TK:.9f} {row.runtime_seconds:.3f}")
    report.write("\nALL 1,041 CONFIGURATIONS\n" + frame.to_string(index=False, float_format=lambda value: f"{value:.9f}") + "\n\n")
    report.flush()
    print("Saved: results/validation_methods_v4.csv; results/validation_methods_v4.txt; results/hybrid_v4_selected.json")
    print(f"Total runtime: {time.perf_counter() - started:.3f} seconds")


if __name__ == "__main__":
    started = time.perf_counter()
    output = ROOT / "results" / "validation_methods_v4.txt"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as report, redirect_stdout(Tee(sys.stdout, report)):
        run(report, started)

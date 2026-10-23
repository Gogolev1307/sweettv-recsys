"""Compare fixed methods on T1-T4 and select a day-based hybrid on strict truths."""

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
from src.cbf import ContentRecommender
from src.data import (
    A_START, B_START, CACHE_DIR, DATA_DIR, ROOT, TEST_START, TRAIN_FILES, VIEW_FILE,
    airings_in_window, cached_result, load_programme_metadata, load_submission, load_training_schedule,
)
from src.experiment import load_calendar_join
from src.hybrid import (
    COMPONENTS, V2_SELECTION_RULE, days_weight_grid, fuse_days_scores, fuse_scores,
    normalize_components, normalize_days_components, strict_truth_mean,
)
from src.join import programme_day_counts, watched_interactions
from src.matrix import InteractionMatrix, aligned_day_matrix, rank_scores
from src.mf import ImplicitALS
from src.popularity import PopularityRecommender
from src.ppr import PersonalizedPageRank
from src.rewatch import RewatchDaysRecommender, RewatchRecommender
from src.submission import load_frozen_configurations
from src.truths import TRUTH_DEFINITIONS, validation_truths

SEED = 42


def json_text(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def run(report, started):
    random.seed(SEED)
    np.random.seed(SEED)
    source = ROOT / "results" / "validation_methods.csv"
    configurations, legacy_digest = load_frozen_configurations(source)
    previous = pd.read_csv(source).loc[lambda frame: frame.selected].set_index("method")
    print("sweet.tv strict-truth validation v2 (MAP@5)")
    print(f"Python {platform.python_version()} | numpy {np.__version__} | scipy {scipy.__version__} | pandas {pd.__version__}")
    try:
        import sklearn
        print(f"scikit-learn {sklearn.__version__} installed; TF-IDF uses NumPy/SciPy")
    except ImportError:
        print("scikit-learn not installed; TF-IDF uses NumPy/SciPy")
    print(f"Seed: {SEED}; BLAS/OpenMP threads: 1; fitting watch threshold: 300s")
    print(f"Fit A=[{A_START}, {B_START}); truth B=[{B_START}, {TEST_START})")
    print("Same 6,376 B-start candidates; re-watching allowed; five distinct ids; A-popularity fill and cold-start fallback.")
    print("T2/T4 sum all positive labelled B overlaps; T1/T3 require a >=300s user-airing total.")
    print("Days are actual viewing calendar dates within the window; a qualifying airing counts every date with positive viewing.")
    print("rewatch_days orders by days, then watched A seconds, then id; fused-score ties use ascending id.")
    print(f"Selection rule: {V2_SELECTION_RULE}.")
    print("Configuration ties retain the first grid point in ascending (w_ppr, w_pop, w_mf) order.")
    print(f"Frozen MF: {json_text(configurations['MF'])}")
    print(f"Frozen PPR: {json_text(configurations['PPR'])}")
    print(f"Frozen old hybrid: {json_text(configurations['hybrid'])}")

    input_started = time.perf_counter()
    schedule, _ = load_training_schedule()
    users = tuple(map(int, load_submission().user_id))
    candidates = frozenset(map(int, airings_in_window(schedule, B_START, TEST_START).tv_show_id)) - {0}
    joined_a = load_calendar_join("A", A_START, B_START)
    joined_b = load_calendar_join("B", B_START, TEST_START)
    protocols, accounting = validation_truths(users, candidates, joined_b)
    if len(users) != 1260 or len(candidates) != 6376 or accounting["T1"]["evaluated_users"] != 1246:
        raise AssertionError("Original validation cohort/candidates changed")
    watched_a = watched_interactions(joined_a.interactions)
    sources = [DATA_DIR / filename for filename in (VIEW_FILE, *TRAIN_FILES)]
    matrix = cached_result(
        CACHE_DIR, "method_matrix", sources + [DATA_DIR / "submission.csv"],
        {"threshold": 300, "matrix_version": 1},
        lambda: InteractionMatrix.build(watched_a, users, candidates),
    )
    candidate_ids = matrix.item_ids[matrix.candidate_columns]
    day_counts = programme_day_counts(joined_a)
    days = aligned_day_matrix(day_counts, matrix)
    if days.nnz != matrix.seconds.nnz or days.data.max() > (B_START - A_START).days:
        raise AssertionError("Fitting days do not match qualifying A programme pairs/window")
    metadata_a, metadata_available = load_programme_metadata()
    print(f"A matrix: {matrix.seconds.shape[0]:,} users x {matrix.seconds.shape[1]:,} programmes; "
          f"{matrix.seconds.nnz:,} observed pairs; distinct-day pairs={days.nnz:,}")
    print(f"Calendar-day attribution: A={len(joined_a.daily_interactions):,}, B={len(joined_b.daily_interactions):,} "
          "user/airing/date rows; daily seconds conserve original airing totals")
    print(f"Shared input preparation: {time.perf_counter() - input_started:.3f}s")
    print("\nTRUTH DEFINITIONS AND COHORTS")
    for name, stats in accounting.items():
        print(f"{name}: {TRUTH_DEFINITIONS[name]}")
        print(f"  1,260 = {stats['evaluated_users']:,} evaluated + {stats['no_relevant_users']:,} no relevant programme; "
              f"median relevant all/evaluated={stats['median_relevant_all_users']:g}/{stats['median_relevant_evaluated_users']:g}")
    del schedule, joined_a, joined_b, day_counts

    records = []
    point_started = time.perf_counter()
    popularity = PopularityRecommender().fit(watched_a, candidates)
    pop_scores = popularity.counts.reindex(candidate_ids).to_numpy(dtype=np.float64)
    cold = set(users) - popularity.history_users
    print(f"\nA-cold test users: {len(cold)}; identical fallback for every method and every truth")

    def record(method, parameters, scores, point_started, *, positive_only=True, secondary=None, diagnostics=None):
        proposed = rank_scores(scores, candidate_ids, users, positive_only=positive_only, secondary_scores=secondary)
        predictions = protocols["T1"].finalize_predictions(proposed, popularity.ranking, popularity.history_users)
        if any(predictions[user] != popularity.ranking[:5] for user in cold):
            raise AssertionError("Inconsistent cold-start recommendations")
        metrics = {name: protocol.score(predictions) for name, protocol in protocols.items()}
        strict_mean = strict_truth_mean(metrics)
        row = {
            "method": method, "configuration_id": f"config_{len(records):02d}",
            "hyperparameters": json_text(parameters), "selected": method != "hybrid_v2", "seed": SEED,
            "strict_mean_map_at_5": strict_mean, "runtime_seconds": time.perf_counter() - point_started,
            "diagnostics": json_text(diagnostics or {}),
        }
        for name, score in metrics.items():
            row[f"map_at_5_{name}"] = score
            for key, value in accounting[name].items():
                row[f"{key}_{name}"] = value
        records.append(row)
        original = "hybrid" if method == "hybrid_old" else method
        if original in previous.index and abs(metrics["T1"] - previous.loc[original, "map_at_5"]) > 1e-9:
            raise AssertionError(f"{method} no longer reproduces its original T1 score")
        print(f"{method} {json_text(parameters)}: "
              + " ".join(f"{name}={metrics[name]:.9f}" for name in protocols)
              + f" strict_mean={strict_mean:.9f}; runtime={row['runtime_seconds']:.3f}s")
        return len(records) - 1

    print("\nFIXED METHODS (identical predictions scored against all four truths)")
    record("popularity", configurations["popularity"], np.broadcast_to(pop_scores, (len(users), len(candidates))), point_started)
    point_started = time.perf_counter()
    seconds_scores = RewatchRecommender().fit(matrix.seconds).score(matrix.test_rows, matrix.candidate_columns)
    record("rewatch", configurations["rewatch"], seconds_scores, point_started)
    point_started = time.perf_counter()
    days_model = RewatchDaysRecommender().fit(days, matrix.seconds)
    days_scores = days_model.score(matrix.test_rows, matrix.candidate_columns)
    record("rewatch_days", {"score": "distinct_viewing_days", "tie_break": "seconds_then_id"},
           days_scores, point_started, secondary=seconds_scores)
    point_started = time.perf_counter()
    mf = ImplicitALS(**configurations["MF"]).fit(matrix.seconds)
    mf_scores = mf.score(matrix.test_rows, matrix.candidate_columns)
    record("MF", configurations["MF"], mf_scores, point_started, positive_only=False)
    del mf
    point_started = time.perf_counter()
    ppr_parameters = configurations["PPR"]
    ppr = PersonalizedPageRank(**{key: value for key, value in ppr_parameters.items() if key != "alpha"}).fit(matrix.seconds)
    ppr_scores, ppr_diagnostics = ppr.score(matrix.test_rows, matrix.candidate_columns, ppr_parameters["alpha"])
    record("PPR", ppr_parameters, ppr_scores, point_started, diagnostics=ppr_diagnostics)
    print(f"PPR: non-converged={ppr_diagnostics['non_converged_users']}; "
          f"max final residual={ppr_diagnostics['max_final_l1_residual']:.3e}; "
          f"max mass error={ppr_diagnostics['max_final_mass_error']:.3e}")
    if ppr_diagnostics["non_converged_users"]:
        raise AssertionError("PPR did not converge")
    del ppr
    point_started = time.perf_counter()
    cbf = ContentRecommender().fit(matrix.seconds, matrix.item_ids, metadata_a, metadata_available)
    record("CBF", configurations["CBF"], cbf.score(matrix.test_rows, matrix.candidate_columns), point_started)
    del cbf
    point_started = time.perf_counter()
    association = AssociationRecommender(
        minimum_support=configurations["association"]["minimum_support"],
        minimum_lift=configurations["association"]["lift_strictly_greater_than"],
    ).fit(matrix.seconds)
    record("association", configurations["association"], association.score(matrix.test_rows, matrix.candidate_columns),
           point_started, diagnostics={"rules_kept": association.rule_count})
    print(f"Association rules kept: {association.rule_count:,}")
    del association
    point_started = time.perf_counter()
    old_components = normalize_components(pop_scores, seconds_scores, ppr_scores, mf_scores)
    old_scores = fuse_scores(old_components, tuple(configurations["hybrid"][name] for name in COMPONENTS))
    record("hybrid_old", configurations["hybrid"], old_scores, point_started)
    del old_components, old_scores, seconds_scores, watched_a

    print("\nDAY-BASED HYBRID GRID (24 configurations; fixed rewatch_days coefficient=1)")
    normalization_started = time.perf_counter()
    components = normalize_days_components(days_scores, ppr_scores, pop_scores, mf_scores)
    normalization_seconds = time.perf_counter() - normalization_started
    del days_scores, ppr_scores, mf_scores, days_model
    best_index, best_weights = None, None
    for weights in days_weight_grid():
        point_started = time.perf_counter()
        index = record("hybrid_v2", weights, fuse_days_scores(components, weights), point_started - normalization_seconds)
        if best_index is None or records[index]["strict_mean_map_at_5"] > records[best_index]["strict_mean_map_at_5"]:
            best_index, best_weights = index, weights
    records[best_index]["selected"] = True
    del components
    frame = pd.DataFrame(records)
    if len(frame) != 32 or frame.selected.sum() != 9:
        raise AssertionError("Missing fixed methods/grid configurations")
    for name in protocols:
        if not frame[f"evaluated_users_{name}"].eq(accounting[name]["evaluated_users"]).all():
            raise AssertionError("Methods used different users for the same truth")
    output = ROOT / "results" / "validation_methods_v2.csv"
    frame.to_csv(output, index=False, float_format="%.9f")
    manifest = {
        "seed": SEED, "threshold_seconds": 300, "selection_rule": V2_SELECTION_RULE,
        "configuration_tie_break": "first ascending (w_ppr, w_pop, w_mf) grid point",
        "weights": best_weights, "component_configurations": {name: configurations[name] for name in ("MF", "PPR")},
        "truth_definitions": TRUTH_DEFINITIONS, "cohort_accounting": accounting,
        "validation_scores": {name: records[best_index][f"map_at_5_{name}"] for name in protocols},
        "strict_mean_map_at_5": records[best_index]["strict_mean_map_at_5"],
        "legacy_config_sha256": legacy_digest, "validation_results_file": output.name,
        "validation_results_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "validation_candidate_count": len(candidates),
        "day_definition": "actual viewing dates with positive overlap on an airing qualifying within the fitting window",
    }
    (ROOT / "results" / "hybrid_v2_selected.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"\nSelected v2 weights: {json_text(best_weights)}; strict mean={records[best_index]['strict_mean_map_at_5']:.9f}")
    print("SELECTED/FIXED COMPARISON")
    print("method          T1          T2          T3          T4          strict_mean")
    for row in frame.loc[frame.selected].itertuples(index=False):
        print(f"{row.method:<14} {row.map_at_5_T1:.9f} {row.map_at_5_T2:.9f} {row.map_at_5_T3:.9f} "
              f"{row.map_at_5_T4:.9f} {row.strict_mean_map_at_5:.9f}")
    pop = frame.loc[frame.method.eq("popularity")].iloc[0]
    for method in ("hybrid_old", "hybrid_v2"):
        row = frame.loc[frame.method.eq(method) & frame.selected].iloc[0]
        print(f"{method}/popularity lift: " + " ".join(
            f"{name}={row[f'map_at_5_{name}'] / pop[f'map_at_5_{name}']:.6f}x" for name in protocols
        ))
    report.write("\nALL 32 CONFIGURATIONS\n" + frame.to_string(index=False, float_format=lambda value: f"{value:.9f}") + "\n\n")
    report.flush()
    print("Saved: results/validation_methods_v2.csv; results/validation_methods_v2.txt; results/hybrid_v2_selected.json")
    print(f"Total runtime: {time.perf_counter() - started:.3f} seconds")


if __name__ == "__main__":
    started = time.perf_counter()
    output = ROOT / "results" / "validation_methods_v2.txt"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as report, redirect_stdout(Tee(sys.stdout, report)):
        run(report, started)

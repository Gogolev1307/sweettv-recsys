"""Refit frozen selected methods on weeks 11-30 and validate final Kaggle files."""

from __future__ import annotations

import os

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[variable] = "1"

import argparse
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
from src.completion import CompletionRecommender
from src.data import (
    A_START, CACHE_DIR, DATA_DIR, ROOT, TEST_END, TEST_FILE, TEST_START, TRAIN_FILES, VIEW_FILE,
    cached_result, load_all_training_metadata, load_submission, load_test_candidates,
    load_training_schedule, load_views,
)
from src.evaluate import ValidationProtocol
from src.experiment import load_calendar_join, load_final_target_airings, load_slot_profile
from src.exposure import ExposureRecommender
from src.hybrid import (
    COMPONENTS, fuse_completion_scores, fuse_days_scores, fuse_habit_scores, fuse_scores, minmax_normalize,
    normalize_components, normalize_days_components, normalize_habit_components,
)
from src.join import join_views_to_airings, programme_day_counts, watched_interactions
from src.matrix import InteractionMatrix, aligned_day_matrix, rank_scores
from src.mf import ImplicitALS
from src.popularity import PopularityRecommender
from src.ppr import PersonalizedPageRank
from src.rewatch import RewatchRecommender
from src.submission import (
    load_frozen_configurations, load_frozen_v2_configuration, load_frozen_v3_configuration,
    load_frozen_v4_configuration, snapshot_submissions,
    validate_submission, verify_preserved_submissions, write_or_preserve_submission,
)

SEED = 42
THRESHOLD = 300


def json_text(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def run(started):
    random.seed(SEED)
    np.random.seed(SEED)
    selected_path = ROOT / "results" / "validation_methods.csv"
    configurations, validation_digest = load_frozen_configurations(selected_path)
    print("sweet.tv final submissions - frozen validation configurations")
    print(f"Python {platform.python_version()} | numpy {np.__version__} | scipy {scipy.__version__} | pandas {pd.__version__}")
    print(f"Seed: {SEED}; watch threshold: {THRESHOLD}s; BLAS/OpenMP threads: 1")
    print(f"Configuration source: results/validation_methods.csv; SHA256={validation_digest}")
    print("Frozen configurations are read from selected rows; no tuning or validation scoring is performed.")
    for method in configurations:
        print(f"Frozen {method}: {json_text(configurations[method])}")
    print(f"Fit all training: [{A_START}, {TEST_START}); combine airing overlaps before the 300s threshold.")
    print(f"Test schedule: {TEST_FILE}; read column tv_show_id only, for candidate availability.")

    preparation_started = time.perf_counter()
    template = load_submission()
    test_users = tuple(map(int, template.user_id))
    candidates = load_test_candidates()
    if len(test_users) != 1260 or len(candidates) < 5:
        raise AssertionError("Incorrect final cohort or candidate pool")
    schedule, _ = load_training_schedule()
    views = load_views()
    sources = [DATA_DIR / name for name in (VIEW_FILE, *TRAIN_FILES)]
    # Join the FULL fitting interval; thresholding A and B separately would lose
    # airings with subthreshold pieces whose combined overlap reaches 300s.
    joined = cached_result(
        CACHE_DIR, "join_all_training", sources,
        {"window_start": str(A_START), "window_end": str(TEST_START), "join_version": 1},
        lambda: join_views_to_airings(views, schedule, A_START, TEST_START),
    )
    watched = watched_interactions(joined.interactions, THRESHOLD)
    matrix = cached_result(
        CACHE_DIR, "final_method_matrix", sources + [DATA_DIR / TEST_FILE, DATA_DIR / "submission.csv"],
        {"threshold": THRESHOLD, "matrix_version": 1, "fit_end": str(TEST_START)},
        lambda: InteractionMatrix.build(watched, test_users, candidates),
    )
    candidate_ids = matrix.item_ids[matrix.candidate_columns]
    if not np.array_equal(candidate_ids, candidates):
        raise AssertionError("Test candidate ids and score columns are misaligned")
    metadata = load_all_training_metadata()
    if not (metadata.start_time < TEST_START).all():
        raise AssertionError("Future metadata entered the final fitting snapshot")
    protocol = ValidationProtocol(test_users, frozenset(map(int, candidates)), {})
    print(f"Training watched airings: {len(watched):,}; users: {watched.user_id.nunique():,}; "
          f"programmes: {watched.tv_show_id.nunique():,}")
    print(f"Matrix: {matrix.seconds.shape[0]:,} users x {matrix.seconds.shape[1]:,} programmes; "
          f"{matrix.seconds.nnz:,} observed user-programme pairs")
    print(f"Test users: {len(test_users):,}; labelled test candidates: {len(candidates):,}; "
          f"training-only metadata snapshots: {len(metadata):,}")
    print(f"Shared input preparation: {time.perf_counter() - preparation_started:.3f}s")
    del views, schedule, joined

    selected_predictions = {}
    point_started = time.perf_counter()
    popularity = PopularityRecommender().fit(watched, candidates)
    cold_users = set(test_users) - popularity.history_users
    print(f"Cold-start test users (no usable weeks 11-30 history): {len(cold_users):,}")
    popularity_scores = popularity.counts.reindex(candidate_ids).to_numpy(dtype=np.float64)

    def finish(method, scores, point_started, *, positive_only=True):
        proposed = rank_scores(scores, candidate_ids, test_users, positive_only=positive_only)
        predictions = protocol.finalize_predictions(proposed, popularity.ranking, popularity.history_users)
        for user, items in predictions.items():
            if len(items) != 5 or len(set(items)) != 5 or not set(items).issubset(protocol.candidates):
                raise AssertionError(f"Invalid final recommendations for {method}/{user}")
            if user in cold_users and items != popularity.ranking[:5]:
                raise AssertionError("Final cold-start fallback differs across methods")
        selected_predictions[method] = predictions
        print(f"Refit {method}: {len(predictions):,} users; exactly five distinct candidate ids each; "
              f"cold-start fallback={len(cold_users):,}; runtime={time.perf_counter() - point_started:.3f}s")

    if configurations["popularity"] or configurations["rewatch"]:
        raise ValueError("Unexpected reference-baseline hyperparameters")
    finish("popularity", np.broadcast_to(popularity_scores, (len(test_users), len(candidates))), point_started)
    del watched
    point_started = time.perf_counter()
    rewatch_scores = RewatchRecommender().fit(matrix.seconds).score(matrix.test_rows, matrix.candidate_columns)
    finish("rewatch", rewatch_scores, point_started)
    point_started = time.perf_counter()
    mf = ImplicitALS(**configurations["MF"]).fit(matrix.seconds)
    mf_scores = mf.score(matrix.test_rows, matrix.candidate_columns)
    finish("MF", mf_scores, point_started, positive_only=False)
    del mf
    point_started = time.perf_counter()
    ppr_parameters = configurations["PPR"]
    ppr = PersonalizedPageRank(**{key: value for key, value in ppr_parameters.items() if key != "alpha"}).fit(matrix.seconds)
    ppr_scores, ppr_diagnostics = ppr.score(matrix.test_rows, matrix.candidate_columns, ppr_parameters["alpha"])
    finish("PPR", ppr_scores, point_started)
    print(f"PPR convergence: mean/max iterations={ppr_diagnostics['mean_iterations']:.3f}/{ppr_diagnostics['max_iterations']}; "
          f"max final L1 residual={ppr_diagnostics['max_final_l1_residual']:.3e}; "
          f"non-converged={ppr_diagnostics['non_converged_users']}; max mass error={ppr_diagnostics['max_final_mass_error']:.3e}")
    if ppr_diagnostics["non_converged_users"]:
        raise AssertionError("Final PPR did not converge")
    del ppr

    point_started = time.perf_counter()
    if configurations["CBF"] != {
        "idf": "smooth_A_only", "profile_weight": "log1p(seconds/60)",
        "tf": "binary", "year": "first_year_decade",
    }:
        raise ValueError("Unsupported frozen content configuration")
    # 'A_only' records the validation fit scope. For the final refit the SAME
    # smoothed-IDF algorithm is fitted on the expanded weeks 11-30 scope.
    cbf = ContentRecommender().fit(matrix.seconds, matrix.item_ids, metadata, metadata)
    scores = cbf.score(matrix.test_rows, matrix.candidate_columns)
    finish("CBF", scores, point_started)
    print(f"CBF refit vocabulary/IDF: weeks 11-30 metadata only; features={len(cbf.vocabulary):,}; "
          f"zero-vector programmes={cbf.zero_vector_items:,}; test metadata fields read=0")
    del cbf, scores
    point_started = time.perf_counter()
    association_parameters = configurations["association"]
    if association_parameters["score"] != "max_confidence":
        raise ValueError("Unsupported frozen association scoring rule")
    association = AssociationRecommender(
        minimum_support=association_parameters["minimum_support"],
        minimum_lift=association_parameters["lift_strictly_greater_than"],
    ).fit(matrix.seconds)
    scores = association.score(matrix.test_rows, matrix.candidate_columns)
    finish("association", scores, point_started)
    print(f"Association rules kept after final refit: {association.rule_count:,}")
    del association, scores
    point_started = time.perf_counter()
    hybrid_parameters = configurations["hybrid"]
    if set(hybrid_parameters) != set(COMPONENTS):
        raise ValueError("Frozen hybrid component names differ from validation")
    components = normalize_components(popularity_scores, rewatch_scores, ppr_scores, mf_scores)
    scores = fuse_scores(components, tuple(hybrid_parameters[name] for name in COMPONENTS))
    finish("hybrid", scores, point_started)
    del components, scores, rewatch_scores, ppr_scores, mf_scores

    manifest = {
        "seed": SEED, "threshold_seconds": THRESHOLD, "fit_start": str(A_START), "fit_end": str(TEST_START),
        "validation_config_source": "results/validation_methods.csv", "validation_config_sha256": validation_digest,
        "configurations": configurations, "test_candidate_count": len(candidates),
        "test_schedule_columns_read": ["tv_show_id"],
    }
    (ROOT / "results" / "submission_frozen_config.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    output_dir = ROOT / "submissions"
    output_dir.mkdir(parents=True, exist_ok=True)
    print("\nWRITTEN-FILE CHECKS")
    for method, filename in (
        ("popularity", "submission_popularity.csv"), ("PPR", "submission_ppr.csv"),
        ("hybrid", "submission_hybrid.csv"), ("MF", "submission_mf.csv"),
    ):
        file_started = time.perf_counter()
        output = output_dir / filename
        created = write_or_preserve_submission(output, test_users, selected_predictions[method])
        checks = validate_submission(output, test_users, candidates, cold_users, popularity.ranking[:5])
        print(f"submissions/{filename}: rows={checks['rows']:,}; header=PASS; template user order=PASS; "
              "plain integer user_id=PASS; five space-separated distinct ids=PASS; test candidate membership=PASS; "
              f"no id=0=PASS; cold-start fallback users={checks['cold_start_fallback_users']:,}; "
              f"write/check runtime={time.perf_counter() - file_started:.3f}s; {'created' if created else 'existing file preserved'}")
    print("Saved checks: results/submission_checks.txt")
    print("Saved frozen configurations: results/submission_frozen_config.json")
    print(f"Total runtime: {time.perf_counter() - started:.3f} seconds")


def run_v2(started):
    random.seed(SEED)
    np.random.seed(SEED)
    output_dir = ROOT / "submissions"
    output_dir.mkdir(parents=True, exist_ok=True)
    name = "submission_hybrid_v2.csv"
    preserved = snapshot_submissions(output_dir, name)
    manifest, digest = load_frozen_v2_configuration(ROOT / "results" / "hybrid_v2_selected.json")
    components_config = manifest["component_configurations"]
    weights = manifest["weights"]
    print("sweet.tv final hybrid-v2 submission - frozen strict-truth selection")
    print(f"Python {platform.python_version()} | numpy {np.__version__} | scipy {scipy.__version__} | pandas {pd.__version__}")
    print(f"Seed: {SEED}; watch threshold: {THRESHOLD}s; BLAS/OpenMP threads: 1")
    print(f"Frozen selection rule: {manifest['selection_rule']}")
    print(f"Frozen v2 weights: {json_text(weights)}")
    print(f"Frozen MF: {json_text(components_config['MF'])}")
    print(f"Frozen PPR: {json_text(components_config['PPR'])}")
    print(f"Selection manifest SHA256: {digest}")
    print(f"Refit [{A_START}, {TEST_START}); count actual viewing dates over all weeks 11-30.")
    print("Test schedule supplies tv_show_id only; complete candidate-pool min-max normalization; no retuning.")
    input_started = time.perf_counter()
    users = tuple(map(int, load_submission().user_id))
    candidates = load_test_candidates()
    joined = load_calendar_join("all_training", A_START, TEST_START)
    watched = watched_interactions(joined.interactions)
    sources = [DATA_DIR / filename for filename in (VIEW_FILE, *TRAIN_FILES)]
    matrix = cached_result(
        CACHE_DIR, "final_method_matrix", sources + [DATA_DIR / TEST_FILE, DATA_DIR / "submission.csv"],
        {"threshold": THRESHOLD, "matrix_version": 1, "fit_end": str(TEST_START)},
        lambda: InteractionMatrix.build(watched, users, candidates),
    )
    candidate_ids = matrix.item_ids[matrix.candidate_columns]
    if len(users) != 1260 or not np.array_equal(candidate_ids, candidates):
        raise AssertionError("Final user/candidate id alignment failed")
    days = aligned_day_matrix(programme_day_counts(joined), matrix)
    if days.nnz != matrix.seconds.nnz or days.data.max() > (TEST_START - A_START).days:
        raise AssertionError("Full-training day counts differ from the watched programme pairs/window")
    popularity = PopularityRecommender().fit(watched, candidates)
    cold = set(users) - popularity.history_users
    print(f"Full-training watched airings={len(watched):,}; user-programme pairs={matrix.seconds.nnz:,}; "
          f"day-count pairs={days.nnz:,}; max distinct days={int(days.data.max())}")
    print(f"Test users={len(users):,}; labelled test candidates={len(candidates):,}; cold-start fallback users={len(cold)}")
    print(f"Shared input preparation: {time.perf_counter() - input_started:.3f}s")
    del joined, watched
    point_started = time.perf_counter()
    days_scores = days[matrix.test_rows][:, matrix.candidate_columns].toarray()
    pop_scores = popularity.counts.reindex(candidate_ids).to_numpy(dtype=np.float64)
    print(f"Refit day/popularity components: {time.perf_counter() - point_started:.3f}s")
    point_started = time.perf_counter()
    mf = ImplicitALS(**components_config["MF"]).fit(matrix.seconds)
    mf_scores = mf.score(matrix.test_rows, matrix.candidate_columns)
    print(f"Refit MF: {time.perf_counter() - point_started:.3f}s")
    del mf
    point_started = time.perf_counter()
    ppr_parameters = components_config["PPR"]
    ppr = PersonalizedPageRank(**{key: value for key, value in ppr_parameters.items() if key != "alpha"}).fit(matrix.seconds)
    ppr_scores, diagnostics = ppr.score(matrix.test_rows, matrix.candidate_columns, ppr_parameters["alpha"])
    print(f"Refit PPR: {time.perf_counter() - point_started:.3f}s; mean/max iterations="
          f"{diagnostics['mean_iterations']:.3f}/{diagnostics['max_iterations']}; "
          f"max residual={diagnostics['max_final_l1_residual']:.3e}; non-converged={diagnostics['non_converged_users']}")
    if diagnostics["non_converged_users"]:
        raise AssertionError("Final v2 PPR did not converge")
    del ppr
    point_started = time.perf_counter()
    scores = fuse_days_scores(normalize_days_components(days_scores, ppr_scores, pop_scores, mf_scores), weights)
    protocol = ValidationProtocol(users, frozenset(map(int, candidates)), {})
    predictions = protocol.finalize_predictions(
        rank_scores(scores, candidate_ids, users), popularity.ranking, popularity.history_users,
    )
    output = output_dir / name
    created = write_or_preserve_submission(output, users, predictions)
    checks = validate_submission(output, users, candidates, cold, popularity.ranking[:5])
    print(f"Hybrid fusion/ranking/write/check: {time.perf_counter() - point_started:.3f}s")
    print(f"submissions/{name}: rows={checks['rows']:,}; header=PASS; template user order=PASS; "
          "plain integer user_id=PASS; five space-separated distinct ids=PASS; test candidate membership=PASS; "
          f"no id=0=PASS; cold-start fallback users={checks['cold_start_fallback_users']}; "
          f"{'created' if created else 'existing v2 file preserved'}")
    verify_preserved_submissions(output_dir, preserved)
    for old_name in preserved:
        print(f"Preserved {old_name}: SHA256 and modification time unchanged")
    final_manifest = {
        "seed": SEED, "threshold_seconds": THRESHOLD, "fit_start": str(A_START), "fit_end": str(TEST_START),
        "validation_selection_manifest_sha256": digest, "weights": weights,
        "component_configurations": components_config, "test_candidate_count": len(candidates),
        "preserved_submissions": preserved, "test_schedule_columns_read": ["tv_show_id"],
    }
    (ROOT / "results" / "submission_hybrid_v2_frozen.json").write_text(json.dumps(final_manifest, indent=2, sort_keys=True) + "\n")
    print("Saved: submissions/submission_hybrid_v2.csv; results/submission_checks_v2.txt; results/submission_hybrid_v2_frozen.json")
    print(f"Total runtime: {time.perf_counter() - started:.3f} seconds")


def run_v3(started):
    random.seed(SEED)
    np.random.seed(SEED)
    output_dir = ROOT / "submissions"
    output_dir.mkdir(parents=True, exist_ok=True)
    name = "submission_hybrid_v3.csv"
    preserved = snapshot_submissions(output_dir, name)
    manifest, digest = load_frozen_v3_configuration(ROOT / "results" / "hybrid_v3_selected.json")
    weights, parameters = manifest["weights"], manifest["component_configurations"]
    print("sweet.tv final hybrid-v3 submission - calibrated frozen habits")
    print(f"Python {platform.python_version()} | numpy {np.__version__} | scipy {scipy.__version__} | pandas {pd.__version__}")
    print("Seed: 42; BLAS/OpenMP threads: 1; labelled-airing watch threshold: 300s")
    print(f"Frozen selection rule: {manifest['selection_rule']}")
    print(manifest["calibration"]["statement"])
    print(f"Frozen v3 weights: {json_text(weights)}")
    print(f"Frozen MF: {json_text(parameters['MF'])}; frozen PPR: {json_text(parameters['PPR'])}")
    print(f"Selection manifest SHA256: {digest}")
    print(f"Fit [{A_START}, {TEST_START}); target [{TEST_START}, {TEST_END})")
    print("Target schedule fields: channel_id, start_time, duration, tv_show_id only; known export-boundary carry-in included.")
    prep = time.perf_counter()
    users = tuple(map(int, load_submission().user_id))
    candidates = load_test_candidates()
    joined = load_calendar_join("all_training", A_START, TEST_START)
    watched = watched_interactions(joined.interactions)
    matrix = cached_result(
        CACHE_DIR, "final_method_matrix", [DATA_DIR / filename for filename in
                                          (VIEW_FILE, *TRAIN_FILES, TEST_FILE, "submission.csv")],
        {"threshold": THRESHOLD, "matrix_version": 1, "fit_end": str(TEST_START)},
        lambda: InteractionMatrix.build(watched, users, candidates),
    )
    candidate_ids = matrix.item_ids[matrix.candidate_columns]
    if len(users) != 1260 or not np.array_equal(candidate_ids, candidates):
        raise AssertionError("Final candidate/user index alignment failed")
    days = aligned_day_matrix(programme_day_counts(joined), matrix)
    fit_schedule, _ = load_training_schedule()
    target_schedule = load_final_target_airings()
    popularity = PopularityRecommender().fit(watched, candidates)
    cold = set(users) - popularity.history_users
    print(f"Training watched airings={len(watched):,}; observed user-programme/day pairs={matrix.seconds.nnz:,}/{days.nnz:,}")
    print(f"Test users={len(users):,}; labelled candidates={len(candidates):,}; cold-start fallback users={len(cold)}")
    print(f"Shared input preparation: {time.perf_counter() - prep:.3f}s")
    del joined, watched
    day_scores = days[matrix.test_rows][:, matrix.candidate_columns].toarray()
    pop_scores = popularity.counts.reindex(candidate_ids).to_numpy(dtype=np.float64)
    point_started = time.perf_counter()
    slot = load_slot_profile("all_training", users, A_START, TEST_START)
    slot_scores = slot.score(users, candidate_ids, target_schedule, TEST_START, TEST_END)
    print(f"Slot: fit views={slot.diagnostics['view_intervals']:,}; all-view seconds={slot.diagnostics['view_seconds']:,}; "
          f"max row-sum error={slot.diagnostics['max_row_sum_error']:.3e}; runtime={time.perf_counter() - point_started:.3f}s")
    del slot
    point_started = time.perf_counter()
    exposure = ExposureRecommender(smoothing_days=manifest["exposure_smoothing_days"]).fit(
        days, matrix.item_ids, fit_schedule, A_START, TEST_START,
    )
    exposure_scores = exposure.score(matrix.test_rows, matrix.candidate_columns, target_schedule, TEST_START, TEST_END)
    print(f"Exposure: max scheduled fit/target days={exposure.diagnostics['max_fit_airing_days']}/"
          f"{exposure.diagnostics['max_target_airing_days']}; smoothing=3; runtime={time.perf_counter() - point_started:.3f}s")
    del exposure, fit_schedule, target_schedule
    point_started = time.perf_counter()
    mf = ImplicitALS(**parameters["MF"]).fit(matrix.seconds)
    mf_scores = mf.score(matrix.test_rows, matrix.candidate_columns)
    print(f"Refit MF: {time.perf_counter() - point_started:.3f}s")
    del mf
    point_started = time.perf_counter()
    ppr_parameters = parameters["PPR"]
    ppr = PersonalizedPageRank(**{key: value for key, value in ppr_parameters.items() if key != "alpha"}).fit(matrix.seconds)
    ppr_scores, diagnostics = ppr.score(matrix.test_rows, matrix.candidate_columns, ppr_parameters["alpha"])
    print(f"Refit PPR: {time.perf_counter() - point_started:.3f}s; iterations mean/max="
          f"{diagnostics['mean_iterations']:.3f}/{diagnostics['max_iterations']}; "
          f"max residual={diagnostics['max_final_l1_residual']:.3e}; non-converged={diagnostics['non_converged_users']}")
    if diagnostics["non_converged_users"]:
        raise AssertionError("V3 final PPR did not converge")
    del ppr
    point_started = time.perf_counter()
    scores = fuse_habit_scores(
        normalize_habit_components(day_scores, exposure_scores, slot_scores, ppr_scores, mf_scores, pop_scores), weights,
    )
    protocol = ValidationProtocol(users, frozenset(map(int, candidates)), {})
    predictions = protocol.finalize_predictions(
        rank_scores(scores, candidate_ids, users), popularity.ranking, popularity.history_users,
    )
    output = output_dir / name
    created = write_or_preserve_submission(output, users, predictions)
    checks = validate_submission(output, users, candidates, cold, popularity.ranking[:5])
    print(f"Fusion/ranking/write/check: {time.perf_counter() - point_started:.3f}s")
    print(f"submissions/{name}: rows={checks['rows']:,}; header=PASS; template user order=PASS; "
          "plain integer user_id=PASS; five space-separated distinct ids=PASS; test candidate membership=PASS; "
          f"no id=0=PASS; cold-start fallback users={checks['cold_start_fallback_users']}; "
          f"{'created' if created else 'existing v3 file preserved'}")
    verify_preserved_submissions(output_dir, preserved)
    for old_name in preserved:
        print(f"Preserved {old_name}: SHA256 and modification time unchanged")
    final_manifest = {
        "seed": SEED, "threshold_seconds": THRESHOLD, "fit_start": str(A_START), "fit_end": str(TEST_START),
        "target_start": str(TEST_START), "target_end": str(TEST_END), "weights": weights,
        "component_configurations": parameters, "test_candidate_count": len(candidates),
        "validation_selection_manifest_sha256": digest, "preserved_submissions": preserved,
        "target_schedule_fields": manifest["target_schedule_fields"],
    }
    (ROOT / "results" / "submission_hybrid_v3_frozen.json").write_text(json.dumps(final_manifest, indent=2, sort_keys=True) + "\n")
    print("Saved: submissions/submission_hybrid_v3.csv; results/submission_checks_v3.txt; results/submission_hybrid_v3_frozen.json")
    print(f"Total runtime: {time.perf_counter() - started:.3f} seconds")


def run_v4(started):
    random.seed(SEED)
    np.random.seed(SEED)
    output_dir = ROOT / "submissions"
    output_dir.mkdir(parents=True, exist_ok=True)
    name = "submission_hybrid_v4.csv"
    preserved = snapshot_submissions(output_dir, name)
    manifest, digest = load_frozen_v4_configuration(ROOT / "results" / "hybrid_v4_selected.json")
    configuration = manifest["configuration"]
    ppr_parameters = manifest["component_configurations"]["PPR"]
    print("sweet.tv final hybrid-v4 submission - frozen published-rule completion")
    print(f"Python {platform.python_version()} | numpy {np.__version__} | scipy {scipy.__version__} | pandas {pd.__version__}")
    print("Seed: 42; BLAS/OpenMP threads: 1; shared labelled-history threshold: 300s")
    print(f"Frozen selection rule: {manifest['selection_rule']}")
    print(f"Frozen v4 configuration: {json_text(configuration)}; score80 coefficient=1")
    print(f"Frozen PPR: {json_text(ppr_parameters)}")
    print(f"Selection manifest SHA256: {digest}")
    print(f"Fit [{A_START}, {TEST_START}); target [{TEST_START}, {TEST_END})")
    print("Completion counts use full-duration >=80%/>=50% airings, with no 300s cutoff; nA/nT are unique intersecting channel/start airings.")
    print("Target schedule fields: channel_id, start_time, duration, tv_show_id only; slot uses all fitting views.")
    prep = time.perf_counter()
    users = tuple(map(int, load_submission().user_id))
    candidates = load_test_candidates()
    joined = load_calendar_join("all_training", A_START, TEST_START)
    watched = watched_interactions(joined.interactions)
    matrix = cached_result(
        CACHE_DIR, "final_method_matrix", [DATA_DIR / filename for filename in
                                          (VIEW_FILE, *TRAIN_FILES, TEST_FILE, "submission.csv")],
        {"threshold": THRESHOLD, "matrix_version": 1, "fit_end": str(TEST_START)},
        lambda: InteractionMatrix.build(watched, users, candidates),
    )
    candidate_ids = matrix.item_ids[matrix.candidate_columns]
    if len(users) != 1260 or not np.array_equal(candidate_ids, candidates):
        raise AssertionError("Final v4 user/candidate index alignment failed")
    fit_schedule, _ = load_training_schedule()
    target_schedule = load_final_target_airings()
    completion = CompletionRecommender().fit(joined.interactions, users, candidate_ids, fit_schedule, A_START, TEST_START)
    completion.prepare_target(target_schedule, TEST_START, TEST_END)
    popularity = PopularityRecommender().fit(watched, candidates)
    cold = set(users) - popularity.history_users
    print(f"Training watched >=300s airings={len(watched):,}; PPR user-programme pairs={matrix.seconds.nnz:,}")
    print(f"Completion user-airings: >=80%={completion.diagnostics['completed_80_user_airings']:,}; "
          f">=50%={completion.diagnostics['completed_50_user_airings']:,}; "
          f"programme/channel pairs={completion.diagnostics['fit_candidate_programme_channels']:,}")
    print(f"Test users={len(users):,}; labelled candidates={len(candidates):,}; cold-start fallback users={len(cold)}")
    print(f"Shared input/count preparation: {time.perf_counter() - prep:.3f}s")
    del joined, watched, fit_schedule
    point_started = time.perf_counter()
    score80, score50 = completion.scores(users, configuration["k"])
    frequency = completion.popularity_freq.reindex(candidate_ids, fill_value=0).to_numpy(dtype=np.float64)
    maximum = float(frequency.max(initial=0))
    pop_frequency = np.divide(frequency, maximum, out=np.zeros_like(frequency), where=maximum != 0)
    print(f"Completion scoring: k={configuration['k']}; popularity_freq maximum={int(maximum):,}; "
          f"runtime={time.perf_counter() - point_started:.3f}s")
    del completion
    point_started = time.perf_counter()
    slot = load_slot_profile("all_training", users, A_START, TEST_START)
    slot_scores = slot.score(users, candidate_ids, target_schedule, TEST_START, TEST_END)
    print(f"Slot: all-view seconds={slot.diagnostics['view_seconds']:,}; max row-sum error="
          f"{slot.diagnostics['max_row_sum_error']:.3e}; runtime={time.perf_counter() - point_started:.3f}s")
    del slot, target_schedule
    point_started = time.perf_counter()
    ppr = PersonalizedPageRank(**{key: value for key, value in ppr_parameters.items() if key != "alpha"}).fit(matrix.seconds)
    ppr_scores, diagnostics = ppr.score(matrix.test_rows, matrix.candidate_columns, ppr_parameters["alpha"])
    print(f"Refit PPR: {time.perf_counter() - point_started:.3f}s; iterations mean/max="
          f"{diagnostics['mean_iterations']:.3f}/{diagnostics['max_iterations']}; "
          f"max residual={diagnostics['max_final_l1_residual']:.3e}; non-converged={diagnostics['non_converged_users']}")
    if diagnostics["non_converged_users"]:
        raise AssertionError("Final v4 PPR did not converge")
    del ppr
    point_started = time.perf_counter()
    components = {
        "score80": minmax_normalize(score80), "score50": minmax_normalize(score50),
        "slot": minmax_normalize(slot_scores), "ppr": minmax_normalize(ppr_scores), "pop": pop_frequency,
    }
    scores = fuse_completion_scores(components, configuration)
    protocol = ValidationProtocol(users, frozenset(map(int, candidates)), {})
    predictions = protocol.finalize_predictions(
        rank_scores(scores, candidate_ids, users), popularity.ranking, popularity.history_users,
    )
    output = output_dir / name
    created = write_or_preserve_submission(output, users, predictions)
    checks = validate_submission(output, users, candidates, cold, popularity.ranking[:5])
    print(f"Fusion/ranking/write/check: {time.perf_counter() - point_started:.3f}s")
    print(f"submissions/{name}: rows={checks['rows']:,}; header=PASS; template user order=PASS; "
          "plain integer user_id=PASS; five space-separated distinct ids=PASS; test candidate membership=PASS; "
          f"no id=0=PASS; cold-start fallback users={checks['cold_start_fallback_users']}; "
          f"{'created' if created else 'existing v4 file preserved'}")
    verify_preserved_submissions(output_dir, preserved)
    for old_name in preserved:
        print(f"Preserved {old_name}: SHA256 and modification time unchanged")
    final_manifest = {
        "seed": SEED, "history_threshold_seconds": THRESHOLD, "fit_start": str(A_START), "fit_end": str(TEST_START),
        "target_start": str(TEST_START), "target_end": str(TEST_END), "configuration": configuration,
        "component_configurations": manifest["component_configurations"], "test_candidate_count": len(candidates),
        "validation_selection_manifest_sha256": digest, "preserved_submissions": preserved,
        "target_schedule_fields": manifest["target_schedule_fields"], "screen_time_percentages": [80, 50],
        "completion_requires_300_seconds": False,
    }
    (ROOT / "results" / "submission_hybrid_v4_frozen.json").write_text(json.dumps(final_manifest, indent=2, sort_keys=True) + "\n")
    print("Saved: submissions/submission_hybrid_v4.csv; results/submission_checks_v4.txt; results/submission_hybrid_v4_frozen.json")
    print(f"Total runtime: {time.perf_counter() - started:.3f} seconds")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--v2", action="store_true", help="Refit the frozen day-based hybrid and write only the new v2 CSV")
    mode.add_argument("--v3", action="store_true", help="Refit the frozen calibrated habit hybrid and write only the new v3 CSV")
    mode.add_argument("--v4", action="store_true", help="Refit the frozen TK-selected completion hybrid and write only the new v4 CSV")
    args = parser.parse_args()
    started = time.perf_counter()
    if args.v4:
        runner, filename = run_v4, "submission_checks_v4.txt"
    elif args.v3:
        runner, filename = run_v3, "submission_checks_v3.txt"
    elif args.v2:
        runner, filename = run_v2, "submission_checks_v2.txt"
    else:
        runner, filename = run, "submission_checks.txt"
    output = ROOT / "results" / filename
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as report, redirect_stdout(Tee(sys.stdout, report)):
        runner(started)


if __name__ == "__main__":
    main()

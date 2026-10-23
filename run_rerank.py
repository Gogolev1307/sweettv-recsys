"""One leakage-controlled learned re-ranking experiment, then an isolated v5 CSV."""

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
import sklearn

from run_validation import Tee
from src.data import (
    A_START, B_START, CACHE_DIR, DATA_DIR, ROOT, TEST_END, TEST_FILE, TEST_START, TRAIN_FILES, VIEW_FILE,
    airings_in_window, cached_result, load_submission, load_test_candidates, load_training_schedule,
)
from src.evaluate import ValidationProtocol, average_precision_at_k
from src.experiment import load_calendar_join, load_final_target_airings, load_overlap_join, load_slot_profile
from src.hybrid import fuse_completion_scores, minmax_normalize
from src.join import watched_interactions
from src.matrix import InteractionMatrix, rank_scores
from src.popularity import PopularityRecommender
from src.ppr import PersonalizedPageRank
from src.rerank import (
    CONFIGURATIONS, FEATURE_NAMES, PROMOTION_RULE, LearnedReranker, RerankFeatures,
    check_temporal_split, user_partition_robustness,
)
from src.submission import (
    load_frozen_v4_configuration, snapshot_submissions, validate_submission,
    verify_preserved_submissions, write_or_preserve_submission,
)
from src.truths import competition_truth

VALIDATION_SPLITS = ((pd.Timestamp("2020-04-06"), B_START), (pd.Timestamp("2020-04-20"), B_START))
FINAL_SPLITS = ((B_START, TEST_START), (pd.Timestamp("2020-06-15"), TEST_START))
BASE_PARAMETERS = {"k": 80, "score80": 1.0, "score50": 0.5, "popularity_freq": 0.1, "candidates": 50}


def json_text(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def prepare_training_split(fit_end, label_end, ceiling):
    check_temporal_split(A_START, fit_end, fit_end, label_end, ceiling)
    sources = [DATA_DIR / name for name in (VIEW_FILE, *TRAIN_FILES)]

    def build():
        schedule, _ = load_training_schedule()
        fitting = load_overlap_join(A_START, fit_end)
        labelled = load_overlap_join(fit_end, label_end)
        users = np.sort(watched_interactions(fitting.interactions).user_id.unique()).astype(np.uint64)
        candidates = np.sort(airings_in_window(schedule, fit_end, label_end).loc[lambda frame: frame.tv_show_id.ne(0), "tv_show_id"].unique())
        builder = RerankFeatures().fit(
            fitting.interactions, users, candidates, schedule, A_START, fit_end, schedule, fit_end, label_end,
        )
        rows = builder.generate(users)
        truth, accounting = competition_truth(users, candidates, labelled)
        labels = rows.labels(truth.relevant_by_user)
        if rows.items.shape[1] != 50:
            raise AssertionError("Every internal fitting user must have 50 distinct candidates")
        summary = {
            "fit_start": str(A_START), "fit_end": str(fit_end), "label_start": str(fit_end), "label_end": str(label_end),
            "users": len(users), "candidate_pool": len(candidates), "rows": len(labels), "positives": int(labels.sum()),
            "users_with_label_truth": accounting["evaluated_users"], "features": len(FEATURE_NAMES),
        }
        return rows, labels, summary

    return cached_result(
        CACHE_DIR, "rerank_training_rows", sources,
        {"fit_end": str(fit_end), "label_end": str(label_end), "label_ceiling": str(ceiling),
         "training_users": "all_300s_fit_history_users", "base": BASE_PARAMETERS, "feature_version": 1}, build,
    )


def training_data(splits, ceiling):
    datasets, labels, summaries = [], [], []
    for fit_end, label_end in splits:
        started = time.perf_counter()
        rows, target, summary = prepare_training_split(fit_end, label_end, ceiling)
        datasets.append(rows.features)
        labels.append(target)
        summaries.append(summary)
        print(f"  fit=[{A_START.date()},{fit_end.date()}) label=[{fit_end.date()},{label_end.date()}): "
              f"users={summary['users']:,}; pool={summary['candidate_pool']:,}; rows={summary['rows']:,}; "
              f"positives={summary['positives']:,}; preparation={time.perf_counter() - started:.3f}s")
    return np.vstack(datasets), np.concatenate(labels), summaries


def per_user_ap(predictions, truth, users):
    return np.array([average_precision_at_k(predictions[user], truth[user]) for user in users], dtype=np.float64)


def run(report, started):
    random.seed(42)
    np.random.seed(42)
    output_dir = ROOT / "submissions"
    output_dir.mkdir(parents=True, exist_ok=True)
    preserved = snapshot_submissions(output_dir, "submission_rerank_v5.csv")
    frozen, frozen_digest = load_frozen_v4_configuration(ROOT / "results" / "hybrid_v4_selected.json")
    if frozen["configuration"]["k"] != 80 or frozen["configuration"]["w50"] != .5 or frozen["configuration"]["w_pop"] != .1:
        raise ValueError("This experiment requires the stated frozen v4 candidate-generation coefficients")
    print("sweet.tv learned re-ranker v5 experiment (TK and T1 MAP@5)")
    print(f"Python {platform.python_version()} | numpy {np.__version__} | scipy {scipy.__version__} | pandas {pd.__version__} | scikit-learn {sklearn.__version__}")
    print("Seed: 42; BLAS/OpenMP threads: 1; HistGradientBoostingClassifier random_state=42; early_stopping=False")
    print("Candidates: top 50 by minmax(score80) + 0.5*minmax(score50) + 0.1*popularity_freq/max; k=80; normalization over the complete window pool.")
    print("Internal training uses all users with >=300s fit history; users with empty label truth contribute negative rows. No B labels enter validation fitting.")
    print("Features: threshold-specific n, channel-aware s, nt on channels where that user watched that programme at that threshold, and recency.")
    print("Recency is elapsed days from the fit cutoff to the last qualifying airing START; 999 if none. User total80 includes programmes outside the candidate pool.")
    print("Target schedule supplies programme/channel/time availability only. Re-watching, id ties, popularity fill and cold-start policy match v4.")
    print(f"Feature order ({len(FEATURE_NAMES)}): {', '.join(FEATURE_NAMES)}")
    print("Configuration selection: maximum TK MAP@5 on B; first configuration retained on ties.")
    print(f"Clear-win rule (declared before fitting): {PROMOTION_RULE}; otherwise negative result and hybrid_v4 stays final.")
    print("Robustness: 200 seed-42 random 50/50 partitions of the same TK-evaluated users; first-half statistics are primary, complementary halves also reported.")
    print("\nVALIDATION TRAINING SPLITS (labels wholly inside A)")
    train_x, train_y, validation_training = training_data(VALIDATION_SPLITS, B_START)
    print(f"Combined validation training: rows={len(train_y):,}; positives={int(train_y.sum()):,}; positive fraction={train_y.mean():.6f}")

    prep = time.perf_counter()
    schedule, _ = load_training_schedule()
    users = tuple(map(int, load_submission().user_id))
    candidates = np.sort(airings_in_window(schedule, B_START, TEST_START).loc[lambda frame: frame.tv_show_id.ne(0), "tv_show_id"].unique())
    joined_a, joined_b = load_calendar_join("A", A_START, B_START), load_calendar_join("B", B_START, TEST_START)
    tk, tk_accounting = competition_truth(users, candidates, joined_b)
    t1 = ValidationProtocol.from_interactions(users, airings_in_window(schedule, B_START, TEST_START), watched_interactions(joined_b.interactions))
    watched = watched_interactions(joined_a.interactions)
    popularity = PopularityRecommender().fit(watched, candidates)
    cold = set(users) - popularity.history_users
    builder = RerankFeatures().fit(joined_a.interactions, users, candidates, schedule, A_START, B_START, schedule, B_START, TEST_START)
    validation_rows = builder.generate(users)
    validation_labels = validation_rows.labels(tk.relevant_by_user)
    relevant_total = sum(map(len, tk.relevant_by_user.values()))
    print(f"\nOUTER VALIDATION: fit=A; label=B; pool={len(candidates):,}; generated rows={validation_rows.items.size:,}; "
          f"TK candidate recall={validation_labels.sum() / relevant_total:.6%}")
    print(f"TK cohort: 1,260 = {tk_accounting['evaluated_users']:,} evaluated + {tk_accounting['no_relevant_users']:,} no truth; "
          f"T1 evaluated={len(t1.relevant_by_user):,}; A-cold users={len(cold)}")
    print(f"Outer feature preparation: {time.perf_counter() - prep:.3f}s")

    point_started = time.perf_counter()
    raw80, raw50 = builder.completion.scores(users, 80)
    frequency = builder.frequency
    maximum = float(frequency.max(initial=0))
    pop_frequency = np.divide(frequency, maximum, out=np.zeros_like(frequency), where=maximum != 0)
    slot = load_slot_profile("A", users, A_START, B_START)
    slot_scores = slot.score(users, candidates, schedule, B_START, TEST_START)
    matrix = cached_result(
        CACHE_DIR, "method_matrix", [DATA_DIR / name for name in (VIEW_FILE, *TRAIN_FILES, "submission.csv")],
        {"threshold": 300, "matrix_version": 1}, lambda: InteractionMatrix.build(watched, users, candidates),
    )
    ppr_parameters = frozen["component_configurations"]["PPR"]
    ppr = PersonalizedPageRank(**{key: value for key, value in ppr_parameters.items() if key != "alpha"}).fit(matrix.seconds)
    ppr_scores, ppr_diagnostics = ppr.score(matrix.test_rows, matrix.candidate_columns, ppr_parameters["alpha"])
    if ppr_diagnostics["non_converged_users"]:
        raise AssertionError("Frozen v4 baseline PPR did not converge")
    baseline_scores = fuse_completion_scores({
        "score80": minmax_normalize(raw80), "score50": minmax_normalize(raw50),
        "slot": minmax_normalize(slot_scores), "ppr": minmax_normalize(ppr_scores), "pop": pop_frequency,
    }, frozen["configuration"])
    baseline_predictions = t1.finalize_predictions(rank_scores(baseline_scores, candidates, users), popularity.ranking, popularity.history_users)
    baseline_tk, baseline_t1 = tk.score(baseline_predictions), t1.score(baseline_predictions)
    if abs(baseline_tk - frozen["validation_scores"]["TK"]) > 1e-9 or abs(baseline_t1 - frozen["validation_scores"]["T1"]) > 1e-9:
        raise AssertionError("Frozen hybrid_v4 validation control changed")
    baseline_seconds = time.perf_counter() - point_started
    evaluated_users = tuple(sorted(tk.relevant_by_user))
    baseline_ap = per_user_ap(baseline_predictions, tk.relevant_by_user, evaluated_users)
    records = [{
        "method": "hybrid_v4", "hyperparameters": json_text(frozen["configuration"]), "selected": False,
        "map_at_5_TK": baseline_tk, "map_at_5_T1": baseline_t1, "evaluated_users_TK": len(evaluated_users),
        "evaluated_users_T1": len(t1.relevant_by_user), "runtime_seconds": baseline_seconds, "seed": 42,
    }]
    print(f"hybrid_v4 control: TK={baseline_tk:.9f}; T1={baseline_t1:.9f}; runtime={baseline_seconds:.3f}s")
    del raw80, raw50, slot_scores, ppr_scores, baseline_scores, slot, ppr, matrix, builder, joined_a, joined_b, watched

    print("\nRE-RANKER GRID (all fits use only the two internal-A labelled datasets)")
    best_index, best_configuration = None, None
    for configuration in CONFIGURATIONS:
        point_started = time.perf_counter()
        fit_started = time.perf_counter()
        model = LearnedReranker(configuration).fit(train_x, train_y)
        fit_seconds = time.perf_counter() - fit_started
        predictions = t1.finalize_predictions(model.predict(validation_rows), popularity.ranking, popularity.history_users)
        score_tk, score_t1 = tk.score(predictions), t1.score(predictions)
        if any(predictions[user] != popularity.ranking[:5] for user in cold):
            raise AssertionError("Re-ranker cold-start fallback differs from frozen v4")
        ap = per_user_ap(predictions, tk.relevant_by_user, evaluated_users)
        robust = user_partition_robustness(baseline_ap, ap)
        record = {
            "method": "rerank", "hyperparameters": json_text(configuration), "selected": False,
            "map_at_5_TK": score_tk, "map_at_5_T1": score_t1, "evaluated_users_TK": len(evaluated_users),
            "evaluated_users_T1": len(t1.relevant_by_user), "runtime_seconds": time.perf_counter() - point_started,
            "fit_seconds": fit_seconds, "actual_iterations": model.model.n_iter_, "seed": 42,
            "TK_ratio_over_v4": score_tk / baseline_tk, **robust,
        }
        records.append(record)
        index = len(records) - 1
        print(f"{json_text(configuration)}: TK={score_tk:.9f}; T1={score_t1:.9f}; TK/v4={score_tk / baseline_tk:.6f}x; "
              f"iterations={model.model.n_iter_}; fit={fit_seconds:.3f}s; total={record['runtime_seconds']:.3f}s")
        print(f"  first-half wins={robust['half_sample_wins']}/200; ratio min/median/max="
              f"{robust['ratio_min']:.6f}/{robust['ratio_median']:.6f}/{robust['ratio_max']:.6f}; "
              f"complement wins={robust['complementary_wins']}/200; both halves win={robust['both_halves_wins']}/200")
        if best_index is None or score_tk > records[best_index]["map_at_5_TK"]:
            best_index, best_configuration = index, dict(configuration)
        del model
    records[best_index]["selected"] = True
    selected = records[best_index]
    clear_win = selected["TK_ratio_over_v4"] >= 1.01 and selected["half_sample_wins"] >= 190
    outcome = "positive" if clear_win else "negative"
    final_choice = "submissions/submission_rerank_v5.csv" if clear_win else "submissions/submission_hybrid_v4.csv"
    for index, record in enumerate(records):
        record["experiment_outcome"] = outcome
        record["recommended_final"] = index == (best_index if clear_win else 0)
    print(f"\nSelected re-ranker: {json_text(best_configuration)}; TK={selected['map_at_5_TK']:.9f}; T1={selected['map_at_5_T1']:.9f}")
    print(f"Selected robustness, all 400 halves: ratio min/median/max="
          f"{selected['all_halves_ratio_min']:.6f}/{selected['all_halves_ratio_median']:.6f}/{selected['all_halves_ratio_max']:.6f}")
    print(f"Outcome under the declared clear-win rule: {outcome.upper()}; recommended final={final_choice}")
    frame = pd.DataFrame(records)
    frame.to_csv(ROOT / "results" / "validation_rerank.csv", index=False, float_format="%.9f")
    selection_manifest = {
        "configuration": best_configuration, "seed": 42, "early_stopping": False, "feature_names": FEATURE_NAMES,
        "base_parameters": BASE_PARAMETERS, "validation_training_splits": validation_training,
        "selected_validation_results": selected, "baseline_TK": baseline_tk, "baseline_T1": baseline_t1,
        "frozen_v4_manifest_sha256": frozen_digest, "promotion_rule": PROMOTION_RULE,
        "outcome": outcome, "recommended_final": final_choice, "sklearn_version": sklearn.__version__,
    }
    (ROOT / "results" / "rerank_v5_selected.json").write_text(json.dumps(selection_manifest, indent=2, sort_keys=True) + "\n")
    report.write("\nVALIDATION CONFIGURATION TABLE\n" + frame.to_string(index=False, float_format=lambda value: f"{value:.9f}") + "\n\n")
    report.flush()
    del train_x, train_y, validation_rows, validation_labels, baseline_predictions, predictions

    print("\nFINAL TRAINING SPLITS (selected parameters frozen; labels stay within weeks 11-30)")
    final_x, final_y, final_training = training_data(FINAL_SPLITS, TEST_START)
    fit_started = time.perf_counter()
    final_model = LearnedReranker(best_configuration).fit(final_x, final_y)
    print(f"Final model fit: rows={len(final_y):,}; positives={int(final_y.sum()):,}; "
          f"iterations={final_model.model.n_iter_}; runtime={time.perf_counter() - fit_started:.3f}s")
    del final_x, final_y
    prep = time.perf_counter()
    full = load_calendar_join("all_training", A_START, TEST_START)
    final_candidates = load_test_candidates()
    target_schedule = load_final_target_airings()
    final_builder = RerankFeatures().fit(
        full.interactions, users, final_candidates, schedule, A_START, TEST_START, target_schedule, TEST_START, TEST_END,
    )
    final_rows = final_builder.generate(users)
    final_popularity = PopularityRecommender().fit(watched_interactions(full.interactions), final_candidates)
    final_protocol = ValidationProtocol(users, frozenset(map(int, final_candidates)), {})
    final_predictions = final_protocol.finalize_predictions(
        final_model.predict(final_rows), final_popularity.ranking, final_popularity.history_users,
    )
    final_cold = set(users) - final_popularity.history_users
    path = output_dir / "submission_rerank_v5.csv"
    created = write_or_preserve_submission(path, users, final_predictions)
    checks = validate_submission(path, users, final_candidates, final_cold, final_popularity.ranking[:5])
    print(f"Final feature/scoring/write/check: rows={final_rows.items.size:,}; target candidates={len(final_candidates):,}; "
          f"runtime={time.perf_counter() - prep:.3f}s")
    print(f"submissions/submission_rerank_v5.csv: rows={checks['rows']:,}; header=PASS; template user order=PASS; "
          "plain integer user_id=PASS; five space-separated distinct ids=PASS; test candidate membership=PASS; "
          f"no id=0=PASS; cold-start fallback users={checks['cold_start_fallback_users']}; "
          f"{'created' if created else 'existing v5 file preserved'}")
    verify_preserved_submissions(output_dir, preserved)
    for name in preserved:
        print(f"Preserved {name}: SHA256 and modification time unchanged")
    final_manifest = {
        "configuration": best_configuration, "feature_names": FEATURE_NAMES, "base_parameters": BASE_PARAMETERS,
        "seed": 42, "early_stopping": False, "final_training_splits": final_training,
        "fit_start": str(A_START), "fit_end": str(TEST_START), "target_start": str(TEST_START), "target_end": str(TEST_END),
        "frozen_v4_manifest_sha256": frozen_digest, "outcome": outcome, "recommended_final": final_choice,
        "preserved_submissions": preserved, "test_candidate_count": len(final_candidates),
        "cold_start_fallback_users": len(final_cold), "sklearn_version": sklearn.__version__,
    }
    (ROOT / "results" / "rerank_v5_final.json").write_text(json.dumps(final_manifest, indent=2, sort_keys=True) + "\n")
    print("Saved: results/validation_rerank.csv; results/validation_rerank.txt; results/rerank_v5_selected.json; results/rerank_v5_final.json")
    print(f"Experiment conclusion: {outcome.upper()}; recommended final={final_choice}")
    print(f"Total runtime: {time.perf_counter() - started:.3f} seconds")


if __name__ == "__main__":
    started = time.perf_counter()
    output = ROOT / "results" / "validation_rerank.txt"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as report, redirect_stdout(Tee(sys.stdout, report)):
        run(report, started)

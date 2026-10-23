"""Compare all recommendation methods under the frozen A -> B protocol."""

from __future__ import annotations

# Set before importing NumPy: many tiny exact ALS solves are faster with one
# BLAS thread. This also fixes the numerical execution policy for all grids.
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
from itertools import product

import numpy as np
import pandas as pd
import scipy

from run_validation import Tee
from src.assoc import AssociationRecommender
from src.cbf import ContentRecommender
from src.data import (
    A_START, B_START, CACHE_DIR, DATA_DIR, ROOT, TEST_START, TRAIN_FILES, VIEW_FILE,
    airings_in_window, cached_result, load_programme_metadata, load_submission,
    load_training_schedule, load_views,
)
from src.evaluate import ValidationProtocol, average_precision_at_k
from src.hybrid import COMPONENTS, fuse_scores, normalize_components, weight_grid
from src.join import join_views_to_airings, watched_interactions
from src.matrix import InteractionMatrix, rank_scores
from src.mf import ImplicitALS
from src.popularity import PopularityRecommender
from src.ppr import PersonalizedPageRank
from src.rewatch import RewatchRecommender

SEED = 42
THRESHOLD = 300


def json_text(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def md(value):
    if value is None or pd.isna(value) or not str(value).strip():
        return "—"
    return str(value).replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ").replace("\r", " ")


def write_examples(path, protocol, matrix, airing_counts, metadata, popularity, predictions, parameters):
    """Use selected configurations; B is consulted only for eligibility/AP labels."""
    counts = {int(user): int(count) for user, count in airing_counts.items()}
    heaviest = min(protocol.test_users, key=lambda user: (-counts.get(user, 0), user))
    eligible = [user for user in protocol.test_users if counts.get(user, 0) > 0 and user in protocol.relevant_by_user]
    lightest = min(eligible, key=lambda user: (counts[user], user))
    lookup = metadata.set_index("tv_show_id").to_dict("index")
    lines = [
        "# Recommendation examples", "",
        "All models use the selected configurations from validation_methods.csv. Seed: 42; watch threshold: 300 seconds.",
        "The heaviest viewer maximises the number of watched A airings among all 1,260 test users. "
        "The lightest viewer minimises that count among test users with at least one A watched airing and nonempty B truth. "
        "User-id ties are broken in ascending order. The top 10 history programmes are ordered by A seconds, then programme id.",
        "Titles/categories use the latest A metadata, with earliest B metadata as a fallback. "
        "A-history flags and distinct-viewer popularity use A only; relevance and AP use B only.", "",
    ]
    for label, user in (("Heaviest", heaviest), ("Lightest eligible", lightest)):
        row = int(np.searchsorted(matrix.user_ids, np.uint64(user)))
        history = matrix.seconds.getrow(row)
        seen = set(map(int, matrix.item_ids[history.indices]))
        relevant = protocol.relevant_by_user.get(user, frozenset())
        lines.extend([
            f"## {label} viewer: `{user}`", "",
            f"History: **{counts.get(user, 0):,} watched A airings**, **{history.nnz:,} distinct A programmes**, "
            f"**{int(history.sum()):,} watched A seconds**. B truth: **{len(relevant):,} distinct programmes**.", "",
            "### Top programmes watched in A", "",
            "| Programme id | Title | Category | A seconds |", "|---:|---|---|---:|",
        ])
        order = np.lexsort((matrix.item_ids[history.indices], -history.data))[:10]
        for index in order:
            item = int(matrix.item_ids[history.indices[index]])
            fields = lookup.get(item, {})
            lines.append(f"| {item} | {md(fields.get('tv_show_title'))} | {md(fields.get('tv_show_category'))} | {int(history.data[index]):,} |")
        lines.append("")
        for method in ("popularity", "rewatch", "MF", "PPR", "CBF", "association", "hybrid"):
            top_five = predictions[method][user]
            ap = average_precision_at_k(top_five, relevant)
            lines.extend([
                f"### {method}: AP@5 = {ap:.9f}", "",
                f"Configuration: `{json_text(parameters[method])}`", "",
                "| Rank | Programme id | Title | Category | Distinct A viewers | Watched in A? | Relevant in B? |",
                "|---:|---:|---|---|---:|:---:|:---:|",
            ])
            for rank, item in enumerate(top_five, 1):
                fields = lookup.get(item, {})
                lines.append(
                    f"| {rank} | {item} | {md(fields.get('tv_show_title'))} | {md(fields.get('tv_show_category'))} | "
                    f"{int(popularity.counts.loc[item]):,} | {'yes' if item in seen else 'no'} | {'yes' if item in relevant else 'no'} |"
                )
            lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return heaviest, lightest


def run(args, report, started):
    random.seed(SEED)
    np.random.seed(SEED)
    cache_dir = None if args.no_cache else CACHE_DIR
    print("sweet.tv method comparison (MAP@5)")
    print(f"Python {platform.python_version()} | numpy {np.__version__} | scipy {scipy.__version__} | pandas {pd.__version__}")
    try:
        import sklearn
        print(f"scikit-learn {sklearn.__version__} installed; TF-IDF uses NumPy/SciPy")
    except ImportError:
        print("scikit-learn not installed; TF-IDF uses NumPy/SciPy")
    print(f"Seed: {SEED}; watch threshold: {THRESHOLD}s; BLAS/OpenMP threads: 1")
    print(f"Fit A=[{A_START}, {B_START}); truth B=[{B_START}, {TEST_START})")
    print("Common contract: 6,376 B-start candidates; re-watching allowed; ascending-id score ties; popularity fill/cold start.")
    print("Model/feature fitting uses A only; B viewing is used for validation scores and example relevance labels.")
    print("Per-configuration runtimes include fit, scoring, ranking and evaluation; hybrid times cover fusion/ranking/evaluation.")

    preparation_started = time.perf_counter()
    sources = [DATA_DIR / filename for filename in (VIEW_FILE, *TRAIN_FILES)]
    views = load_views(DATA_DIR, cache_dir)
    schedule, _ = load_training_schedule(DATA_DIR, cache_dir)
    test_users = tuple(map(int, load_submission().user_id))
    joins = []
    for name, lower, upper in (("A", A_START, B_START), ("B", B_START, TEST_START)):
        joins.append(cached_result(
            cache_dir, f"join_{name}", sources,
            {"window_start": str(lower), "window_end": str(upper), "join_version": 1},
            lambda lower=lower, upper=upper: join_views_to_airings(views, schedule, lower, upper),
        ))
    watched_a, watched_b = (watched_interactions(result.interactions, THRESHOLD) for result in joins)
    protocol = ValidationProtocol.from_interactions(test_users, airings_in_window(schedule, B_START, TEST_START), watched_b)
    if len(protocol.candidates) != 6376 or len(protocol.relevant_by_user) != 1246 or len(test_users) != 1260:
        raise AssertionError("The frozen validation candidate/user cohort changed")
    matrix = cached_result(
        cache_dir, "method_matrix", sources + [DATA_DIR / "submission.csv"],
        {"threshold": THRESHOLD, "matrix_version": 1},
        lambda: InteractionMatrix.build(watched_a, test_users, protocol.candidates),
    )
    candidate_ids = matrix.item_ids[matrix.candidate_columns]
    airing_counts = watched_a.groupby("user_id").size()
    metadata_a, metadata_available = load_programme_metadata(DATA_DIR, cache_dir)
    del views, schedule, joins, watched_b
    print(f"Matrix: {matrix.seconds.shape[0]:,} users x {matrix.seconds.shape[1]:,} programmes; "
          f"{matrix.seconds.nnz:,} observed pairs; values=total A overlap seconds after the airing threshold")
    print(f"Cohort: 1,260 = 1,246 evaluated + 14 with no B truth; "
          f"A-cold test users: {int((np.diff(matrix.seconds.indptr)[matrix.test_rows] == 0).sum())}")
    print(f"Metadata: {len(metadata_a):,} A snapshots; {len(metadata_available):,} A-first/B-fill snapshots; "
          "TF-IDF vocabulary/IDF fitted on A only")
    print(f"Shared input preparation: {time.perf_counter() - preparation_started:.3f}s")

    records, selected_predictions, selected_parameters = [], {}, {}
    popularity = PopularityRecommender()
    point_started = time.perf_counter()
    popularity.fit(watched_a, protocol.candidates)
    popularity_scores = popularity.counts.reindex(candidate_ids).to_numpy(dtype=np.float64)

    def record(method, parameters, scores, point_started, *, positive_only=True, diagnostics=None):
        proposed = rank_scores(scores, candidate_ids, test_users, positive_only=positive_only)
        predictions = protocol.finalize_predictions(proposed, popularity.ranking, popularity.history_users)
        score = protocol.score(predictions)
        for user in test_users:
            if user not in popularity.history_users and predictions[user] != popularity.ranking[:5]:
                raise AssertionError("Method-specific cold-start fallback")
        row = {
            "method": method, "map_at_5": score, "evaluated_users": len(protocol.relevant_by_user),
            "runtime_seconds": time.perf_counter() - point_started, "hyperparameters": json_text(parameters),
            "selected": False, "reference_baseline": method in ("popularity", "rewatch"),
            "best_two_component": False, "component_pair": "", "seed": SEED,
            "diagnostics": json_text(diagnostics or {}),
        }
        records.append(row)
        return len(records) - 1, predictions

    def select(index, predictions, parameters):
        records[index]["selected"] = True
        method = records[index]["method"]
        selected_predictions[method] = predictions
        selected_parameters[method] = parameters

    def print_point(index):
        row = records[index]
        print(f"{row['method']} {row['hyperparameters']}: MAP@5={row['map_at_5']:.9f}; "
              f"users={row['evaluated_users']:,}; runtime={row['runtime_seconds']:.3f}s")

    print("\nREFERENCE BASELINES")
    index, predictions = record(
        "popularity", {}, np.broadcast_to(popularity_scores, (len(test_users), len(candidate_ids))), point_started
    )
    select(index, predictions, {})
    print_point(index)
    if not np.isclose(records[index]["map_at_5"], 0.737594971, atol=1e-9, rtol=0):
        raise AssertionError("Popularity no longer matches the existing validation")
    point_started = time.perf_counter()
    rewatch_scores = RewatchRecommender().fit(matrix.seconds).score(matrix.test_rows, matrix.candidate_columns)
    index, predictions = record("rewatch", {}, rewatch_scores, point_started)
    select(index, predictions, {})
    print_point(index)
    del watched_a

    print("\nIMPLICIT ALS GRID (10 exact alternating user/item solves)")
    best_index, best_predictions, best_parameters, mf_scores = None, None, None, None
    for factors, reg, alpha in product((4, 8, 16, 32), (1, 10), (1, 10)):
        parameters = {"factors": factors, "reg": reg, "alpha": alpha, "iterations": 10, "seed": SEED}
        point_started = time.perf_counter()
        model = ImplicitALS(**parameters).fit(matrix.seconds)
        scores = model.score(matrix.test_rows, matrix.candidate_columns)
        index, predictions = record("MF", parameters, scores, point_started, positive_only=False)
        print_point(index)
        if best_index is None or records[index]["map_at_5"] > records[best_index]["map_at_5"]:
            best_index, best_predictions, best_parameters, mf_scores = index, predictions, parameters, scores
    select(best_index, best_predictions, best_parameters)
    print(f"Selected MF: {json_text(best_parameters)}; MAP@5={records[best_index]['map_at_5']:.9f}")
    del model, scores

    print("\nPERSONALISED PAGERANK (alpha=follow probability; restart=1-alpha)")
    graph_started = time.perf_counter()
    ppr = PersonalizedPageRank(batch_size=args.ppr_batch_size).fit(matrix.seconds)
    graph_seconds = time.perf_counter() - graph_started
    d = ppr.diagnostics
    print(f"Transition max row-sum errors (non-dangling rows target 1, dangling rows target 0): "
          f"U->I={d['user_row_sum_error']:.3e}; I->U={d['item_row_sum_error']:.3e}")
    print(f"Dangling nodes: {d['dangling_users'] + d['dangling_items']:,} "
          f"(users={d['dangling_users']:,}, programmes={d['dangling_items']:,})")
    normal_target = int(matrix.test_rows[np.flatnonzero(~ppr.dangling_users[matrix.test_rows])[0]])
    before, after = ppr.one_update_mass_check(normal_target)
    print(f"One update, ordinary restart: total mass before={before:.12f}; after={after:.12f}")
    if ppr.dangling_users.any():
        before, after = ppr.one_update_mass_check(int(np.flatnonzero(ppr.dangling_users)[0]))
        print(f"One update, dangling restart: total mass before={before:.12f}; after={after:.12f}")
    best_index, best_predictions, best_parameters, ppr_scores = None, None, None, None
    for alpha in (0.3, 0.5, 0.6, 0.7, 0.8, 0.85):
        parameters = {"alpha": alpha, "tolerance": 1e-6, "max_iterations": 500, "batch_size": args.ppr_batch_size}
        point_started = time.perf_counter()
        scores, diagnostics = ppr.score(matrix.test_rows, matrix.candidate_columns, alpha)
        # Attribute the measured shared graph construction to every PPR config.
        index, predictions = record("PPR", parameters, scores, point_started - graph_seconds, diagnostics=diagnostics)
        print_point(index)
        print(f"  iterations mean/max={diagnostics['mean_iterations']:.3f}/{diagnostics['max_iterations']}; "
              f"max final L1 residual={diagnostics['max_final_l1_residual']:.3e}; "
              f"non-converged users={diagnostics['non_converged_users']}; "
              f"max mass error={diagnostics['max_final_mass_error']:.3e}")
        if best_index is None or records[index]["map_at_5"] > records[best_index]["map_at_5"]:
            best_index, best_predictions, best_parameters, ppr_scores = index, predictions, parameters, scores
    select(best_index, best_predictions, best_parameters)
    print(f"Selected PPR: {json_text(best_parameters)}; MAP@5={records[best_index]['map_at_5']:.9f}")
    del ppr, scores

    print("\nCONTENT AND ASSOCIATION")
    point_started = time.perf_counter()
    cbf = ContentRecommender().fit(matrix.seconds, matrix.item_ids, metadata_a, metadata_available)
    scores = cbf.score(matrix.test_rows, matrix.candidate_columns)
    parameters = {"tf": "binary", "idf": "smooth_A_only", "profile_weight": "log1p(seconds/60)", "year": "first_year_decade"}
    index, predictions = record("CBF", parameters, scores, point_started)
    select(index, predictions, parameters)
    print_point(index)
    print(f"CBF features: {len(cbf.vocabulary):,}; zero-vector programmes: {cbf.zero_vector_items:,}")
    del cbf, scores
    point_started = time.perf_counter()
    association = AssociationRecommender().fit(matrix.seconds)
    scores = association.score(matrix.test_rows, matrix.candidate_columns)
    parameters = {"minimum_support": 20, "lift_strictly_greater_than": 1, "score": "max_confidence"}
    diagnostics = {"rules_kept": association.rule_count, "nonempty_A_baskets": association.basket_count}
    index, predictions = record("association", parameters, scores, point_started, diagnostics=diagnostics)
    select(index, predictions, parameters)
    print_point(index)
    print(f"Association rules kept: {association.rule_count:,}; nonempty A baskets: {association.basket_count:,}")
    del association, scores

    print("\nHYBRID GRID (255 weight configurations; order: popularity, rewatch, PPR, MF)")
    normalization_started = time.perf_counter()
    components = normalize_components(popularity_scores, rewatch_scores, ppr_scores, mf_scores)
    normalization_seconds = time.perf_counter() - normalization_started
    del rewatch_scores, ppr_scores, mf_scores
    best_index, best_predictions, best_parameters = None, None, None
    pair_indices = {}
    for weights in weight_grid():
        parameters = dict(zip(COMPONENTS, weights))
        point_started = time.perf_counter()
        scores = fuse_scores(components, weights)
        index, predictions = record("hybrid", parameters, scores, point_started - normalization_seconds)
        active = tuple(name for name, weight in zip(COMPONENTS, weights) if weight)
        if len(active) == 2:
            records[index]["component_pair"] = "+".join(active)
            previous = pair_indices.get(active)
            if previous is None or records[index]["map_at_5"] > records[previous]["map_at_5"]:
                pair_indices[active] = index
        if best_index is None or records[index]["map_at_5"] > records[best_index]["map_at_5"]:
            best_index, best_predictions, best_parameters = index, predictions, parameters
    select(best_index, best_predictions, best_parameters)
    print(f"Selected hybrid: {json_text(best_parameters)}; MAP@5={records[best_index]['map_at_5']:.9f}")
    print("Best two-component fusions (best grid weights for each pair):")
    for pair, index in sorted(pair_indices.items()):
        records[index]["best_two_component"] = True
        row = records[index]
        print(f"  {'+'.join(pair)}: {row['hyperparameters']}; MAP@5={row['map_at_5']:.9f}; runtime={row['runtime_seconds']:.3f}s")
    del components, scores

    frame = pd.DataFrame(records)
    if len(frame) != 281 or not (frame.evaluated_users == 1246).all() or frame.selected.sum() != 7:
        raise AssertionError("Missing configurations or inconsistent evaluation cohorts")
    results_dir = ROOT / "results"
    frame.to_csv(results_dir / "validation_methods.csv", index=False, float_format="%.9f")
    heaviest, lightest = write_examples(
        results_dir / "examples.md", protocol, matrix, airing_counts, metadata_available,
        popularity, selected_predictions, selected_parameters,
    )
    print("\nSELECTED CONFIGURATIONS")
    for row in frame.loc[frame.selected].itertuples(index=False):
        print(f"{row.method}: MAP@5={row.map_at_5:.9f}; users={row.evaluated_users:,}; "
              f"runtime={row.runtime_seconds:.3f}s; parameters={row.hyperparameters}")
    print(f"Examples: heaviest test user={heaviest}; lightest eligible test user={lightest}")
    report.write("\nALL 281 CONFIGURATIONS (one row per configuration)\n")
    report.write(frame.to_string(index=False, float_format=lambda value: f"{value:.9f}"))
    report.write("\n\n")
    report.flush()
    print("Saved: results/validation_methods.csv; results/validation_methods.txt; results/examples.md")
    print(f"Total runtime: {time.perf_counter() - started:.3f} seconds")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-cache", action="store_true", help="Rebuild preprocessing without cache access")
    parser.add_argument("--ppr-batch-size", type=int, default=128)
    args = parser.parse_args()
    if args.ppr_batch_size <= 0:
        parser.error("--ppr-batch-size must be positive")
    started = time.perf_counter()
    results_dir = ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    with (results_dir / "validation_methods.txt").open("w", encoding="utf-8") as report:
        with redirect_stdout(Tee(sys.stdout, report)):
            run(args, report, started)


if __name__ == "__main__":
    main()

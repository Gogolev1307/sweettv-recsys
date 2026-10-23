"""Per-user min-max score fusion on the complete, fixed candidate pool."""

from __future__ import annotations

from itertools import product

import numpy as np

COMPONENTS = ("popularity", "rewatch", "PPR", "MF")
WEIGHT_VALUES = (0.0, 0.5, 1.0, 2.0)
V2_SELECTION_RULE = "maximise the unweighted mean of MAP@5 over T2, T3 and T4, each on its nonempty-truth cohort"
V3_SELECTION_RULE = "maximise the unweighted mean of MAP@5 over T2 and T3 only, each on its nonempty-truth cohort"
V3_COMPONENTS = ("days", "exp", "slot", "ppr", "mf", "pop")
V4_SELECTION_RULE = "maximise MAP@5 under TK; TK reproduces the competition's published ground-truth definition"
COMPLETION_K_VALUES = (20, 40, 80, 160)


def strict_truth_mean(scores):
    return sum(float(scores[name]) for name in ("T2", "T3", "T4")) / 3


def calibrated_truth_mean(scores):
    return (float(scores["T2"]) + float(scores["T3"])) / 2


def competition_truth_score(scores):
    return float(scores["TK"])


def minmax_normalize(scores: np.ndarray) -> np.ndarray:
    minimum = scores.min(axis=1, keepdims=True)
    span = scores.max(axis=1, keepdims=True) - minimum
    return np.divide(scores - minimum, span, out=np.zeros_like(scores, dtype=np.float64), where=span != 0)


def normalize_components(popularity, rewatch, ppr, mf):
    maximum = float(np.max(popularity))
    pop = np.divide(popularity, maximum, out=np.zeros_like(popularity, dtype=np.float64), where=maximum != 0)
    return (pop, minmax_normalize(rewatch), minmax_normalize(ppr), minmax_normalize(mf))


def weight_grid():
    return (weights for weights in product(WEIGHT_VALUES, repeat=4) if any(weights))


def fuse_scores(components, weights):
    if len(weights) != 4 or not any(weights):
        raise ValueError("Four weights with at least one nonzero component are required")
    shape = next(component.shape for component in components if component.ndim == 2)
    scores = np.zeros(shape, dtype=np.float64)
    for component, weight in zip(components, weights):
        if weight:
            scores += weight * component
    return scores


def days_weight_grid():
    """Fixed day component weight 1; 4 * 3 * 2 = 24 configurations."""
    for ppr, popularity, mf in product((0.0, 0.25, 0.5, 1.0), (0.0, 0.1, 0.3), (0.0, 0.2)):
        yield {"rewatch_days": 1.0, "w_ppr": ppr, "w_pop": popularity, "w_mf": mf}


def normalize_days_components(days, ppr, popularity, mf):
    maximum = float(np.max(popularity))
    pop = np.divide(popularity, maximum, out=np.zeros_like(popularity, dtype=np.float64), where=maximum != 0)
    return (minmax_normalize(days), minmax_normalize(ppr), pop, minmax_normalize(mf))


def fuse_days_scores(components, weights):
    if set(weights) != {"rewatch_days", "w_ppr", "w_pop", "w_mf"} or weights["rewatch_days"] != 1:
        raise ValueError("V2 needs fixed day weight 1 and named PPR/popularity/MF weights")
    if any(not np.isfinite(value) or value < 0 for value in weights.values()):
        raise ValueError("V2 weights must be finite and nonnegative")
    days, ppr, pop, mf = components
    return days + weights["w_ppr"] * ppr + weights["w_pop"] * pop + weights["w_mf"] * mf


def habit_weight_grid():
    """4^4 * 2 * 3 - 1 = 1,535 nonzero v3 configurations, ascending order."""
    for values in product((0.0, 0.5, 1.0, 2.0), (0.0, 0.5, 1.0, 2.0),
                          (0.0, 0.5, 1.0, 2.0), (0.0, 0.5, 1.0, 2.0), (0.0, 0.2), (0.0, 0.1, 0.3)):
        if any(values):
            yield dict(zip((f"w_{name}" for name in V3_COMPONENTS), values))


def normalize_habit_components(days, exposure, slot, ppr, mf, popularity):
    result = {name: minmax_normalize(scores) for name, scores in
              zip(V3_COMPONENTS[:5], (days, exposure, slot, ppr, mf))}
    maximum = float(np.max(popularity))
    result["pop"] = np.divide(popularity, maximum, out=np.zeros_like(popularity, dtype=np.float64), where=maximum != 0)
    return result


def fuse_habit_scores(components, weights):
    if set(weights) != {f"w_{name}" for name in V3_COMPONENTS} or not any(weights.values()):
        raise ValueError("V3 requires six named weights and a nonzero component")
    if any(not np.isfinite(value) or value < 0 for value in weights.values()):
        raise ValueError("V3 weights must be finite and nonnegative")
    scores = np.zeros_like(components["days"], dtype=np.float64)
    for name in V3_COMPONENTS:
        if weights[f"w_{name}"]:
            scores += weights[f"w_{name}"] * components[name]
    return scores


def completion_weight_grid(k=None):
    values = COMPLETION_K_VALUES if k is None else (k,)
    if any(value not in COMPLETION_K_VALUES for value in values):
        raise ValueError("Completion k is outside the validation grid")
    for smoothing in values:
        for w50, slot, ppr, pop in product((0.0, 0.15, 0.3, 0.5), (0.0, 0.05, 0.1, 0.2),
                                           (0.0, 0.05, 0.1, 0.2), (0.0, 0.05, 0.1, 0.2)):
            yield {"k": smoothing, "w50": w50, "w_slot": slot, "w_ppr": ppr, "w_pop": pop}


def fuse_completion_scores(components, weights):
    if set(weights) != {"k", "w50", "w_slot", "w_ppr", "w_pop"}:
        raise ValueError("V4 requires k and four named component weights")
    if weights["k"] <= 0 or any(not np.isfinite(weights[name]) or weights[name] < 0 for name in ("w50", "w_slot", "w_ppr", "w_pop")):
        raise ValueError("V4 smoothing must be positive and weights finite/nonnegative")
    scores = components["score80"].copy()
    for component, weight in (("score50", "w50"), ("slot", "w_slot"), ("ppr", "w_ppr"), ("pop", "w_pop")):
        if weights[weight]:
            scores += weights[weight] * components[component]
    return scores

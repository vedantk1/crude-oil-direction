"""Binary classification metrics with explicit undefined and clipping rules."""

import math
from numbers import Integral, Real


LOG_LOSS_EPSILON = 1e-15


def _binary_values(values, name: str) -> list[int]:
    result = list(values)
    if any(isinstance(value, bool) or not isinstance(value, Integral) or value not in (0, 1)
           for value in result):
        raise ValueError(f"{name} must contain integer 0 or 1 values")
    return result


def _roc_auc(labels: list[int], probabilities: list[float]) -> float | None:
    """Rank AUC gives tied scores half credit and needs both actual classes."""
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None
    ordered = sorted(zip(probabilities, labels))
    positive_rank_sum = 0.0
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][0] == ordered[index][0]:
            end += 1
        average_rank = (index + 1 + end) / 2
        positive_rank_sum += average_rank * sum(label for _, label in ordered[index:end])
        index = end
    return (positive_rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def classification_metrics(y_true, predictions, probabilities=None) -> dict:
    """Score paired binary observations without inferring rule probabilities.

    Balanced accuracy is the mean of up and not-up recall; it is null when
    either actual class is absent. ROC AUC is also null for a single class.
    Probability metrics are null unless a complete probability vector is
    provided. Brier score uses the original probabilities; natural-log loss
    clips probabilities to [1e-15, 1 - 1e-15]. Empty inputs return zero counts
    and null scores. Length mismatches and invalid values raise ValueError.
    """
    labels = _binary_values(y_true, "Labels")
    predicted = _binary_values(predictions, "Predictions")
    if len(labels) != len(predicted):
        raise ValueError("Labels and predictions must have matching lengths")
    probs = None if probabilities is None else list(probabilities)
    if probs is not None:
        if len(probs) != len(labels):
            raise ValueError("Labels and probabilities must have matching lengths")
        if any(isinstance(value, bool) or not isinstance(value, Real)
               or not math.isfinite(value) or not 0 <= value <= 1 for value in probs):
            raise ValueError("Probabilities must be finite numbers between zero and one")
    counts = {"tn": 0, "fp": 0, "fn": 0, "tp": 0}
    for actual, prediction in zip(labels, predicted):
        counts[{(0, 0): "tn", (0, 1): "fp", (1, 0): "fn", (1, 1): "tp"}[
            (actual, prediction)]] += 1
    count = len(labels)
    positives = counts["tp"] + counts["fn"]
    negatives = counts["tn"] + counts["fp"]
    result = {
        "count": count,
        "accuracy": (counts["tn"] + counts["tp"]) / count if count else None,
        "balanced_accuracy": ((counts["tp"] / positives + counts["tn"] / negatives) / 2
                              if positives and negatives else None),
        "confusion_counts": counts,
        "roc_auc": None,
        "brier_score": None,
        "log_loss": None,
    }
    if probs is not None and count:
        clipped = [min(1 - LOG_LOSS_EPSILON, max(LOG_LOSS_EPSILON, p)) for p in probs]
        result.update({
            "roc_auc": _roc_auc(labels, probs),
            "brier_score": math.fsum((p - y) ** 2 for p, y in zip(probs, labels)) / count,
            "log_loss": -math.fsum(math.log(p) if y else math.log1p(-p)
                                   for p, y in zip(clipped, labels)) / count,
        })
    return result

"""Metrics cover tied scores, missing classes, empty data, and invalid inputs."""

import math
import unittest

from oil_eval.metrics import LOG_LOSS_EPSILON, classification_metrics


class MetricTests(unittest.TestCase):
    def test_known_binary_example(self):
        result = classification_metrics([0, 0, 1, 1], [0, 1, 0, 1], [0.1, 0.8, 0.4, 0.9])
        self.assertEqual(result["count"], 4)
        self.assertEqual(result["accuracy"], 0.5)
        self.assertEqual(result["balanced_accuracy"], 0.5)
        self.assertEqual(result["confusion_counts"], {"tn": 1, "fp": 1, "fn": 1, "tp": 1})
        self.assertEqual(result["roc_auc"], 0.75)
        self.assertAlmostEqual(result["brier_score"], 0.255)
        self.assertAlmostEqual(result["log_loss"], -math.log(0.9 * 0.2 * 0.4 * 0.9) / 4)

    def test_balanced_accuracy_corrects_class_imbalance(self):
        result = classification_metrics([0, 0, 0, 1], [0, 0, 0, 0])
        self.assertEqual(result["accuracy"], 0.75)
        self.assertEqual(result["balanced_accuracy"], 0.5)
        self.assertIsNone(result["roc_auc"])
        self.assertIsNone(result["brier_score"])
        self.assertIsNone(result["log_loss"])

    def test_auc_half_credit_for_ties(self):
        result = classification_metrics([0, 1, 0, 1], [1, 1, 1, 1], [0.5] * 4)
        self.assertEqual(result["roc_auc"], 0.5)
        self.assertEqual(result["brier_score"], 0.25)

    def test_auc_perfect_and_reversed_order(self):
        self.assertEqual(classification_metrics([0, 1], [0, 1], [0, 1])["roc_auc"], 1)
        self.assertEqual(classification_metrics([0, 1], [1, 0], [1, 0])["roc_auc"], 0)

    def test_single_class_retains_defined_metrics(self):
        result = classification_metrics([1, 1], [1, 0], [0.8, 0.2])
        self.assertEqual(result["accuracy"], 0.5)
        self.assertIsNone(result["balanced_accuracy"])
        self.assertIsNone(result["roc_auc"])
        self.assertAlmostEqual(result["brier_score"], 0.34)
        self.assertIsNotNone(result["log_loss"])

    def test_empty_inputs_have_zero_counts_and_null_scores(self):
        result = classification_metrics([], [], [])
        self.assertEqual(result["count"], 0)
        self.assertEqual(result["confusion_counts"], {"tn": 0, "fp": 0, "fn": 0, "tp": 0})
        for name in ("accuracy", "balanced_accuracy", "roc_auc", "brier_score", "log_loss"):
            self.assertIsNone(result[name])

    def test_log_loss_clips_but_brier_uses_original_probabilities(self):
        result = classification_metrics([1, 0], [0, 1], [0, 1])
        expected = -(math.log(LOG_LOSS_EPSILON) + math.log1p(-(1 - LOG_LOSS_EPSILON))) / 2
        self.assertEqual(result["brier_score"], 1)
        self.assertAlmostEqual(result["log_loss"], expected)
        self.assertTrue(math.isfinite(result["log_loss"]))

    def test_invalid_lengths_labels_and_probabilities_fail(self):
        for args in (([1], []), ([1], [1], []), ([2], [1]), ([1], [0.5]),
                     ([True], [1]), ([1], [1], [None]), ([1], [1], [float("nan")]),
                     ([1], [1], [float("inf")]), ([1], [1], [-0.1]), ([1], [1], [1.1])):
            with self.subTest(args=args), self.assertRaises(ValueError):
                classification_metrics(*args)


if __name__ == "__main__":
    unittest.main()

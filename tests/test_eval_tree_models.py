"""Synthetic checks for the single fixed, imputer-only tree candidate."""

from copy import deepcopy
import unittest
from unittest.mock import patch

import numpy as np

from oil_eval.inventory_features import INVENTORY_FEATURE_COLUMNS, INVENTORY_MODEL_COLUMNS
from oil_eval.models import (GradientBoostingClassifier, INVENTORY_REGULARIZATION_MODEL_NAMES,
                             INVENTORY_TREE_MODEL_NAMES, Pipeline, TREE_PARAMETERS,
                             fit_predict_inventory_regularization_models,
                             fit_predict_inventory_tree_models)
from oil_eval.protocol import protocol_config
from tests.test_eval_models import inventory_row


def training_examples():
    rows = [inventory_row(index - 90, index % 17 - 8, int(index >= 90)) for index in range(180)]
    for index, row in enumerate(rows):
        row["row_id"] = f"synthetic-{index}"
        row["features"][INVENTORY_FEATURE_COLUMNS[0]] = None if index % 4 == 0 else index - 90
    return rows


class InventoryTreeModelTests(unittest.TestCase):
    def test_all_five_control_payloads_inputs_and_fixed_parameters_remain_exact(self):
        training, forecast = training_examples(), inventory_row(12, None)
        before = deepcopy((training, forecast, TREE_PARAMETERS))
        expected = fit_predict_inventory_regularization_models(training, forecast)
        actual = fit_predict_inventory_tree_models(training, forecast)
        self.assertEqual(tuple(actual), INVENTORY_TREE_MODEL_NAMES)
        self.assertEqual({name: actual[name] for name in INVENTORY_REGULARIZATION_MODEL_NAMES}, expected)
        self.assertEqual((training, forecast, TREE_PARAMETERS), before)
        self.assertNotIn("logistic_wti_inventory_c01_rolling", actual)

    def test_same_ordered_rows_and_labels_with_imputer_only_and_no_fit_weights(self):
        training, forecast, calls = training_examples(), inventory_row(12, None), []
        original_pipeline_fit = Pipeline.fit
        original_tree_fit = GradientBoostingClassifier.fit
        tree_fit_calls = []

        def capture_pipeline(model, train, labels, *args, **kwargs):
            calls.append((train.copy(), list(labels), tuple(model.named_steps),
                          model.named_steps["classifier"].get_params(),
                          model.named_steps["imputer"].get_params()))
            return original_pipeline_fit(model, train, labels, *args, **kwargs)

        def capture_tree(model, train, labels, *args, **kwargs):
            tree_fit_calls.append((train.copy(), list(labels), args, kwargs))
            return original_tree_fit(model, train, labels, *args, **kwargs)

        with patch("oil_eval.models.Pipeline.fit", new=capture_pipeline), \
                patch("oil_eval.models.GradientBoostingClassifier.fit", new=capture_tree):
            fit_predict_inventory_tree_models(training, forecast)
        self.assertEqual(len(calls), 4)
        logistic, tree = calls[-2:]
        self.assertEqual(tree[0].shape, (180, 11))
        np.testing.assert_allclose(tree[0], logistic[0], equal_nan=True)
        self.assertEqual(tree[1], [row["label"] for row in training])
        self.assertEqual(tree[1], logistic[1])
        self.assertEqual(tree[2], ("imputer", "classifier"))
        self.assertEqual(tree[3], TREE_PARAMETERS)
        self.assertEqual(tree[3], protocol_config("inventory-tree-comparison")["tree"]["parameters"])
        self.assertEqual(tree[4]["strategy"], "median")
        self.assertTrue(tree[4]["keep_empty_features"])
        self.assertFalse(tree[4]["add_indicator"])
        self.assertEqual(len(tree_fit_calls), 1)
        self.assertEqual(tree_fit_calls[0][0].shape, (180, 11))
        self.assertTrue(np.isfinite(tree_fit_calls[0][0]).all())
        self.assertEqual(tree_fit_calls[0][2:], ((), {}))

    def test_forecast_changes_cannot_fit_imputation_or_change_fitted_tree_diagnostics(self):
        training = training_examples()
        low, high = inventory_row(None, None), inventory_row(100000, 5000)
        low["features"].update(dict.fromkeys(INVENTORY_FEATURE_COLUMNS))
        high["features"].update(dict.fromkeys(INVENTORY_FEATURE_COLUMNS, 100000))
        first = fit_predict_inventory_tree_models(training, low)["tree_wti_inventory"]
        second = fit_predict_inventory_tree_models(training, high)["tree_wti_inventory"]
        left, right = first["diagnostics"], second["diagnostics"]
        self.assertEqual({key: value for key, value in left.items() if key != "forecast_missing_columns"},
                         {key: value for key, value in right.items() if key != "forecast_missing_columns"})
        column = INVENTORY_FEATURE_COLUMNS[1]
        self.assertEqual(left["imputer_statistics"][column], 0)
        self.assertEqual(left["training_missing_counts"][column], 180)
        self.assertEqual(left["feature_importances"][column], 0)
        self.assertEqual(left["training_missing_counts"][INVENTORY_FEATURE_COLUMNS[0]], 45)
        self.assertIn(column, left["forecast_missing_columns"])
        self.assertNotIn(column, right["forecast_missing_columns"])
        self.assertNotIn("scaler_mean", left)
        self.assertEqual(tuple(left["imputer_statistics"]), INVENTORY_MODEL_COLUMNS)

    def test_exact_tree_count_depth_and_leaf_constraints_with_deterministic_output(self):
        training, forecast = training_examples(), inventory_row(12, 2)
        first = fit_predict_inventory_tree_models(training, forecast)["tree_wti_inventory"]
        self.assertEqual(first, fit_predict_inventory_tree_models(training, forecast)["tree_wti_inventory"])
        diagnostics = first["diagnostics"]
        self.assertEqual(diagnostics["n_estimators_actual"], 50)
        self.assertLessEqual(diagnostics["max_depth_actual"], 2)
        self.assertGreater(diagnostics["max_depth_actual"], 0)
        self.assertEqual(len(diagnostics["estimator_depths"]), 50)
        self.assertEqual(len(diagnostics["estimator_leaf_counts"]), 50)
        self.assertEqual(len(diagnostics["estimator_leaf_sample_counts"]), 50)
        for depth, leaves, counts in zip(diagnostics["estimator_depths"], diagnostics["estimator_leaf_counts"],
                                         diagnostics["estimator_leaf_sample_counts"]):
            self.assertLessEqual(depth, 2)
            self.assertLessEqual(leaves, 4)
            self.assertEqual(len(counts), leaves)
            self.assertTrue(all(count >= 30 for count in counts))
            self.assertEqual(sum(counts), len(training))
        self.assertEqual(tuple(diagnostics["feature_importances"]), INVENTORY_MODEL_COLUMNS)
        self.assertAlmostEqual(sum(diagnostics["feature_importances"].values()), 1)

    def test_future_columns_outcomes_and_provenance_never_become_tree_inputs(self):
        training, forecast = training_examples(), inventory_row(12, 2)
        expected = fit_predict_inventory_tree_models(training, forecast)
        for row in training + [forecast]:
            row["features"].update(future_target=1e20, future_inventory_change=-1e20, label=123)
            row["target_price"] = 9999
            row["inventory_provenance"] = {"future_revision": 123}
        forecast["label"] = 999
        self.assertEqual(expected, fit_predict_inventory_tree_models(training, forecast))

    def test_probability_ties_predict_up_and_invalid_training_inputs_fail(self):
        balanced = [inventory_row(10, 0, label) for label in (0, 1, 0, 1)]
        result = fit_predict_inventory_tree_models(balanced, inventory_row(10, 0))["tree_wti_inventory"]
        self.assertEqual(result["probability_up"], 0.5)
        self.assertEqual(result["prediction"], 1)
        for training in ([], balanced[:1], [{**balanced[0], "label": 2}, balanced[1]]):
            with self.assertRaises(ValueError):
                fit_predict_inventory_tree_models(training, inventory_row(10, 0))
        for value in (float("nan"), float("inf"), True, "2"):
            forecast = inventory_row(10, 0)
            forecast["features"][INVENTORY_FEATURE_COLUMNS[0]] = value
            with self.assertRaisesRegex(ValueError, "finite numeric"):
                fit_predict_inventory_tree_models(balanced, forecast)

    def test_tree_errors_or_invalid_probabilities_never_fall_back_to_controls(self):
        training = [inventory_row(10, 0, label) for label in (0, 1, 0, 1)]
        with patch("oil_eval.models.GradientBoostingClassifier.fit", side_effect=RuntimeError("tree fit failed")):
            with self.assertRaisesRegex(RuntimeError, "tree fit failed"):
                fit_predict_inventory_tree_models(training, inventory_row(12, 2))
        for probability in (float("nan"), 1.1, -0.1):
            with patch("oil_eval.models.GradientBoostingClassifier.predict_proba",
                       return_value=np.asarray([[1 - probability, probability]])):
                with self.assertRaisesRegex(RuntimeError, "invalid probability"):
                    fit_predict_inventory_tree_models(training, inventory_row(12, 2))


if __name__ == "__main__":
    unittest.main()

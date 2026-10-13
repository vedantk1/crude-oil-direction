"""Training-only preprocessing and the fixed feature/model contract."""

from copy import deepcopy
import unittest
from unittest.mock import patch
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning

from oil_audit.features import FEATURE_COLUMNS, WTI_FEATURE_COLUMNS
from oil_eval.inventory_features import INVENTORY_FEATURE_COLUMNS, INVENTORY_MODEL_COLUMNS
from oil_eval.models import (
    COMPARISON_MODEL_NAMES, INVENTORY_MODEL_NAMES, LOGISTIC_PARAMETERS, Pipeline,
    fit_predict_comparison_models, fit_predict_inventory_models, fit_predict_models,
)


def row(price, momentum, label=0):
    return {"features": dict(zip(WTI_FEATURE_COLUMNS, [price, 2, momentum, None, None, 0.8])),
            "label": label}


def combined_row(price, momentum, label=0):
    result = row(price, momentum, label)
    result["features"].update(dict(zip(
        FEATURE_COLUMNS[len(WTI_FEATURE_COLUMNS):],
        [50, 3, momentum, None, None, 0.8, 10, 3],
    )))
    return result


def inventory_row(price, momentum, label=0):
    result = row(price, momentum, label)
    result["features"].update(dict(zip(INVENTORY_FEATURE_COLUMNS, [momentum, None, 1, None, 2])))
    return result


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.training = [row(-30, -3, 0), row(10, None, 1), row(20, 1, 0), row(40, 5, 1)]

    def test_forecast_values_do_not_fit_imputation_or_scaling(self):
        low = fit_predict_models(self.training, row(None, None))
        high = fit_predict_models(self.training, row(100000, 5000))
        diagnostics = low["logistic"]["diagnostics"]
        self.assertEqual(diagnostics, high["logistic"]["diagnostics"])
        self.assertEqual(diagnostics["imputer_statistics"]["wti_latest_price"], 15)
        self.assertEqual(diagnostics["imputer_statistics"]["wti_change_5_business_days_usd"], 1)
        self.assertEqual(diagnostics["scaler_mean"]["wti_latest_price"], 10)
        self.assertEqual(diagnostics["scaler_mean"]["wti_change_5_business_days_usd"], 1)
        self.assertEqual(diagnostics["imputer_statistics"]["wti_change_20_business_days_usd"], 0)
        self.assertEqual(diagnostics["scaler_scale"]["wti_change_20_business_days_usd"], 1)
        self.assertEqual(len(diagnostics["feature_columns"]), 6)

    def test_majority_tie_and_missing_momentum_fallback(self):
        result = fit_predict_models(self.training, row(10, None))
        self.assertEqual(result["majority"], {"prediction": 1, "probability_up": 0.5})
        self.assertEqual(result["momentum"], {
            "prediction": 1, "probability_up": None, "used_majority_fallback": True})
        self.training.append(row(0, -1, 0))
        result = fit_predict_models(self.training, row(10, None))
        self.assertEqual(result["majority"]["probability_up"], 0.4)
        self.assertEqual(result["momentum"]["prediction"], 0)

    def test_momentum_positive_negative_and_zero(self):
        for momentum, prediction in ((-0.5, 0), (0, 0), (0.5, 1)):
            with self.subTest(momentum=momentum):
                result = fit_predict_models(self.training, row(-36.98, momentum))
                self.assertEqual(result["momentum"]["prediction"], prediction)
                self.assertFalse(result["momentum"]["used_majority_fallback"])
                self.assertGreaterEqual(result["logistic"]["probability_up"], 0)
                self.assertLessEqual(result["logistic"]["probability_up"], 1)

    def test_ignored_columns_and_forecast_label_cannot_affect_predictions(self):
        forecast = row(12, 2)
        expected = fit_predict_models(self.training, forecast)
        augmented = deepcopy(self.training)
        for item in augmented:
            item["features"].update({"brent_latest_price": 1e20, "future_target": 999})
            item["target_price"] = -999
        forecast["label"] = 999
        forecast["features"].update({"label": -999, "brent_latest_price": -1e20})
        self.assertEqual(expected, fit_predict_models(augmented, forecast))

    def test_no_input_mutation(self):
        forecast = row(None, None)
        before = deepcopy((self.training, forecast))
        fit_predict_models(self.training, forecast)
        self.assertEqual((self.training, forecast), before)

    def test_empty_and_single_class_training_fail_explicitly(self):
        with self.assertRaisesRegex(ValueError, "without training"):
            fit_predict_models([], row(1, 1))
        with self.assertRaisesRegex(ValueError, "both training label classes"):
            fit_predict_models([row(1, 1, 1)], row(1, 1))

    def test_invalid_labels_and_features_are_rejected(self):
        for value in (2, 1.0, None, True):
            with self.subTest(label=value), self.assertRaisesRegex(ValueError, "labels"):
                fit_predict_models([row(1, 1, value), row(2, 2, 0)], row(1, 1))
        for value in (float("nan"), float("inf"), "2", True):
            with self.subTest(feature=value), self.assertRaisesRegex(ValueError, "finite numeric"):
                fit_predict_models(self.training, row(value, 1))
        malformed = row(1, 1)
        del malformed["features"][WTI_FEATURE_COLUMNS[0]]
        with self.assertRaisesRegex(ValueError, "Missing required"):
            fit_predict_models(self.training, malformed)

    def test_convergence_warning_becomes_failure(self):
        def failed_fit(*args, **kwargs):
            warnings.warn("did not converge", ConvergenceWarning)
        with patch("oil_eval.models.Pipeline.fit", side_effect=failed_fit):
            with self.assertRaisesRegex(RuntimeError, "did not converge"):
                fit_predict_models(self.training, row(1, 1))


class ComparisonModelTests(unittest.TestCase):
    def setUp(self):
        self.training = [combined_row(-30, -3, 0), combined_row(10, None, 1),
                         combined_row(20, 1, 0), combined_row(40, 5, 1)]
        for forecast_id, item in zip(("c", "a", "d", "b"), self.training):
            item["forecast_id"] = forecast_id

    def test_wti_results_unchanged_by_brent_training_and_forecast_values(self):
        forecast = combined_row(12, 2)
        expected = fit_predict_models(self.training, forecast)
        combined = fit_predict_comparison_models(self.training, forecast)
        self.assertEqual(tuple(combined), COMPARISON_MODEL_NAMES)
        self.assertEqual(expected, {name: combined[name] for name in expected})
        changed_training = deepcopy(self.training)
        for index, item in enumerate(changed_training + [forecast]):
            for column in FEATURE_COLUMNS[len(WTI_FEATURE_COLUMNS):]:
                item["features"][column] = 1000 * (index + 1)
        changed = fit_predict_comparison_models(changed_training, forecast)
        self.assertEqual(expected, {name: changed[name] for name in expected})

    def test_combined_uses_all_fourteen_columns_and_aligned_spread(self):
        training = [combined_row(10, 0, int(spread > 0)) for spread in (-5, -3, 3, 5)]
        for item, spread in zip(training, (-5, -3, 3, 5)):
            item["features"]["brent_wti_spread_usd"] = spread
        low, high = combined_row(10, 0), combined_row(10, 0)
        low["features"]["brent_wti_spread_usd"] = -4
        high["features"]["brent_wti_spread_usd"] = 4
        low_result = fit_predict_comparison_models(training, low)
        high_result = fit_predict_comparison_models(training, high)
        diagnostics = low_result["logistic_wti_brent"]["diagnostics"]
        self.assertEqual(diagnostics["feature_columns"], list(FEATURE_COLUMNS))
        self.assertEqual(len(diagnostics["feature_columns"]), 14)
        self.assertGreater(diagnostics["coef"]["brent_wti_spread_usd"], 0)
        self.assertIn("spread_age_calendar_days", diagnostics["coef"])
        self.assertLess(low_result["logistic_wti_brent"]["probability_up"],
                        high_result["logistic_wti_brent"]["probability_up"])
        self.assertEqual(low_result["logistic"], high_result["logistic"])

    def test_both_fits_use_identical_ordered_samples_labels_and_parameters(self):
        calls = []
        original_fit = Pipeline.fit

        def capture_fit(model, train, labels, *args, **kwargs):
            calls.append((train.copy(), list(labels),
                          model.named_steps["classifier"].get_params()))
            return original_fit(model, train, labels, *args, **kwargs)

        with patch("oil_eval.models.Pipeline.fit", new=capture_fit):
            fit_predict_comparison_models(self.training, combined_row(12, 2))
        self.assertEqual(len(calls), 2)
        wti, combined = calls
        np.testing.assert_allclose(wti[0], combined[0][:, :6], equal_nan=True)
        self.assertEqual(wti[0][:, 0].tolist(), [-30, 10, 20, 40])
        self.assertEqual(wti[1], [0, 1, 0, 1])
        self.assertEqual(wti[1], combined[1])
        self.assertEqual(wti[0].shape, (4, 6))
        self.assertEqual(combined[0].shape, (4, 14))
        for _, _, parameters in calls:
            self.assertEqual({key: parameters[key] for key in LOGISTIC_PARAMETERS},
                             LOGISTIC_PARAMETERS)
        self.assertEqual([item["forecast_id"] for item in self.training], ["c", "a", "d", "b"])

    def test_brent_nulls_imputed_only_from_training(self):
        low, high = combined_row(None, None), combined_row(100000, 5000)
        for column in FEATURE_COLUMNS[len(WTI_FEATURE_COLUMNS):]:
            low["features"][column] = None
            high["features"][column] = 100000
        low_result = fit_predict_comparison_models(self.training, low)
        high_result = fit_predict_comparison_models(self.training, high)
        for name in ("logistic", "logistic_wti_brent"):
            self.assertEqual(low_result[name]["diagnostics"], high_result[name]["diagnostics"])
        diagnostics = low_result["logistic_wti_brent"]["diagnostics"]
        self.assertEqual(diagnostics["imputer_statistics"]["brent_latest_price"], 50)
        self.assertEqual(diagnostics["imputer_statistics"]["brent_change_5_business_days_usd"], 1)
        self.assertEqual(diagnostics["imputer_statistics"]["brent_change_20_business_days_usd"], 0)
        self.assertEqual(diagnostics["scaler_scale"]["brent_change_20_business_days_usd"], 1)

    def test_outcome_metadata_and_extra_columns_cannot_change_either_fit(self):
        forecast = combined_row(12, 2)
        expected = fit_predict_comparison_models(self.training, forecast)
        augmented = deepcopy(self.training)
        for item in augmented + [forecast]:
            item["features"].update({"future_target": 1e20, "label": -1e20})
            item["target_price"] = 123456
            item["future_label"] = 999
        forecast["label"] = -999
        self.assertEqual(expected, fit_predict_comparison_models(augmented, forecast))

    def test_combined_does_not_mutate_inputs(self):
        forecast = combined_row(None, None)
        before = deepcopy((self.training, forecast))
        fit_predict_comparison_models(self.training, forecast)
        self.assertEqual((self.training, forecast), before)

    def test_invalid_or_absent_brent_columns_fail_explicitly(self):
        for value in (float("nan"), float("inf"), "2", True):
            forecast = combined_row(12, 2)
            forecast["features"]["brent_wti_spread_usd"] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "finite numeric"):
                fit_predict_comparison_models(self.training, forecast)
        forecast = combined_row(12, 2)
        del forecast["features"]["spread_age_calendar_days"]
        with self.assertRaisesRegex(ValueError, "Missing required WTI/Brent feature"):
            fit_predict_comparison_models(self.training, forecast)


class InventoryModelTests(unittest.TestCase):
    def setUp(self):
        self.training = [inventory_row(-30, -3, 0), inventory_row(10, None, 1),
                         inventory_row(20, 1, 0), inventory_row(40, 5, 1)]
        for row_id, item in zip(("c", "a", "d", "b"), self.training):
            item["row_id"] = row_id

    def test_inventory_fit_reproduces_wti_and_baselines_exactly(self):
        forecast = inventory_row(12, 2)
        expected = fit_predict_models(self.training, forecast)
        result = fit_predict_inventory_models(self.training, forecast)
        self.assertEqual(tuple(result), INVENTORY_MODEL_NAMES)
        self.assertEqual(expected, {name: result[name] for name in expected})
        changed = deepcopy(self.training)
        for index, item in enumerate(changed + [forecast]):
            for column in INVENTORY_FEATURE_COLUMNS:
                item["features"][column] = 100 * (index + 1)
        result = fit_predict_inventory_models(changed, forecast)
        self.assertEqual(expected, {name: result[name] for name in expected})
        self.assertNotIn("training_missing_counts", result["logistic"]["diagnostics"])
        self.assertNotIn("forecast_missing_columns", result["logistic"]["diagnostics"])

    def test_missing_inventory_preserves_the_same_ordered_training_cohort(self):
        for column in INVENTORY_FEATURE_COLUMNS:
            self.training[0]["features"][column] = None
        calls = []
        original_fit = Pipeline.fit

        def capture_fit(model, train, labels, *args, **kwargs):
            calls.append((train.copy(), list(labels),
                          model.named_steps["classifier"].get_params(),
                          model.named_steps["imputer"].add_indicator))
            return original_fit(model, train, labels, *args, **kwargs)

        with patch("oil_eval.models.Pipeline.fit", new=capture_fit):
            fit_predict_inventory_models(self.training, inventory_row(12, 2))
        self.assertEqual(len(calls), 2)
        wti, inventory = calls
        self.assertEqual(wti[0].shape, (4, 6))
        self.assertEqual(inventory[0].shape, (4, 11))
        np.testing.assert_allclose(wti[0], inventory[0][:, :6], equal_nan=True)
        self.assertTrue(np.isnan(inventory[0][0, 6:]).all())
        self.assertEqual(wti[0][:, 0].tolist(), [-30, 10, 20, 40])
        self.assertEqual(wti[1], [0, 1, 0, 1])
        self.assertEqual(wti[1], inventory[1])
        for _, _, parameters, missing_indicators in calls:
            self.assertEqual({key: parameters[key] for key in LOGISTIC_PARAMETERS}, LOGISTIC_PARAMETERS)
            self.assertFalse(missing_indicators)
        self.assertEqual([item["row_id"] for item in self.training], ["c", "a", "d", "b"])

    def test_forecast_inventory_does_not_fit_preprocessing_or_coefficients(self):
        low, high = inventory_row(12, 2), inventory_row(12, 2)
        for column in INVENTORY_FEATURE_COLUMNS:
            low["features"][column] = None
            high["features"][column] = 100000
        low_result = fit_predict_inventory_models(self.training, low)
        high_result = fit_predict_inventory_models(self.training, high)
        left = low_result["logistic_wti_inventory"]["diagnostics"]
        right = high_result["logistic_wti_inventory"]["diagnostics"]
        self.assertEqual({key: value for key, value in left.items() if key != "forecast_missing_columns"},
                         {key: value for key, value in right.items() if key != "forecast_missing_columns"})
        self.assertTrue(set(INVENTORY_FEATURE_COLUMNS).issubset(left["forecast_missing_columns"]))
        self.assertFalse(set(INVENTORY_FEATURE_COLUMNS).intersection(right["forecast_missing_columns"]))
        weekly, four_week = INVENTORY_FEATURE_COLUMNS[:2]
        self.assertEqual(left["imputer_statistics"][weekly], 1)
        self.assertEqual(left["training_missing_counts"][weekly], 1)
        self.assertEqual(left["imputer_statistics"][four_week], 0)
        self.assertEqual(left["scaler_scale"][four_week], 1)
        self.assertEqual(left["training_missing_counts"][four_week], len(self.training))

    def test_missingness_diagnostics_cover_all_eleven_inputs_without_extra_features(self):
        forecast = inventory_row(None, None)
        result = fit_predict_inventory_models(self.training, forecast)
        diagnostics = result["logistic_wti_inventory"]["diagnostics"]
        self.assertEqual(tuple(diagnostics["feature_columns"]), INVENTORY_MODEL_COLUMNS)
        self.assertEqual(len(diagnostics["coef"]), 11)
        self.assertEqual(tuple(diagnostics["imputer_statistics"]), INVENTORY_MODEL_COLUMNS)
        self.assertEqual(tuple(diagnostics["training_missing_counts"]), INVENTORY_MODEL_COLUMNS)
        self.assertEqual(diagnostics["forecast_missing_columns"], [
            column for column in INVENTORY_MODEL_COLUMNS if forecast["features"][column] is None])
        self.assertEqual(diagnostics["training_missing_counts"]["wti_latest_price"], 0)
        self.assertEqual(diagnostics["training_missing_counts"]["wti_change_5_business_days_usd"], 1)

    def test_inventory_signal_can_change_combined_prediction_without_changing_wti(self):
        column = INVENTORY_FEATURE_COLUMNS[0]
        training = [inventory_row(10, 0, int(change > 0)) for change in (-5, -3, 3, 5)]
        for item, change in zip(training, (-5, -3, 3, 5)):
            item["features"][column] = change
        low, high = inventory_row(10, 0), inventory_row(10, 0)
        low["features"][column], high["features"][column] = -4, 4
        low_result = fit_predict_inventory_models(training, low)
        high_result = fit_predict_inventory_models(training, high)
        self.assertEqual(low_result["logistic"], high_result["logistic"])
        self.assertGreater(low_result["logistic_wti_inventory"]["diagnostics"]["coef"][column], 0)
        self.assertLess(low_result["logistic_wti_inventory"]["probability_up"],
                        high_result["logistic_wti_inventory"]["probability_up"])

    def test_future_columns_and_outcome_metadata_are_not_model_inputs(self):
        forecast = inventory_row(12, 2)
        expected = fit_predict_inventory_models(self.training, forecast)
        augmented = deepcopy(self.training)
        for item in augmented + [forecast]:
            item["features"].update({"future_inventory_change": 1e20, "label": -1e20,
                                     "brent_latest_price": 1e20})
            item["target_price"] = 9999
            item["inventory_provenance"] = {"future_revision": 9999}
        forecast["label"] = 999
        self.assertEqual(expected, fit_predict_inventory_models(augmented, forecast))

    def test_invalid_or_absent_inventory_inputs_fail_for_training_and_forecast(self):
        column = INVENTORY_FEATURE_COLUMNS[0]
        for malformed_training in (False, True):
            for value in (float("nan"), float("inf"), "2", True):
                training, forecast = deepcopy(self.training), inventory_row(12, 2)
                target = training[0] if malformed_training else forecast
                target["features"][column] = value
                with self.subTest(training=malformed_training, value=value):
                    with self.assertRaisesRegex(ValueError, "WTI/inventory feature.*finite numeric"):
                        fit_predict_inventory_models(training, forecast)
        forecast = inventory_row(12, 2)
        del forecast["features"][column]
        with self.assertRaisesRegex(ValueError, "Missing required WTI/inventory feature"):
            fit_predict_inventory_models(self.training, forecast)

    def test_inventory_fit_does_not_mutate_inputs(self):
        forecast = inventory_row(None, None)
        before = deepcopy((self.training, forecast))
        fit_predict_inventory_models(self.training, forecast)
        self.assertEqual((self.training, forecast), before)


if __name__ == "__main__":
    unittest.main()

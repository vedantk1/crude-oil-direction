"""Training-only preprocessing and the fixed feature/model contract."""

from copy import deepcopy
import unittest
from unittest.mock import patch
import warnings

from sklearn.exceptions import ConvergenceWarning

from oil_audit.features import WTI_FEATURE_COLUMNS
from oil_eval.models import fit_predict_models


def row(price, momentum, label=0):
    return {"features": dict(zip(WTI_FEATURE_COLUMNS, [price, 2, momentum, None, None, 0.8])),
            "label": label}


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


if __name__ == "__main__":
    unittest.main()

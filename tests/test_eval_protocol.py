import unittest

from oil_eval.protocol import forecast_role, protocol_config


class ProtocolTests(unittest.TestCase):
    def test_boundaries_exclude_holdout_and_crossing_targets(self):
        for forecast, target, expected in (
            ("2016-12-29", "2017-01-06", "initial_training"),
            ("2017-01-05", "2017-01-12", "development"),
            ("2022-12-29", "2023-01-06", "crosses_holdout_boundary"),
            ("2023-01-05", "2023-01-12", "holdout"),
        ):
            self.assertEqual(forecast_role({"forecast_date": forecast, "target_date": target}), expected)

    def test_configuration_records_fixed_decisions_before_results(self):
        config = protocol_config()
        self.assertEqual(config["holdout_start"], "2023-01-01")
        self.assertEqual(config["development_scoring_as_of"], "2022-12-31")
        self.assertEqual(len(config["feature_columns"]), 6)
        self.assertEqual(config["logistic"]["C"], 1.0)
        self.assertEqual(config["minimum_training_rows"], 150)

    def test_development_rejects_invalid_target_order(self):
        with self.assertRaises(ValueError):
            forecast_role({"forecast_date": "2020-01-02", "target_date": "2020-01-01"})

    def test_brent_protocol_keeps_dates_and_parameters_fixed(self):
        original = protocol_config()
        comparison = protocol_config("brent-comparison")
        for key in ("initial_training_start", "development_start", "development_scoring_as_of",
                    "holdout_start", "logistic", "minimum_training_rows"):
            self.assertEqual(original[key], comparison[key])
        self.assertEqual(len(comparison["compared_feature_sets"]["logistic"]), 6)
        self.assertEqual(len(comparison["compared_feature_sets"]["logistic_wti_brent"]), 14)
        with self.assertRaises(ValueError):
            protocol_config("holdout")

    def test_inventory_protocol_preserves_original_cohorts_and_parameters(self):
        original, inventory = protocol_config(), protocol_config("inventory-comparison")
        for key in ("initial_training_start", "development_start", "development_scoring_as_of",
                    "holdout_start", "logistic", "minimum_training_rows"):
            self.assertEqual(original[key], inventory[key])
        self.assertEqual(len(inventory["compared_feature_sets"]["logistic_wti_inventory"]), 11)
        self.assertIn("missing inventory never removes", inventory["training_cohort"])
        self.assertIn("no older-issue or PDF fallback", inventory["unusable_inventory_issue"])

    def test_regularization_is_one_retrospective_candidate_with_unchanged_controls(self):
        original = protocol_config("inventory-comparison")
        candidate = protocol_config("inventory-regularization")
        for key in ("initial_training_start", "development_start", "development_scoring_as_of",
                    "holdout_start", "logistic", "minimum_training_rows", "feature_columns",
                    "inventory_cutoff", "inventory_missing", "unusable_inventory_issue",
                    "training_cohort", "forecast_cohort"):
            self.assertEqual(original[key], candidate[key])
        variants = candidate["logistic_variants"]
        self.assertEqual(variants["logistic"], original["logistic"])
        self.assertEqual(variants["logistic_wti_inventory"], original["logistic"])
        self.assertEqual(variants["logistic_wti_inventory_c01"], {**original["logistic"], "C": 0.1})
        self.assertEqual(candidate["compared_feature_sets"]["logistic_wti_inventory_c01"],
                         original["compared_feature_sets"]["logistic_wti_inventory"])
        self.assertIn("retrospective single predefined", candidate["candidate_selection"])
        self.assertIn("Brier score and log loss", candidate["primary_comparison"])
        self.assertIn("without a formal success threshold or significance claim", candidate["success_criteria"])
        self.assertIn("no grid, feature, threshold, date or holdout changes", candidate["tuning"])

    def test_window_changes_only_candidate_history_and_preserves_control_configuration(self):
        original = protocol_config("inventory-regularization")
        candidate = protocol_config("inventory-window-comparison")
        for key in ("initial_training_start", "development_start", "development_scoring_as_of",
                    "holdout_start", "logistic", "minimum_training_rows", "feature_columns",
                    "inventory_cutoff", "inventory_missing", "unusable_inventory_issue", "forecast_cohort"):
            self.assertEqual(original[key], candidate[key])
        for name, parameters in original["logistic_variants"].items():
            self.assertEqual(candidate["logistic_variants"][name], parameters)
        self.assertEqual(candidate["logistic_variants"]["logistic_wti_inventory_c01_rolling"],
                         original["logistic_variants"]["logistic_wti_inventory_c01"])
        self.assertEqual(candidate["compared_feature_sets"]["logistic_wti_inventory_c01_rolling"],
                         original["compared_feature_sets"]["logistic_wti_inventory_c01"])
        self.assertEqual(candidate["rolling_window_years"], 4)
        self.assertEqual(candidate["minimum_training_rows"], 150)
        self.assertIn("capacity-only audit without model predictions", candidate["candidate_selection"])
        self.assertIn("no duration grid", candidate["tuning"])
        self.assertIn("without a formal success threshold or significance claim", candidate["success_criteria"])

    def test_fixed_small_tree_uses_expanding_inventory_protocol_and_unchanged_controls(self):
        original = protocol_config("inventory-regularization")
        candidate = protocol_config("inventory-tree-comparison")
        for key in ("initial_training_start", "development_start", "development_scoring_as_of",
                    "holdout_start", "logistic", "logistic_variants", "minimum_training_rows",
                    "feature_columns", "inventory_cutoff", "inventory_missing", "unusable_inventory_issue",
                    "forecast_cohort", "training_cohort", "fit_schedule"):
            self.assertEqual(original[key], candidate[key])
        self.assertEqual(candidate["compared_feature_sets"]["tree_wti_inventory"],
                         original["compared_feature_sets"]["logistic_wti_inventory_c01"])
        self.assertNotIn("rolling_window_years", candidate)
        self.assertNotIn("logistic_wti_inventory_c01_rolling", candidate["compared_feature_sets"])
        tree = candidate["tree"]
        self.assertIsNone(tree["scaler"])
        self.assertFalse(tree["missing_indicators"])
        for key in ("sample_weight", "class_weight", "calibration"):
            self.assertIsNone(tree[key])
        self.assertEqual(tree["threshold"], 0.5)
        self.assertEqual(tree["tie_prediction"], 1)
        for key, value in {"n_estimators": 50, "learning_rate": 0.05, "max_depth": 2,
                           "min_samples_leaf": 30, "subsample": 1.0, "random_state": 0,
                           "loss": "log_loss", "n_iter_no_change": None, "warm_start": False}.items():
            self.assertEqual(tree["parameters"][key], value)
        self.assertIn("no validation subset", tree["early_stopping"])
        self.assertIn("without causal interpretation", tree["feature_importance"])
        self.assertIn("no grid, calibration", candidate["tuning"])
        self.assertIn("without a formal success threshold or significance claim", candidate["success_criteria"])

    def test_availability_delay_keeps_model_and_targets_fixed_with_one_extra_information_day(self):
        original = protocol_config("inventory-regularization")
        delayed = protocol_config("inventory-availability-delay")
        for key in ("initial_training_start", "development_start", "development_scoring_as_of",
                    "holdout_start", "logistic", "minimum_training_rows", "feature_columns", "scoring_labels"):
            self.assertEqual(original[key], delayed[key])
        self.assertEqual(delayed["logistic_variants"]["logistic_wti_inventory_c01_delayed"],
                         original["logistic_variants"]["logistic_wti_inventory_c01"])
        self.assertEqual(delayed["additional_information_delay_business_days"], 1)
        self.assertIn("P is the previous US business day", delayed["delayed_information_clock"])
        self.assertIn("including revisions and withdrawals", delayed["delayed_training_labels"])
        self.assertIn("fail rather than drop", delayed["forecast_cohort"])
        self.assertIn("before value decoding", delayed["source_admission"])
        self.assertIn("never shift forecast dates", delayed["delayed_features"])
        self.assertIn("no lag grid", delayed["tuning"])

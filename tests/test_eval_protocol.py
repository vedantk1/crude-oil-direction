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

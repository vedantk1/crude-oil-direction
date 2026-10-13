"""Synthetic chronological replay tests never open the real holdout dataset."""

from copy import deepcopy
from datetime import date
import unittest

from oil_audit.calendar import add_business_days, next_business_day
from oil_audit.dataset import json_bytes, sha256
from oil_audit.features import FEATURE_COLUMNS, WTI_FEATURE_COLUMNS
from oil_eval.protocol import MIN_TRAINING_ROWS
from oil_eval.replay import replay_development


def feature_row(forecast, index=0):
    return {
        "row_id": "wti-direction-" + forecast.strftime("%Y%m%d"),
        "forecast_date": forecast.isoformat(),
        "target_date": add_business_days(forecast, 5).isoformat(),
        "features": dict(zip(WTI_FEATURE_COLUMNS, [
            30 + index % 30, 2, (index % 7) - 3, (index % 11) - 5,
            None if index % 4 == 0 else 2, 0.9,
        ])),
        "eligibility": {"wti": True},
    }


def event_history(row, label, available=None):
    available = available or next_business_day(date.fromisoformat(row["target_date"]))
    identity = {key: row[key] for key in ("row_id", "forecast_date", "target_date")}
    missing = {
        **identity, "available_from": row["forecast_date"],
        "available_until": available.isoformat(), "label_status": "missing_both",
        "label": None, "origin_price": None, "target_price": None,
        "origin_vintage_start": None, "target_vintage_start": None,
    }
    known = {
        **identity, "available_from": available.isoformat(), "available_until": None,
        "label_status": "available", "label": label,
        "origin_price": 30, "target_price": 31 if label else 29,
        "origin_vintage_start": row["forecast_date"],
        "target_vintage_start": row["target_date"],
    }
    return [missing, known]


def fixture(warmup=160, development_dates=None):
    development_dates = development_dates or [date(2017, 1, 5), date(2017, 1, 19), date(2017, 2, 2)]
    rows = [feature_row(add_business_days(date(2012, 1, 5), 5 * index), index)
            for index in range(warmup)]
    rows += [feature_row(day, warmup + index) for index, day in enumerate(development_dates)]
    events = [event for index, row in enumerate(rows) for event in event_history(row, index % 2)]
    scoring = [deepcopy(event) for event in events
               if event["forecast_date"] >= "2017-01-01" and event["label_status"] == "available"]
    return rows, events, scoring


def capturing_predictor(calls):
    def predict(training, forecast):
        calls.append((deepcopy(training), deepcopy(forecast)))
        return {
            "majority": {"prediction": 1, "probability_up": 0.5},
            "momentum": {"prediction": 0, "probability_up": None,
                         "used_majority_fallback": False},
            "logistic": {"prediction": 1, "probability_up": 0.6,
                         "diagnostics": {"iterations": 1}},
        }
    return predict


def comparison_fixture():
    rows, events, scoring = fixture()
    for index, row in enumerate(rows):
        row["features"].update({
            "brent_latest_price": 35 + index % 30,
            "brent_age_calendar_days": 3,
            "brent_change_5_business_days_usd": (index % 5) - 2,
            "brent_change_20_business_days_usd": (index % 13) - 6,
            "brent_change_std_20_business_days_usd": None if index % 3 == 0 else 3,
            "brent_coverage_20_business_days": 0.85,
            "brent_wti_spread_usd": 5,
            "spread_age_calendar_days": 3,
        })
        row["eligibility"]["wti_brent"] = True
    return rows, events, scoring


def capturing_comparison_predictor(calls):
    wti_predictor = capturing_predictor(calls)

    def predict(training, forecast):
        models = wti_predictor(training, forecast)
        models["logistic_wti_brent"] = {
            "prediction": 0, "probability_up": 0.4, "diagnostics": {"iterations": 1},
        }
        return models
    return predict


class ReplayTests(unittest.TestCase):
    def test_future_development_data_and_revisions_cannot_change_earlier_fits(self):
        rows, events, scoring = fixture()
        expected, _ = replay_development(rows, events, scoring)
        modified_rows, modified_events = deepcopy(rows), deepcopy(events)
        modified_rows[-1]["features"]["wti_latest_price"] = 10000
        revised_id = modified_rows[0]["row_id"]
        original = next(event for event in modified_events
                        if event["row_id"] == revised_id and event["label_status"] == "available")
        original["available_until"] = "2017-01-20"
        modified_events.append({**original, "available_from": "2017-01-20",
                                "available_until": None, "label": 1,
                                "target_price": 31})
        future = feature_row(date(2017, 2, 16), 500)
        modified_rows.append(future)
        future_missing = event_history(future, 1)[0]
        future_missing["available_until"] = None
        modified_events.append(future_missing)
        actual, _ = replay_development(modified_rows, modified_events, scoring)
        self.assertEqual(expected[:2], actual[:2])
        self.assertNotEqual(expected[2]["training"]["input_sha256"],
                            actual[2]["training"]["input_sha256"])
        self.assertEqual(len(actual), len(expected) + 1)
        self.assertEqual(actual[-1]["scoring_status"], "unavailable_by_development_cutoff")

    def test_only_past_mature_currently_available_labels_enter_training(self):
        rows, events, scoring = fixture(development_dates=[date(2017, 1, 5), date(2017, 1, 19)])
        recent = feature_row(date(2016, 12, 29), 500)
        rows.append(recent)
        events += event_history(recent, 1, date(2017, 1, 10))
        withdrawn_id = rows[0]["row_id"]
        original = next(event for event in events
                        if event["row_id"] == withdrawn_id and event["label_status"] == "available")
        original["available_until"] = "2017-01-10"
        events.append({**original, "available_from": "2017-01-10", "available_until": None,
                       "label_status": "missing_target", "label": None, "target_price": None})
        calls = []
        predictions, _ = replay_development(rows, events, scoring, predictor=capturing_predictor(calls))
        first_ids = {row["row_id"] for row in calls[0][0]}
        later_ids = {row["row_id"] for row in calls[1][0]}
        self.assertIn(withdrawn_id, first_ids)
        self.assertNotIn(withdrawn_id, later_ids)
        self.assertNotIn(recent["row_id"], first_ids)
        self.assertIn(recent["row_id"], later_ids)
        self.assertNotIn(predictions[0]["row_id"], first_ids)
        self.assertIn(predictions[0]["row_id"], later_ids)
        for training, forecast in calls:
            self.assertTrue(all(row["forecast_date"] < forecast["forecast_date"] for row in training))
        for prediction in predictions:
            self.assertLess(prediction["training"]["latest_target_date"], prediction["forecast_date"])
            self.assertLessEqual(prediction["training"]["latest_label_available_from"],
                                 prediction["forecast_date"])

    def test_target_crossing_holdout_boundary_is_purged_before_prediction(self):
        rows, events, scoring = fixture()
        crossing = feature_row(date(2022, 12, 27))
        self.assertGreaterEqual(crossing["target_date"], "2023-01-01")
        rows.append(crossing)
        calls = []
        predictions, summary = replay_development(rows, events, scoring, predictor=capturing_predictor(calls))
        self.assertEqual(len(calls), 3)
        self.assertNotIn(crossing["row_id"], {row["row_id"] for row in predictions})
        self.assertEqual(summary["excluded_pre_holdout_rows"]["crosses_holdout_boundary"], 1)
        self.assertFalse(summary["holdout_evaluated"])

    def test_direct_holdout_feature_or_label_inputs_fail_before_prediction(self):
        rows, events, scoring = fixture()
        holdout = feature_row(date(2023, 1, 3))
        variants = [
            (rows + [holdout], events, scoring),
            (rows, events + event_history(holdout, 1), scoring),
            (rows, events + [{**events[-1], "available_from": "2023-01-03"}], scoring),
            (rows, events, scoring + [{**scoring[-1], "available_from": "2023-01-03"}]),
        ]
        for index, variant in enumerate(variants):
            calls = []
            with self.subTest(variant=index), self.assertRaisesRegex(ValueError, "Holdout"):
                replay_development(*variant, predictor=capturing_predictor(calls))
            self.assertEqual(calls, [])

    def test_unavailable_outcomes_remain_predicted_but_use_identical_scoring_dates(self):
        rows, events, scoring = fixture()
        unavailable_id = scoring[-1]["row_id"]
        events = [event for event in events
                  if event["row_id"] != unavailable_id or event["label_status"] != "available"]
        for event in events:
            if event["row_id"] == unavailable_id:
                event["available_until"] = None
        scoring = scoring[:2]
        predictions, summary = replay_development(rows, events, scoring, predictor=capturing_predictor([]))
        self.assertEqual((summary["prediction_count"], summary["scored_count"], summary["unscored_count"]),
                         (3, 2, 1))
        self.assertIsNone(predictions[-1]["label"])
        self.assertEqual(predictions[-1]["scoring_status"], "unavailable_by_development_cutoff")
        self.assertEqual({metrics["count"] for metrics in summary["metrics"].values()}, {2})
        for comparison in summary["paired_accuracy_comparisons"].values():
            self.assertEqual(sum(comparison[name] for name in (
                "both_correct", "both_wrong", "logistic_only_correct", "baseline_only_correct")), 2)

    def test_minimum_training_history_and_classes_are_required(self):
        rows, events, scoring = fixture(warmup=MIN_TRAINING_ROWS - 1)
        calls = []
        with self.assertRaisesRegex(ValueError, "Insufficient eligible training history"):
            replay_development(rows, events, scoring, predictor=capturing_predictor(calls))
        self.assertEqual(calls, [])
        rows, events, scoring = fixture()
        for event in events:
            if event["label_status"] == "available":
                event.update(label=0, target_price=29)
        with self.assertRaisesRegex(ValueError, "Both training classes"):
            replay_development(rows, events, scoring, predictor=capturing_predictor(calls))
        self.assertEqual(calls, [])

    def test_duplicate_forecast_ids_and_dates_fail(self):
        rows, events, scoring = fixture()
        duplicates = [(deepcopy(rows[-1]), "IDs"),
                      ({**deepcopy(rows[-1]), "row_id": "different-id"}, "dates")]
        for duplicate, message in duplicates:
            calls = []
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                replay_development(rows + [duplicate], events, scoring,
                                   predictor=capturing_predictor(calls))
            self.assertEqual(calls, [])

    def test_available_but_unmatured_training_label_is_an_error(self):
        rows, events, scoring = fixture(development_dates=[date(2017, 1, 5)])
        recent = feature_row(date(2017, 1, 4))
        rows.append(recent)
        # An impossible early label must fail rather than train on a future target.
        events += event_history(recent, 1, date(2017, 1, 5))
        with self.assertRaisesRegex(ValueError, "Training-label maturity violation"):
            replay_development(rows, events, scoring, predictor=capturing_predictor([]))


class BrentComparisonReplayTests(unittest.TestCase):
    def setUp(self):
        self.rows, self.events, self.scoring = comparison_fixture()
        self.reference, _ = replay_development(
            self.rows, self.events, self.scoring, predictor=capturing_predictor([]))

    def replay(self, calls=None, **kwargs):
        return replay_development(
            self.rows, self.events, self.scoring,
            predictor=capturing_comparison_predictor(calls if calls is not None else []),
            experiment="brent-comparison", reference_rows=self.reference, **kwargs)

    def test_comparison_requires_original_reference_before_any_predictor_call(self):
        calls = []
        with self.assertRaisesRegex(ValueError, "reference|Reference"):
            replay_development(
                self.rows, self.events, self.scoring,
                predictor=capturing_comparison_predictor(calls), experiment="brent-comparison")
        self.assertEqual(calls, [])

    def test_late_ineligible_comparison_forecast_fails_before_first_fit(self):
        self.rows[-1]["features"]["brent_latest_price"] = None
        self.rows[-1]["eligibility"]["wti_brent"] = False
        calls = []
        with self.assertRaisesRegex(ValueError, "eligible Brent"):
            self.replay(calls)
        self.assertEqual(calls, [])

    def test_changed_scoring_or_forecast_schedule_fails_before_first_fit(self):
        variants = []
        changed_scoring = deepcopy(self.scoring)
        changed_scoring[-1]["label"] = 1 - changed_scoring[-1]["label"]
        variants.append((self.rows, self.events, changed_scoring))
        variants.append((self.rows, self.events, self.scoring[:-1]))
        changed_rows = deepcopy(self.rows)
        changed_rows[-1]["forecast_date"] = "2017-02-03"
        variants.append((changed_rows, self.events, self.scoring))
        changed_rows = deepcopy(self.rows)
        changed_rows[-1]["target_date"] = "2017-02-10"
        variants.append((changed_rows, self.events, self.scoring))
        for index, variant in enumerate(variants):
            calls = []
            with self.subTest(variant=index), self.assertRaisesRegex(ValueError, "reference|Reference"):
                replay_development(
                    *variant, predictor=capturing_comparison_predictor(calls),
                    experiment="brent-comparison", reference_rows=self.reference)
            self.assertEqual(calls, [])

    def test_common_historical_cohort_excludes_ineligible_brent_for_all_models(self):
        removed_id = self.rows[0]["row_id"]
        self.rows[0]["features"]["brent_wti_spread_usd"] = None
        self.rows[0]["eligibility"]["wti_brent"] = False
        calls = []
        predictions, summary = self.replay(calls)
        for (training, _), prediction in zip(calls, predictions):
            self.assertNotIn(removed_id, {row["row_id"] for row in training})
            self.assertTrue(all(tuple(row["features"]) == FEATURE_COLUMNS for row in training))
            self.assertEqual(prediction["training"]["wti_eligible_count"], len(training) + 1)
            self.assertEqual(prediction["training"]["rows_removed_vs_wti_eligible"], 1)
            projected = [{**row, "features": {name: row["features"][name]
                                             for name in WTI_FEATURE_COLUMNS}} for row in training]
            self.assertEqual(prediction["training"]["input_sha256"], sha256(json_bytes(projected)))
            self.assertEqual(prediction["training"]["combined_input_sha256"], sha256(json_bytes(training)))
            self.assertEqual(set(prediction["models"]),
                             {"majority", "momentum", "logistic", "logistic_wti_brent"})
        self.assertFalse(summary["reference_reproduction"]["all_wti_training_inputs_identical"])
        self.assertTrue(summary["reference_reproduction"]["forecast_cohort_identical"])
        self.assertEqual({metrics["count"] for metrics in summary["metrics"].values()}, {3})

    def test_optional_brent_nulls_remain_in_common_training_and_forecasts(self):
        columns = ("brent_change_5_business_days_usd", "brent_change_20_business_days_usd",
                   "brent_change_std_20_business_days_usd")
        for row in self.rows:
            row["features"].update(dict.fromkeys(columns))
        calls = []
        predictions, summary = self.replay(calls)
        for training, forecast in calls:
            self.assertTrue(all(row["features"][name] is None
                                for row in training + [forecast] for name in columns))
        self.assertTrue(all(row["training"]["rows_removed_vs_wti_eligible"] == 0 for row in predictions))
        self.assertTrue(summary["reference_reproduction"]["all_wti_training_inputs_identical"])
        self.assertTrue(summary["reference_reproduction"]["all_reference_model_outputs_identical"])

    def test_future_brent_changes_do_not_change_earlier_predictions(self):
        reference, _ = replay_development(self.rows, self.events, self.scoring)
        expected, _ = replay_development(
            self.rows, self.events, self.scoring,
            experiment="brent-comparison", reference_rows=reference)
        self.rows[-2]["features"]["brent_latest_price"] = 10000
        self.rows[-2]["features"]["brent_change_5_business_days_usd"] = -10000
        actual, summary = replay_development(
            self.rows, self.events, self.scoring,
            experiment="brent-comparison", reference_rows=reference)
        self.assertEqual(expected[0], actual[0])
        self.assertEqual(expected[1]["training"], actual[1]["training"])
        self.assertNotEqual(expected[2]["training"]["combined_input_sha256"],
                            actual[2]["training"]["combined_input_sha256"])
        self.assertEqual(expected[2]["training"]["input_sha256"], actual[2]["training"]["input_sha256"])
        self.assertTrue(summary["reference_reproduction"]["all_reference_model_outputs_identical"])

    def test_direct_holdout_features_and_label_information_fail_before_any_fit(self):
        holdout = feature_row(date(2023, 1, 3))
        variants = [
            (self.rows + [holdout], self.events, self.scoring),
            (self.rows, self.events + event_history(holdout, 1), self.scoring),
            (self.rows, self.events + [{**self.events[-1], "available_from": "2023-01-03"}], self.scoring),
            (self.rows, self.events, self.scoring + [{**self.scoring[-1], "available_from": "2023-01-03"}]),
        ]
        for index, variant in enumerate(variants):
            calls = []
            with self.subTest(variant=index), self.assertRaisesRegex(ValueError, "Holdout"):
                replay_development(
                    *variant, predictor=capturing_comparison_predictor(calls),
                    experiment="brent-comparison", reference_rows=self.reference)
            self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()

"""Synthetic chronological replay tests never open the real holdout dataset."""

from copy import deepcopy
from datetime import date
import unittest

from oil_audit.calendar import add_business_days, next_business_day
from oil_audit.features import WTI_FEATURE_COLUMNS
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


if __name__ == "__main__":
    unittest.main()

"""Synthetic paired inventory replay tests; no real dataset or holdout reads."""

from copy import deepcopy
from datetime import date
import unittest

from oil_audit.dataset import json_bytes, sha256
from oil_audit.features import WTI_FEATURE_COLUMNS
from oil_eval.inventory_features import INVENTORY_FEATURE_COLUMNS, INVENTORY_MODEL_COLUMNS
from oil_eval.models import MODEL_NAMES
from oil_eval.replay import replay_development
from tests.test_eval_replay import capturing_predictor, event_history, feature_row, fixture


def inventory_fixture(development_dates=None):
    rows, events, scoring = fixture(development_dates=development_dates)
    original_first_id = rows[0]["row_id"]
    rows[0] = feature_row(date(2011, 4, 7))
    events = event_history(rows[0], 0) + [event for event in events if event["row_id"] != original_first_id]
    for index, row in enumerate(rows):
        values = [None] * 5 if index < 80 else [index / 10, index / 5, -index / 20, None, 2]
        row["features"].update(dict(zip(INVENTORY_FEATURE_COLUMNS, values)))
        row["inventory_provenance"] = {
            "status": "no_earlier_release" if index < 80 else "available",
            "frozen_at_forecast_date": row["forecast_date"],
        }
    return rows, events, scoring


def original_reference(rows, events, scoring):
    original = deepcopy(rows)
    for row in original:
        row["features"] = {key: row["features"][key] for key in WTI_FEATURE_COLUMNS}
        row.pop("inventory_provenance", None)
    return replay_development(original, events, scoring, predictor=capturing_predictor([]))[0]


def inventory_predictor(calls):
    base = capturing_predictor(calls)

    def predict(training, forecast):
        models = base(training, forecast)
        models["logistic_wti_inventory"] = {
            "prediction": 0, "probability_up": 0.4,
            "diagnostics": {"iterations": 1,
                            "forecast_missing_columns": [column for column in INVENTORY_MODEL_COLUMNS
                                                         if forecast["features"][column] is None]},
        }
        return models

    return predict


class InventoryReplayTests(unittest.TestCase):
    def test_null_inventory_retains_every_wti_row_and_exact_training_hashes(self):
        rows, events, scoring = inventory_fixture()
        reference = original_reference(rows, events, scoring)
        calls = []
        predictions, summary = replay_development(
            rows, events, scoring, experiment="inventory-comparison", reference_rows=reference,
            predictor=inventory_predictor(calls))
        self.assertEqual(len(predictions), len(reference))
        self.assertTrue(all("wti_brent" not in row["eligibility"] for row in rows))
        for (training, forecast), prediction, original in zip(calls, predictions, reference):
            projection = [{**row, "features": {column: row["features"][column]
                                              for column in WTI_FEATURE_COLUMNS}} for row in training]
            self.assertEqual(prediction["training"]["count"], original["training"]["count"])
            self.assertEqual(prediction["training"]["input_sha256"], original["training"]["input_sha256"])
            self.assertEqual(prediction["training"]["input_sha256"], sha256(json_bytes(projection)))
            self.assertEqual(prediction["training"]["inventory_input_sha256"], sha256(json_bytes(training)))
            self.assertEqual(tuple(training[0]["features"]), INVENTORY_MODEL_COLUMNS)
            self.assertEqual(training[0]["forecast_date"], "2011-04-07")
            self.assertTrue(all(training[0]["features"][column] is None for column in INVENTORY_FEATURE_COLUMNS))
            self.assertEqual(prediction["training"]["rows_removed_vs_wti_eligible"], 0)
            self.assertEqual(prediction["training"]["inventory_missing_counts"][INVENTORY_FEATURE_COLUMNS[0]], 80)
            self.assertTrue(all(row["forecast_date"] < forecast["forecast_date"] for row in training))
            self.assertEqual({name: prediction["models"][name] for name in MODEL_NAMES}, original["models"])
        self.assertTrue(summary["reference_reproduction"]["all_wti_training_inputs_identical"])
        self.assertTrue(summary["reference_reproduction"]["all_reference_model_outputs_identical"])
        self.assertEqual(summary["common_training_cohort"]["sum_rows_removed_across_fits"], 0)

    def test_training_inventory_stays_frozen_at_its_own_issuance_across_later_fits(self):
        rows, events, scoring = inventory_fixture()
        reference = original_reference(rows, events, scoring)
        calls = []
        predictions, _ = replay_development(
            rows, events, scoring, experiment="inventory-comparison", reference_rows=reference,
            predictor=inventory_predictor(calls))
        by_id = {row["row_id"]: row for row in rows}
        for training, forecast in calls:
            for example in training:
                for column in INVENTORY_FEATURE_COLUMNS:
                    self.assertEqual(example["features"][column], by_id[example["row_id"]]["features"][column])
        first_development_id = predictions[0]["row_id"]
        later_example = next(row for row in calls[-1][0] if row["row_id"] == first_development_id)
        self.assertEqual(later_example["features"][INVENTORY_FEATURE_COLUMNS[0]],
                         rows[-3]["features"][INVENTORY_FEATURE_COLUMNS[0]])
        modified = deepcopy(rows)
        modified[-1]["features"][INVENTORY_FEATURE_COLUMNS[0]] = 999999
        changed_calls = []
        changed, _ = replay_development(
            modified, events, scoring, experiment="inventory-comparison", reference_rows=reference,
            predictor=inventory_predictor(changed_calls))
        self.assertEqual(predictions[:2], changed[:2])
        self.assertEqual(calls[-1][0], changed_calls[-1][0])
        self.assertNotEqual(calls[-1][1]["features"], changed_calls[-1][1]["features"])

    def test_stale_inventory_forecast_is_predicted_with_raw_null_changes_and_known_age(self):
        rows, events, scoring = inventory_fixture([date(2019, 7, 5), date(2019, 7, 19)])
        stale = rows[-2]
        stale["features"].update(dict(zip(INVENTORY_FEATURE_COLUMNS, [None, None, None, None, 2])))
        stale["inventory_provenance"] = {"status": "latest_declared_issue_unusable",
                                         "release_date": "2019-07-03"}
        before = deepcopy(rows)
        reference = original_reference(rows, events, scoring)
        calls = []
        predictions, summary = replay_development(
            rows, events, scoring, experiment="inventory-comparison", reference_rows=reference,
            predictor=inventory_predictor(calls))
        saved = predictions[0]
        self.assertEqual(saved["row_id"], stale["row_id"])
        self.assertEqual(saved["inventory"]["features"],
                         dict(zip(INVENTORY_FEATURE_COLUMNS, [None, None, None, None, 2])))
        self.assertEqual(saved["inventory"]["provenance"], stale["inventory_provenance"])
        diagnostic_missing = saved["models"]["logistic_wti_inventory"]["diagnostics"]["forecast_missing_columns"]
        self.assertEqual([column for column in diagnostic_missing if column in INVENTORY_FEATURE_COLUMNS],
                         list(INVENTORY_FEATURE_COLUMNS[:4]))
        later = next(example for example in calls[-1][0] if example["row_id"] == stale["row_id"])
        self.assertTrue(all(later["features"][column] is None for column in INVENTORY_FEATURE_COLUMNS[:4]))
        self.assertEqual(summary["prediction_count"], len(reference))
        self.assertEqual(rows, before)

    def test_changed_wti_training_inputs_fail_even_when_predictions_are_identical(self):
        rows, events, scoring = inventory_fixture()
        reference = original_reference(rows, events, scoring)
        rows[0]["features"][WTI_FEATURE_COLUMNS[0]] += 1
        calls = []
        with self.assertRaisesRegex(ValueError, "changed the original WTI training inputs"):
            replay_development(rows, events, scoring, experiment="inventory-comparison",
                               reference_rows=reference, predictor=inventory_predictor(calls))
        self.assertTrue(calls)

    def test_changed_wti_outputs_fail_with_unchanged_input_hashes(self):
        rows, events, scoring = inventory_fixture()
        reference = original_reference(rows, events, scoring)
        predictor = inventory_predictor([])

        def changed(training, forecast):
            result = predictor(training, forecast)
            result["logistic"]["probability_up"] = 0.7
            return result

        with self.assertRaisesRegex(ValueError, "outputs changed despite identical"):
            replay_development(rows, events, scoring, experiment="inventory-comparison",
                               reference_rows=reference, predictor=changed)

    def test_reference_schedule_or_scoring_mismatch_fails_before_any_fit(self):
        rows, events, scoring = inventory_fixture()
        reference = original_reference(rows, events, scoring)
        altered = deepcopy(reference)
        altered[-1]["target_date"] = "2017-02-10"
        for outcomes, expected in ((scoring, altered), (scoring[:-1], reference), (scoring, None)):
            calls = []
            with self.subTest(outcomes=len(outcomes), reference=expected is not None), self.assertRaises(ValueError):
                replay_development(rows, events, outcomes, experiment="inventory-comparison",
                                   reference_rows=expected, predictor=inventory_predictor(calls))
            self.assertEqual(calls, [])

    def test_missing_columns_even_on_late_forecasts_fail_before_any_fit(self):
        rows, events, scoring = inventory_fixture()
        reference = original_reference(rows, events, scoring)
        for index in (0, -1):
            malformed = deepcopy(rows)
            del malformed[index]["features"][INVENTORY_FEATURE_COLUMNS[-1]]
            calls = []
            with self.subTest(index=index), self.assertRaisesRegex(ValueError, "Missing declared inventory"):
                replay_development(malformed, events, scoring, experiment="inventory-comparison",
                                   reference_rows=reference, predictor=inventory_predictor(calls))
            self.assertEqual(calls, [])

    def test_holdout_features_and_label_information_are_rejected_before_fitting(self):
        rows, events, scoring = inventory_fixture()
        reference = original_reference(rows, events, scoring)
        holdout = feature_row(date(2023, 1, 5))
        for features, history, outcomes in (
            (rows + [holdout], events, scoring),
            (rows, events + event_history(holdout, 1), scoring),
            (rows, events, scoring + [{**scoring[-1], "available_from": "2023-01-03"}]),
        ):
            calls = []
            with self.assertRaisesRegex(ValueError, "Holdout"):
                replay_development(features, history, outcomes, experiment="inventory-comparison",
                                   reference_rows=reference, predictor=inventory_predictor(calls))
            self.assertEqual(calls, [])

    def test_unavailable_outcome_remains_in_the_identical_prediction_cohort(self):
        rows, events, scoring = inventory_fixture()
        missing_id = scoring[-1]["row_id"]
        events = [event for event in events if event["row_id"] != missing_id or event["label_status"] != "available"]
        for event in events:
            if event["row_id"] == missing_id:
                event["available_until"] = None
        scoring = scoring[:-1]
        reference = original_reference(rows, events, scoring)
        predictions, summary = replay_development(
            rows, events, scoring, experiment="inventory-comparison", reference_rows=reference,
            predictor=inventory_predictor([]))
        self.assertEqual([row["row_id"] for row in predictions], [row["row_id"] for row in reference])
        self.assertEqual((summary["prediction_count"], summary["scored_count"], summary["unscored_count"]), (3, 2, 1))
        self.assertEqual(predictions[-1]["scoring_status"], "unavailable_by_development_cutoff")
        self.assertIsNone(predictions[-1]["label"])
        self.assertEqual({result["count"] for result in summary["metrics"].values()}, {2})
        paired = summary["paired_inventory_comparison"]
        self.assertEqual(sum(paired[key] for key in ("both_correct", "both_wrong", "combined_only_correct", "wti_only_correct")), 2)
        self.assertFalse(summary["holdout_evaluated"])


if __name__ == "__main__":
    unittest.main()

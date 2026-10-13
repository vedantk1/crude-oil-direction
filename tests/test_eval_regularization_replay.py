"""Synthetic replay of one fixed regularization candidate and its exact controls."""

from copy import deepcopy
from datetime import date
import unittest

from oil_eval.inventory_features import INVENTORY_FEATURE_COLUMNS
from oil_eval.models import INVENTORY_MODEL_NAMES, INVENTORY_REGULARIZATION_MODEL_NAMES
from oil_eval.replay import replay_development
from tests.test_eval_inventory_replay import (
    inventory_fixture, inventory_predictor, original_reference,
)
from tests.test_eval_replay import event_history, feature_row


def references(rows, events, scoring):
    wti = original_reference(rows, events, scoring)
    inventory = replay_development(
        rows, events, scoring, experiment="inventory-comparison", reference_rows=wti,
        predictor=inventory_predictor([]))[0]
    return wti, inventory


def regularization_predictor(calls):
    base = inventory_predictor(calls)

    def predict(training, forecast):
        models = base(training, forecast)
        models["logistic_wti_inventory_c01"] = {
            "prediction": 1, "probability_up": 0.55, "diagnostics": {"iterations": 2},
        }
        return models

    return predict


class InventoryRegularizationReplayTests(unittest.TestCase):
    def test_same_forecasts_training_inventory_and_original_four_payloads(self):
        rows, events, scoring = inventory_fixture()
        wti, inventory = references(rows, events, scoring)
        calls = []
        predictions, summary = replay_development(
            rows, events, scoring, experiment="inventory-regularization", reference_rows=wti,
            inventory_reference_rows=inventory, predictor=regularization_predictor(calls))
        self.assertEqual(len(calls), len(inventory))
        for actual, control in zip(predictions, inventory):
            self.assertEqual(tuple(actual["models"]), INVENTORY_REGULARIZATION_MODEL_NAMES)
            self.assertEqual(actual["training"], control["training"])
            self.assertEqual(actual["inventory"], control["inventory"])
            self.assertEqual({name: actual["models"][name] for name in INVENTORY_MODEL_NAMES},
                             control["models"])
            self.assertEqual({key: value for key, value in actual.items() if key != "models"},
                             {key: value for key, value in control.items() if key != "models"})
        self.assertIn("inventory_reference_reproduction", summary)
        self.assertTrue(summary["reference_reproduction"]["all_reference_model_outputs_identical"])
        self.assertEqual(summary["common_training_cohort"]["sum_rows_removed_across_fits"], 0)
        self.assertEqual(set(summary["constant_benchmarks"]["comparator_metrics"]),
                         {"logistic", "logistic_wti_inventory", "logistic_wti_inventory_c01"})
        self.assertEqual(summary["maximum_iterations_by_logistic_model"]["logistic_wti_inventory_c01"], 2)

    def test_probability_and_direction_comparisons_use_the_identical_scored_cohort(self):
        rows, events, scoring = inventory_fixture()
        missing_id = scoring[-1]["row_id"]
        events = [event for event in events if event["row_id"] != missing_id or event["label_status"] != "available"]
        for event in events:
            if event["row_id"] == missing_id:
                event["available_until"] = None
        scoring = scoring[:-1]
        wti, inventory = references(rows, events, scoring)
        predictions, summary = replay_development(
            rows, events, scoring, experiment="inventory-regularization", reference_rows=wti,
            inventory_reference_rows=inventory, predictor=regularization_predictor([]))
        self.assertEqual((summary["prediction_count"], summary["scored_count"], summary["unscored_count"]), (3, 2, 1))
        self.assertIsNone(predictions[-1]["label"])
        self.assertEqual({value["count"] for value in summary["metrics"].values()}, {2})
        paired = summary["paired_inventory_regularization"]
        self.assertEqual(sum(paired[key] for key in
                             ("both_correct", "both_wrong", "candidate_only_correct", "control_only_correct")), 2)
        for metric, difference in paired["metric_differences_candidate_minus_control"].items():
            candidate = summary["metrics"]["logistic_wti_inventory_c01"][metric]
            control = summary["metrics"]["logistic_wti_inventory"][metric]
            self.assertEqual(difference, candidate - control if candidate is not None and control is not None else None)
        self.assertEqual(paired["yearly_metric_differences_candidate_minus_control"]["2017"],
                         paired["metric_differences_candidate_minus_control"])
        self.assertFalse(summary["holdout_evaluated"])

    def test_both_references_and_exact_inventory_forecast_state_required_before_fit(self):
        rows, events, scoring = inventory_fixture()
        wti, inventory = references(rows, events, scoring)
        for wti_reference, inventory_reference in ((None, inventory), (wti, None)):
            calls = []
            with self.assertRaises(ValueError):
                replay_development(rows, events, scoring, experiment="inventory-regularization",
                                   reference_rows=wti_reference, inventory_reference_rows=inventory_reference,
                                   predictor=regularization_predictor(calls))
            self.assertEqual(calls, [])
        for mutate in (
            lambda row: row["features"].update({INVENTORY_FEATURE_COLUMNS[0]: 999}),
            lambda row: row["inventory_provenance"].update(status="different"),
            lambda row: row["features"].pop(INVENTORY_FEATURE_COLUMNS[0]),
        ):
            changed = deepcopy(rows)
            mutate(changed[-1])
            calls = []
            with self.assertRaises(ValueError):
                replay_development(changed, events, scoring, experiment="inventory-regularization",
                                   reference_rows=wti, inventory_reference_rows=inventory,
                                   predictor=regularization_predictor(calls))
            self.assertEqual(calls, [])

    def test_changed_control_training_or_payload_never_claims_reproduction(self):
        rows, events, scoring = inventory_fixture()
        wti, inventory = references(rows, events, scoring)
        changed_rows = deepcopy(rows)
        changed_rows[0]["features"][INVENTORY_FEATURE_COLUMNS[0]] = 123
        with self.assertRaises(ValueError):
            replay_development(changed_rows, events, scoring, experiment="inventory-regularization",
                               reference_rows=wti, inventory_reference_rows=inventory,
                               predictor=regularization_predictor([]))
        predict = regularization_predictor([])

        def wrong_control(training, forecast):
            result = predict(training, forecast)
            result["logistic_wti_inventory"]["probability_up"] = 0.45
            return result

        with self.assertRaises(ValueError):
            replay_development(rows, events, scoring, experiment="inventory-regularization",
                               reference_rows=wti, inventory_reference_rows=inventory, predictor=wrong_control)

    def test_target_or_scoring_mismatch_fails_before_any_prediction(self):
        rows, events, scoring = inventory_fixture()
        wti, inventory = references(rows, events, scoring)
        for field, value in (("target_date", "2017-02-10"), ("label", 1)):
            changed = deepcopy(inventory)
            changed[-1][field] = value
            calls = []
            with self.assertRaises(ValueError):
                replay_development(rows, events, scoring, experiment="inventory-regularization",
                                   reference_rows=wti, inventory_reference_rows=changed,
                                   predictor=regularization_predictor(calls))
            self.assertEqual(calls, [])

    def test_holdout_information_is_rejected_before_fitting(self):
        rows, events, scoring = inventory_fixture()
        wti, inventory = references(rows, events, scoring)
        holdout = feature_row(date(2023, 1, 5))
        for features, history, outcomes in (
            (rows + [holdout], events, scoring),
            (rows, events + event_history(holdout, 1), scoring),
            (rows, events, scoring + [{**scoring[-1], "available_from": "2023-01-03"}]),
        ):
            calls = []
            with self.assertRaisesRegex(ValueError, "Holdout"):
                replay_development(features, history, outcomes, experiment="inventory-regularization",
                                   reference_rows=wti, inventory_reference_rows=inventory,
                                   predictor=regularization_predictor(calls))
            self.assertEqual(calls, [])

    def test_null_inventory_and_boundary_crossing_rows_keep_original_rules(self):
        rows, events, scoring = inventory_fixture([date(2019, 7, 5), date(2019, 7, 19)])
        rows[-2]["features"].update(dict(zip(INVENTORY_FEATURE_COLUMNS, [None, None, None, None, 2])))
        rows[-2]["inventory_provenance"] = {"status": "latest_declared_issue_unusable"}
        rows[-1]["features"][INVENTORY_FEATURE_COLUMNS[3]] = 0
        crossing = feature_row(date(2022, 12, 29))
        rows.append(crossing)
        wti, inventory = references(rows, events, scoring)
        calls = []
        predictions, summary = replay_development(
            rows, events, scoring, experiment="inventory-regularization", reference_rows=wti,
            inventory_reference_rows=inventory, predictor=regularization_predictor(calls))
        self.assertEqual(len(predictions), 2)
        self.assertNotIn(crossing["row_id"], {row["row_id"] for row in predictions})
        self.assertEqual(predictions[0]["inventory"]["features"],
                         dict(zip(INVENTORY_FEATURE_COLUMNS, [None, None, None, None, 2])))
        self.assertEqual(summary["development_forecasts_with_missing_inventory"], 1)
        self.assertEqual(summary["excluded_pre_holdout_rows"]["crosses_holdout_boundary"], 1)
        self.assertTrue(all(calls[0][0][0]["features"][column] is None
                            for column in INVENTORY_FEATURE_COLUMNS))


if __name__ == "__main__":
    unittest.main()

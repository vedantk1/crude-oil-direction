"""Synthetic fixed-tree replay; all model outputs are deterministic test doubles."""

from copy import deepcopy
from datetime import date
import math
from statistics import median
import unittest

from oil_audit.dataset import json_bytes, sha256
from oil_audit.features import WTI_FEATURE_COLUMNS
from oil_eval.inventory_features import INVENTORY_FEATURE_COLUMNS, INVENTORY_MODEL_COLUMNS
from oil_eval.models import INVENTORY_REGULARIZATION_MODEL_NAMES, INVENTORY_TREE_MODEL_NAMES
from oil_eval.replay import replay_development
from tests.test_eval_inventory_replay import inventory_fixture
from tests.test_eval_regularization_replay import references as inventory_references, regularization_predictor
from tests.test_eval_replay import event_history, feature_row


def references(rows, events, scoring):
    wti, inventory = inventory_references(rows, events, scoring)
    regularization = replay_development(
        rows, events, scoring, experiment="inventory-regularization", reference_rows=wti,
        inventory_reference_rows=inventory, predictor=regularization_predictor([]))[0]
    return {"reference_rows": wti, "inventory_reference_rows": inventory,
            "regularization_reference_rows": regularization}


def tree_predictor(calls):
    controls = regularization_predictor(calls)

    def predict(training, forecast):
        result = controls(training, forecast)
        statistics = {}
        for column in INVENTORY_MODEL_COLUMNS:
            known = [row["features"][column] for row in training if row["features"][column] is not None]
            statistics[column] = median(known) if known else 0
        half = len(training) // 2
        result["tree_wti_inventory"] = {
            "prediction": 0, "probability_up": 0.2,
            "diagnostics": {
                "feature_columns": list(INVENTORY_MODEL_COLUMNS),
                "imputer_statistics": statistics,
                "training_missing_counts": {
                    column: sum(row["features"][column] is None for row in training)
                    for column in INVENTORY_MODEL_COLUMNS},
                "forecast_missing_columns": [column for column in INVENTORY_MODEL_COLUMNS
                                             if forecast["features"][column] is None],
                "n_estimators_actual": 50, "max_depth_actual": 1,
                "feature_importances": {column: float(index == 0)
                                        for index, column in enumerate(INVENTORY_MODEL_COLUMNS)},
                "estimator_depths": [1] * 50,
                "estimator_leaf_counts": [2] * 50,
                "estimator_leaf_sample_counts": [[half, len(training) - half] for _ in range(50)],
            },
        }
        return result

    return predict


def replay(rows, events, scoring, refs, calls, predictor=None):
    return replay_development(rows, events, scoring, experiment="inventory-tree-comparison",
                              predictor=predictor or tree_predictor(calls), **refs)


class InventoryTreeReplayTests(unittest.TestCase):
    def test_all_six_models_share_the_expanding_cohort_and_five_exact_controls(self):
        rows, events, scoring = inventory_fixture()
        frozen = deepcopy((rows, events, scoring))
        refs, calls = references(rows, events, scoring), []
        predictions, summary = replay(rows, events, scoring, refs, calls)
        self.assertEqual(len(calls), len(refs["regularization_reference_rows"]))
        for actual, control, (training, forecast) in zip(
                predictions, refs["regularization_reference_rows"], calls):
            self.assertEqual(tuple(actual["models"]), INVENTORY_TREE_MODEL_NAMES)
            self.assertEqual({name: actual["models"][name] for name in INVENTORY_REGULARIZATION_MODEL_NAMES},
                             control["models"])
            self.assertEqual({key: value for key, value in actual.items() if key != "models"},
                             {key: value for key, value in control.items() if key != "models"})
            self.assertNotIn("rolling_training", actual)
            self.assertEqual(training[0]["forecast_date"], "2011-04-07")
            self.assertTrue(all(row["forecast_date"] < forecast["forecast_date"] for row in training))
            self.assertTrue(all(tuple(row["features"]) == INVENTORY_MODEL_COLUMNS for row in training))
            self.assertLess(actual["training"]["latest_target_date"], forecast["forecast_date"])
            self.assertLessEqual(actual["training"]["latest_label_available_from"], forecast["forecast_date"])
            self.assertEqual(actual["training"]["inventory_input_sha256"], sha256(json_bytes(training)))
            projection = [{**row, "features": {column: row["features"][column]
                                               for column in WTI_FEATURE_COLUMNS}} for row in training]
            self.assertEqual(actual["training"]["input_sha256"], sha256(json_bytes(projection)))
            self.assertEqual(actual["training"]["count"], len(training))
            self.assertEqual(actual["training"]["inventory_missing_counts"], {
                column: sum(row["features"][column] is None for row in training)
                for column in INVENTORY_FEATURE_COLUMNS})
        self.assertEqual((rows, events, scoring), frozen)
        self.assertEqual(summary["common_training_cohort"]["sum_rows_removed_across_fits"], 0)
        self.assertTrue(summary["regularization_reference_reproduction"]["all_expanding_c01_model_outputs_identical"])
        self.assertNotIn("rolling_training_count_range", summary)
        self.assertNotIn("tree_wti_inventory", summary["maximum_iterations_by_logistic_model"])
        samples = [count for training, _ in calls for count in
                   (len(training) // 2, len(training) - len(training) // 2)]
        self.assertEqual(summary["tree_fit_summary"], {
            "fit_count": len(calls), "estimator_count_range": [50, 50],
            "maximum_observed_depth": 1, "leaf_count_range": [2, 2],
            "leaf_training_sample_count_range": [min(samples), max(samples)],
        })

    def test_active_label_versions_and_frozen_missing_features_remain_causal(self):
        rows, events, scoring = inventory_fixture()
        recent = feature_row(date(2016, 12, 29), 500)
        recent["features"].update(dict.fromkeys(INVENTORY_FEATURE_COLUMNS))
        recent["inventory_provenance"] = {"status": "no_earlier_release"}
        rows.append(recent)
        events += event_history(recent, 1, date(2017, 1, 10))
        withdrawn = next(event for event in events
                         if event["row_id"] == rows[100]["row_id"] and event["label_status"] == "available")
        withdrawn["available_until"] = "2017-01-10"
        events.append({**withdrawn, "available_from": "2017-01-10", "available_until": None,
                       "label_status": "missing_target", "label": None, "target_price": None})
        refs, calls = references(rows, events, scoring), []
        original, _ = replay(rows, events, scoring, refs, calls)
        first, later = [{row["row_id"]: row for row in call[0]} for call in calls[:2]]
        self.assertIn(rows[100]["row_id"], first)
        self.assertNotIn(rows[100]["row_id"], later)
        self.assertNotIn(recent["row_id"], first)
        self.assertIn(recent["row_id"], later)
        self.assertTrue(all(later[recent["row_id"]]["features"][column] is None
                            for column in INVENTORY_FEATURE_COLUMNS))
        self.assertTrue(all(first[rows[0]["row_id"]]["features"][column] is None
                            for column in INVENTORY_FEATURE_COLUMNS))
        frozen = next(row for row in rows if row["row_id"] == scoring[0]["row_id"])
        self.assertEqual(next(row for row in calls[-1][0] if row["row_id"] == frozen["row_id"])["features"],
                         frozen["features"])
        changed_rows, changed_events = deepcopy(rows), deepcopy(events)
        next(row for row in changed_rows if row["row_id"] == scoring[-1]["row_id"])["features"].update(
            dict.fromkeys(INVENTORY_FEATURE_COLUMNS, 999))
        revised = next(event for event in changed_events
                       if event["row_id"] == rows[50]["row_id"] and event["label_status"] == "available")
        revised["available_until"] = "2017-01-25"
        changed_events.append({**revised, "available_from": "2017-01-25", "available_until": None,
                               "label": 1, "target_price": 31})
        changed_refs, changed_calls = references(changed_rows, changed_events, scoring), []
        changed, _ = replay(changed_rows, changed_events, scoring, changed_refs, changed_calls)
        self.assertEqual(changed[:2], original[:2])
        self.assertEqual(changed_calls[:2], calls[:2])
        self.assertEqual(next(row for row in changed_calls[-1][0] if row["row_id"] == rows[50]["row_id"])["label"], 1)

    def test_scored_denominators_tree_pairs_and_constants_use_identical_dates(self):
        rows, events, scoring = inventory_fixture()
        missing_id = scoring[-1]["row_id"]
        events = [event for event in events if event["row_id"] != missing_id or event["label_status"] != "available"]
        for event in events:
            if event["row_id"] == missing_id:
                event["available_until"] = None
        scoring = scoring[:-1]
        refs = references(rows, events, scoring)
        predictions, summary = replay(rows, events, scoring, refs, [])
        self.assertEqual((summary["prediction_count"], summary["scored_count"], summary["unscored_count"]), (3, 2, 1))
        self.assertIsNone(predictions[-1]["label"])
        self.assertIsNone(predictions[-1]["label_available_from"])
        self.assertEqual({metrics["count"] for metrics in summary["metrics"].values()}, {2})
        tree = summary["metrics"]["tree_wti_inventory"]
        self.assertAlmostEqual(tree["brier_score"], (0.2 ** 2 + 0.8 ** 2) / 2)
        self.assertAlmostEqual(tree["log_loss"], -(math.log(0.8) + math.log(0.2)) / 2)
        paired = summary["paired_inventory_tree_comparison"]
        self.assertEqual({key: paired[key] for key in
                          ("both_correct", "both_wrong", "candidate_only_correct", "control_only_correct")},
                         {"both_correct": 0, "both_wrong": 0, "candidate_only_correct": 1, "control_only_correct": 1})
        for metric, difference in paired["metric_differences_tree_minus_logistic"].items():
            self.assertAlmostEqual(difference, tree[metric] - summary["metrics"]["logistic_wti_inventory_c01"][metric])
        self.assertEqual(paired["yearly_metric_differences_tree_minus_logistic"]["2017"],
                         paired["metric_differences_tree_minus_logistic"])
        constants = summary["constant_benchmarks"]
        self.assertEqual(set(constants["comparator_metrics"]), {
            "logistic", "logistic_wti_inventory", "logistic_wti_inventory_c01", "tree_wti_inventory"})
        self.assertEqual((constants["prediction_count"], constants["scored_count"], constants["unscored_count"]), (3, 2, 1))
        self.assertEqual(constants["comparator_metrics"]["tree_wti_inventory"], tree)
        for name, comparison in constants["paired_comparisons"]["tree_wti_inventory"].items():
            self.assertEqual(sum(comparison[key] for key in
                                 ("both_correct", "both_wrong", "model_only_correct", "benchmark_only_correct")), 2)
            for metric, difference in comparison["metric_differences_model_minus_benchmark"].items():
                expected = constants["metrics"][name][metric]
                self.assertEqual(difference, None if expected is None else tree[metric] - expected)

    def test_missing_or_mismatched_references_fail_before_any_prediction(self):
        rows, events, scoring = inventory_fixture()
        refs = references(rows, events, scoring)
        variants = [{**refs, name: None} for name in refs]
        for mutate in (
            lambda row: row.update(label=1),
            lambda row: row.update(target_date="2017-02-10"),
            lambda row: row["inventory"]["features"].update({INVENTORY_FEATURE_COLUMNS[0]: 999}),
            lambda row: row["inventory"]["provenance"].update(status="different"),
        ):
            changed = deepcopy(refs)
            mutate(changed["regularization_reference_rows"][-1])
            variants.append(changed)
        for variant in variants:
            calls = []
            with self.assertRaises(ValueError):
                replay(rows, events, scoring, variant, calls)
            self.assertEqual(calls, [])

    def test_changed_expanding_c01_cannot_claim_control_reproduction(self):
        rows, events, scoring = inventory_fixture()
        refs, calls = references(rows, events, scoring), []
        original = tree_predictor(calls)

        def changed(training, forecast):
            result = original(training, forecast)
            result["logistic_wti_inventory_c01"]["probability_up"] = 0.6
            return result

        with self.assertRaisesRegex(ValueError, "Expanding C=0.1 model outputs differ"):
            replay(rows, events, scoring, refs, calls, predictor=changed)
        self.assertTrue(calls)

    def test_boundary_forecast_is_purged_and_holdout_poison_fails_before_prediction(self):
        rows, events, scoring = inventory_fixture()
        crossing = feature_row(date(2022, 12, 29))
        rows.append(crossing)
        refs, calls = references(rows, events, scoring), []
        predictions, summary = replay(rows, events, scoring, refs, calls)
        self.assertEqual(len(predictions), 3)
        self.assertNotIn(crossing["row_id"], {row["row_id"] for row in predictions})
        self.assertTrue(all(row["row_id"] != crossing["row_id"] for training, _ in calls for row in training))
        self.assertEqual(summary["excluded_pre_holdout_rows"]["crosses_holdout_boundary"], 1)
        self.assertFalse(summary["holdout_evaluated"])
        holdout = feature_row(date(2023, 1, 5))
        for features, history, outcomes in (
            (rows + [holdout], events, scoring),
            (rows, events + event_history(holdout, 1), scoring),
            (rows, events, scoring + [{**scoring[-1], "available_from": "2023-01-03"}]),
        ):
            calls = []
            with self.assertRaisesRegex(ValueError, "Holdout"):
                replay(features, history, outcomes, refs, calls)
            self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()

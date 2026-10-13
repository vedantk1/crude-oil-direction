"""Synthetic rolling replay, with authenticated expanding controls as fixtures."""

from copy import deepcopy
from datetime import date
import math
import unittest

from oil_audit.dataset import json_bytes, sha256
from oil_audit.features import WTI_FEATURE_COLUMNS
from oil_eval.inventory_features import INVENTORY_FEATURE_COLUMNS
from oil_eval.models import INVENTORY_REGULARIZATION_MODEL_NAMES, INVENTORY_WINDOW_MODEL_NAMES
from oil_eval.replay import replay_development
from tests.test_eval_inventory_replay import inventory_predictor, original_reference
from tests.test_eval_regularization_replay import regularization_predictor
from tests.test_eval_replay import event_history, feature_row, fixture


def window_fixture(warmup=240):
    rows, events, scoring = fixture(warmup=warmup)
    for index, row in enumerate(rows):
        row["features"].update(dict(zip(INVENTORY_FEATURE_COLUMNS,
                                       [index / 10, None, -index / 20, None, 2])))
        row["inventory_provenance"] = {"status": "available",
                                         "frozen_at_forecast_date": row["forecast_date"]}
    return rows, events, scoring


def references(rows, events, scoring):
    wti = original_reference(rows, events, scoring)
    inventory = replay_development(
        rows, events, scoring, experiment="inventory-comparison", reference_rows=wti,
        predictor=inventory_predictor([]))[0]
    regularization = replay_development(
        rows, events, scoring, experiment="inventory-regularization", reference_rows=wti,
        inventory_reference_rows=inventory, predictor=regularization_predictor([]))[0]
    return {"reference_rows": wti, "inventory_reference_rows": inventory,
            "regularization_reference_rows": regularization}


def window_predictor(calls):
    controls = regularization_predictor([])

    def predict(expanding, rolling, forecast):
        calls.append(deepcopy((expanding, rolling, forecast)))
        result = controls(expanding, forecast)
        result["logistic_wti_inventory_c01_rolling"] = {
            "prediction": 0, "probability_up": 0.2, "diagnostics": {"iterations": 3},
        }
        return result

    return predict


def replay(rows, events, scoring, refs, calls, predictor=None):
    return replay_development(rows, events, scoring, experiment="inventory-window-comparison",
                              predictor=predictor or window_predictor(calls), **refs)


class InventoryWindowReplayTests(unittest.TestCase):
    def test_ordered_causal_subset_hashes_metadata_and_all_five_controls(self):
        rows, events, scoring = window_fixture()
        refs, calls = references(rows, events, scoring), []
        predictions, summary = replay(rows, events, scoring, refs, calls)
        for actual, control, (expanding, rolling, forecast) in zip(
                predictions, refs["regularization_reference_rows"], calls):
            self.assertEqual(tuple(actual["models"]), INVENTORY_WINDOW_MODEL_NAMES)
            self.assertEqual({name: actual["models"][name] for name in INVENTORY_REGULARIZATION_MODEL_NAMES},
                             control["models"])
            self.assertEqual({key: value for key, value in actual.items()
                              if key not in ("models", "rolling_training")},
                             {key: value for key, value in control.items() if key != "models"})
            start = forecast["forecast_date"].replace("2017", "2013", 1)
            self.assertEqual(rolling, [row for row in expanding if row["forecast_date"] >= start])
            self.assertTrue(all(start <= row["forecast_date"] < forecast["forecast_date"] for row in rolling))
            metadata = actual["rolling_training"]
            self.assertEqual(metadata["window_start_date"], start)
            self.assertEqual(metadata["window_end_exclusive"], forecast["forecast_date"])
            self.assertEqual(metadata["count"], len(rolling))
            self.assertEqual(metadata["up_count"], sum(row["label"] for row in rolling))
            self.assertEqual(metadata["first_forecast_date"], rolling[0]["forecast_date"])
            self.assertEqual(metadata["last_forecast_date"], rolling[-1]["forecast_date"])
            self.assertEqual(metadata["expanding_eligible_count"], len(expanding))
            self.assertEqual(metadata["rows_excluded_before_window"], len(expanding) - len(rolling))
            self.assertGreater(metadata["rows_excluded_before_window"], 0)
            self.assertGreaterEqual(metadata["count"], 150)
            rolling_ids = {row["row_id"] for row in rolling}
            active = [event for event in events if event["row_id"] in rolling_ids
                      and event["label_status"] == "available"]
            self.assertEqual(metadata["latest_target_date"], max(event["target_date"] for event in active))
            self.assertEqual(metadata["latest_label_available_from"], max(event["available_from"] for event in active))
            self.assertLess(metadata["latest_target_date"], forecast["forecast_date"])
            self.assertLessEqual(metadata["latest_label_available_from"], forecast["forecast_date"])
            self.assertEqual(metadata["inventory_input_sha256"], sha256(json_bytes(rolling)))
            projection = [{**row, "features": {column: row["features"][column]
                                               for column in WTI_FEATURE_COLUMNS}} for row in rolling]
            self.assertEqual(metadata["input_sha256"], sha256(json_bytes(projection)))
            self.assertEqual(metadata["inventory_missing_counts"], {
                column: sum(row["features"][column] is None for row in rolling)
                for column in INVENTORY_FEATURE_COLUMNS})
        self.assertTrue(summary["regularization_reference_reproduction"]["all_expanding_c01_model_outputs_identical"])
        self.assertEqual(summary["rolling_training_count_range"],
                         [min(len(call[1]) for call in calls), max(len(call[1]) for call in calls)])

    def test_active_labels_apply_before_window_and_frozen_nulls_never_backfill(self):
        rows, events, scoring = window_fixture()
        recent = feature_row(date(2016, 12, 29), 500)
        recent["features"].update(dict.fromkeys(INVENTORY_FEATURE_COLUMNS))
        recent["inventory_provenance"] = {"status": "no_earlier_release"}
        rows.append(recent)
        events += event_history(recent, 1, date(2017, 1, 10))
        withdrawn = rows[100]
        known = next(event for event in events
                     if event["row_id"] == withdrawn["row_id"] and event["label_status"] == "available")
        known["available_until"] = "2017-01-10"
        events.append({**known, "available_from": "2017-01-10", "available_until": None,
                       "label_status": "missing_target", "label": None, "target_price": None})
        refs, calls = references(rows, events, scoring), []
        replay(rows, events, scoring, refs, calls)
        first, later = [{row["row_id"]: row for row in call[1]} for call in calls[:2]]
        self.assertIn(withdrawn["row_id"], first)
        self.assertNotIn(withdrawn["row_id"], later)
        self.assertNotIn(recent["row_id"], first)
        self.assertIn(recent["row_id"], later)
        self.assertTrue(all(later[recent["row_id"]]["features"][column] is None
                            for column in INVENTORY_FEATURE_COLUMNS))
        oldest_id = rows[0]["row_id"]
        for expanding, rolling, forecast in calls:
            self.assertIn(oldest_id, {row["row_id"] for row in expanding})
            self.assertNotIn(oldest_id, {row["row_id"] for row in rolling})
            self.assertTrue(all(row["forecast_date"] >= forecast["forecast_date"].replace("2017", "2013", 1)
                                for row in rolling))
        frozen_id = scoring[0]["row_id"]
        frozen = next(row for row in rows if row["row_id"] == frozen_id)
        later_training = next(row for row in calls[-1][1] if row["row_id"] == frozen_id)
        self.assertEqual(later_training["features"], frozen["features"])

    def test_same_scored_and_unscored_dates_and_paired_probability_denominators(self):
        rows, events, scoring = window_fixture()
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
        self.assertEqual({metrics["count"] for metrics in summary["metrics"].values()}, {2})
        candidate = summary["metrics"]["logistic_wti_inventory_c01_rolling"]
        self.assertAlmostEqual(candidate["brier_score"], (0.2 ** 2 + 0.8 ** 2) / 2)
        self.assertAlmostEqual(candidate["log_loss"], -(math.log(0.8) + math.log(0.2)) / 2)
        paired = summary["paired_inventory_window_comparison"]
        self.assertEqual({key: paired[key] for key in
                          ("both_correct", "both_wrong", "candidate_only_correct", "control_only_correct")},
                         {"both_correct": 0, "both_wrong": 0, "candidate_only_correct": 1, "control_only_correct": 1})
        for metric, difference in paired["metric_differences_rolling_minus_expanding"].items():
            self.assertAlmostEqual(difference, candidate[metric] - summary["metrics"]["logistic_wti_inventory_c01"][metric])
        self.assertEqual(paired["yearly_metric_differences_rolling_minus_expanding"]["2017"],
                         paired["metric_differences_rolling_minus_expanding"])
        self.assertEqual(summary["constant_benchmarks"]["scored_count"], 2)

    def test_missing_or_changed_regularization_reference_fails_before_prediction(self):
        rows, events, scoring = window_fixture()
        refs = references(rows, events, scoring)
        variants = [{**refs, "regularization_reference_rows": None}]
        for mutate in (
            lambda row: row.update(label=1),
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

    def test_rolling_minimum_and_both_classes_fail_without_expanding_fallback(self):
        sparse = window_fixture(warmup=160)
        rows, events, scoring = window_fixture()
        for event in events + scoring:
            if event["forecast_date"] >= "2013-01-05" and event["label_status"] == "available":
                event.update(label=0, target_price=29)
        for (features, history, outcomes), error in (
            (sparse, "Insufficient eligible rolling training history"),
            ((rows, events, scoring), "Both rolling training classes"),
        ):
            refs = references(features, history, outcomes)
            calls = []
            self.assertGreaterEqual(refs["regularization_reference_rows"][0]["training"]["count"], 150)
            with self.assertRaisesRegex(ValueError, error):
                replay(features, history, outcomes, refs, calls)
            self.assertEqual(calls, [])

    def test_changed_expanding_c01_output_cannot_claim_control_reproduction(self):
        rows, events, scoring = window_fixture()
        refs, calls = references(rows, events, scoring), []
        original = window_predictor(calls)

        def changed(expanding, rolling, forecast):
            result = original(expanding, rolling, forecast)
            result["logistic_wti_inventory_c01"]["probability_up"] = 0.6
            return result

        with self.assertRaisesRegex(ValueError, "Expanding C=0.1 model outputs differ"):
            replay(rows, events, scoring, refs, calls, predictor=changed)
        self.assertTrue(calls)

    def test_holdout_poison_rejected_and_boundary_crossing_forecast_purged(self):
        rows, events, scoring = window_fixture()
        crossing = feature_row(date(2022, 12, 29))
        rows.append(crossing)
        refs, calls = references(rows, events, scoring), []
        predictions, summary = replay(rows, events, scoring, refs, calls)
        self.assertEqual(len(predictions), 3)
        self.assertNotIn(crossing["row_id"], {row["row_id"] for row in predictions})
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

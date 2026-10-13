"""Synthetic clock-stress replay tests; no historical source or holdout reads."""

from copy import deepcopy
from datetime import date, timedelta
import unittest

from oil_audit.calendar import previous_business_day
from oil_audit.dataset import json_bytes, sha256
from oil_audit.features import WTI_FEATURE_COLUMNS
from oil_eval.delay import DELAYED_MODEL_NAME, replay_delayed_development
from oil_eval.inventory_features import INVENTORY_FEATURE_COLUMNS, INVENTORY_MODEL_COLUMNS
from oil_eval.models import INVENTORY_REGULARIZATION_MODEL_NAMES
from oil_eval.replay import replay_development
from tests.test_eval_inventory_replay import inventory_fixture
from tests.test_eval_regularization_replay import references, regularization_predictor
from tests.test_eval_replay import event_history, feature_row


def delayed_copy(rows):
    """Supply frozen inputs with independently computed Monday/holiday clocks."""
    copied = deepcopy(rows)
    for index, row in enumerate(copied):
        issued = date.fromisoformat(row["forecast_date"])
        knowledge = previous_business_day(issued)
        cutoff = (knowledge - timedelta(days=1)).isoformat()
        row["features"]["wti_latest_price"] += 0.25 + index / 100
        row["features"]["wti_age_calendar_days"] += (issued - knowledge).days
        row["features"][INVENTORY_FEATURE_COLUMNS[0]] = None if index % 2 else index / 20
        row["provenance"] = {"fixture_frozen_at": row["forecast_date"]}
        row["information_cutoff_date"] = cutoff
        row["availability_delay_provenance"] = {
            "label_knowledge_date": knowledge.isoformat(), "information_cutoff_date": cutoff,
            "feature_age_reference_date": row["forecast_date"],
        }
    return copied


def all_references(rows, events, scoring):
    wti, inventory = references(rows, events, scoring)
    regularization = replay_development(
        rows, events, scoring, experiment="inventory-regularization", reference_rows=wti,
        inventory_reference_rows=inventory, predictor=regularization_predictor([]))[0]
    return {"reference_rows": wti, "inventory_reference_rows": inventory,
            "regularization_reference_rows": regularization}


def candidate_predictor(calls):
    def predict(training, forecast):
        calls.append((deepcopy(training), deepcopy(forecast)))
        return {"prediction": 0, "probability_up": 0.45, "diagnostics": {"iterations": 3}}
    return predict


class DelayReplayTests(unittest.TestCase):
    def setUp(self):
        # Jan 9 is Monday; Jan 17 follows a Monday US holiday.
        self.rows, self.events, self.scoring = inventory_fixture(
            [date(2017, 1, 9), date(2017, 1, 17), date(2017, 2, 2)])
        self.delayed = delayed_copy(self.rows)

    def replay(self, *, rows=None, delayed=None, events=None, scoring=None,
               saved=None, control_calls=None, candidate_calls=None, control=None):
        rows = self.rows if rows is None else rows
        events = self.events if events is None else events
        scoring = self.scoring if scoring is None else scoring
        saved = all_references(rows, events, scoring) if saved is None else saved
        return replay_delayed_development(
            rows, self.delayed if delayed is None else delayed, events, scoring, **saved,
            control_predictor=control or regularization_predictor(control_calls if control_calls is not None else []),
            candidate_predictor=candidate_predictor(candidate_calls if candidate_calls is not None else []))

    def test_fixed_origins_scores_controls_and_own_issuance_training_inputs(self):
        saved = all_references(self.rows, self.events, self.scoring)
        before = deepcopy((self.rows, self.delayed, self.events, self.scoring, saved))
        calls = []
        predictions, summary = self.replay(saved=saved, candidate_calls=calls)
        by_id = {row["row_id"]: row for row in self.delayed}
        for prediction, reference, (training, forecast) in zip(
                predictions, saved["regularization_reference_rows"], calls):
            for field in ("row_id", "forecast_date", "target_date", "scoring_status", "label", "label_available_from",
                          "training", "inventory"):
                self.assertEqual(prediction[field], reference[field])
            self.assertEqual({name: prediction["models"][name] for name in INVENTORY_REGULARIZATION_MODEL_NAMES},
                             reference["models"])
            self.assertEqual(prediction["delayed_information"]["features"], forecast["features"])
            for example in training:
                self.assertEqual(example["features"], by_id[example["row_id"]]["features"])
                self.assertLess(example["forecast_date"], forecast["forecast_date"])
                self.assertEqual(tuple(example["features"]), INVENTORY_MODEL_COLUMNS)
            metadata = prediction["delayed_training"]
            self.assertEqual(metadata["inventory_input_sha256"], sha256(json_bytes(training)))
            projection = [{**row, "features": {key: row["features"][key] for key in WTI_FEATURE_COLUMNS}}
                          for row in training]
            self.assertEqual(metadata["input_sha256"], sha256(json_bytes(projection)))
            self.assertEqual(metadata["selected_row_ids"], [row["row_id"] for row in training])
            self.assertLess(metadata["latest_target_date"], metadata["label_knowledge_date"])
            self.assertLessEqual(metadata["latest_original_label_available_from"], metadata["label_knowledge_date"])
            self.assertLessEqual(metadata["latest_delayed_label_available_from"], forecast["forecast_date"])
        self.assertEqual(predictions[0]["delayed_training"]["label_knowledge_date"], "2017-01-06")
        self.assertEqual(predictions[0]["delayed_training"]["information_cutoff_date"], "2017-01-05")
        self.assertEqual(predictions[1]["delayed_training"]["label_knowledge_date"], "2017-01-13")
        self.assertEqual(predictions[1]["delayed_training"]["information_cutoff_date"], "2017-01-12")
        first_origin = predictions[0]["row_id"]
        self.assertEqual(next(row for row in calls[-1][0] if row["row_id"] == first_origin)["features"],
                         by_id[first_origin]["features"])
        self.assertFalse(summary["holdout_evaluated"])
        self.assertEqual((self.rows, self.delayed, self.events, self.scoring, saved), before)

    def test_active_revisions_withdrawals_and_new_labels_use_delayed_clock(self):
        events = deepcopy(self.events)
        revised_id, withdrawn_id, late_id, ineligible_id = [row["row_id"] for row in self.rows[:4]]
        for event in events:
            if event["row_id"] == late_id:
                if event["label_status"] == "available":
                    event["available_from"] = "2017-01-09"
                else:
                    event["available_until"] = "2017-01-09"
        for identity, withdrawal in ((revised_id, False), (withdrawn_id, True)):
            old = next(event for event in events if event["row_id"] == identity and event["label_status"] == "available")
            old["available_until"] = "2017-01-09"
            events.append({**old, "available_from": "2017-01-09", "available_until": None,
                           "label_status": "missing_target" if withdrawal else "available",
                           "label": None if withdrawal else 1 - old["label"],
                           "target_price": None if withdrawal else (29 if old["label"] else 31)})
        delayed = deepcopy(self.delayed)
        delayed[3]["features"]["wti_latest_price"] = None
        delayed[3]["eligibility"]["wti"] = False
        calls = []
        predictions, _ = self.replay(events=events, delayed=delayed, candidate_calls=calls)
        first = predictions[0]["delayed_training"]
        labels = {row["row_id"]: row["label"] for row in calls[0][0]}
        self.assertEqual(labels[revised_id], 0)
        self.assertIn(withdrawn_id, labels)
        self.assertNotIn(late_id, labels)
        self.assertNotIn(ineligible_id, labels)
        self.assertEqual(first["added_vs_original_training_ids"], [withdrawn_id])
        self.assertEqual(first["removed_original_training_ids"], [late_id, ineligible_id])
        self.assertEqual(first["changed_active_labels"], [{"row_id": revised_id, "original_label": 1, "delayed_label": 0}])
        self.assertEqual(first["excluded_past_rows"]["wti_ineligible"], [ineligible_id])
        self.assertIn(late_id, first["excluded_past_rows"]["label_unavailable"])
        later = {row["row_id"]: row["label"] for row in calls[1][0]}
        self.assertEqual(later[revised_id], 1)
        self.assertNotIn(withdrawn_id, later)
        self.assertIn(late_id, later)

    def test_future_feature_changes_do_not_rewrite_earlier_delayed_fits(self):
        original_calls, changed_calls = [], []
        expected, _ = self.replay(candidate_calls=original_calls)
        changed = deepcopy(self.delayed)
        changed[-1]["features"][INVENTORY_FEATURE_COLUMNS[0]] = 999999
        actual, _ = self.replay(delayed=changed, candidate_calls=changed_calls)
        self.assertEqual(actual[:-1], expected[:-1])
        self.assertEqual(changed_calls[-1][0], original_calls[-1][0])
        self.assertNotEqual(changed_calls[-1][1]["features"], original_calls[-1][1]["features"])

    def test_late_forecast_preflight_failures_happen_before_any_fit(self):
        saved = all_references(self.rows, self.events, self.scoring)
        mutations = (
            lambda row: row["features"].pop(INVENTORY_FEATURE_COLUMNS[-1]),
            lambda row: row["features"].update({INVENTORY_FEATURE_COLUMNS[-1]: float("inf")}),
            lambda row: row["availability_delay_provenance"].update(label_knowledge_date=row["forecast_date"]),
            lambda row: row.pop("inventory_provenance"),
            lambda row: row.update(target_date="2017-02-10"),
            lambda row: (row["features"].update(wti_latest_price=None), row["eligibility"].update(wti=False)),
        )
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                changed = deepcopy(self.delayed)
                mutate(changed[-1])
                controls, candidates = [], []
                with self.assertRaises(ValueError):
                    self.replay(saved=saved, delayed=changed, control_calls=controls, candidate_calls=candidates)
                self.assertEqual((controls, candidates), ([], []))

    def test_missing_rows_references_and_scoring_changes_fail_before_any_fit(self):
        saved = all_references(self.rows, self.events, self.scoring)
        cases = [(deepcopy(saved), self.delayed[:-1], self.scoring)]
        for name in saved:
            changed = deepcopy(saved)
            changed[name] = None
            cases.append((changed, self.delayed, self.scoring))
        changed_scoring = deepcopy(self.scoring)
        changed_scoring[-1]["label"] = 1 - changed_scoring[-1]["label"]
        cases.append((saved, self.delayed, changed_scoring))
        for refs, delayed, scoring in cases:
            controls, candidates = [], []
            with self.assertRaises(ValueError):
                self.replay(saved=refs, delayed=delayed, scoring=scoring,
                            control_calls=controls, candidate_calls=candidates)
            self.assertEqual((controls, candidates), ([], []))

    def test_delayed_capacity_and_class_failures_are_preflighted(self):
        saved = all_references(self.rows, self.events, self.scoring)
        delayed = deepcopy(self.delayed)
        for index in range(11):
            delayed[index]["features"]["wti_latest_price"] = None
            delayed[index]["eligibility"]["wti"] = False
        controls, candidates = [], []
        with self.assertRaisesRegex(ValueError, "Insufficient delayed training"):
            self.replay(saved=saved, delayed=delayed, control_calls=controls, candidate_calls=candidates)
        self.assertEqual((controls, candidates), ([], []))
        # All 160 past labels are zero at P; half become one on D. Thus the
        # baseline is valid while the delayed candidate lacks a class, not rows.
        events = deepcopy(self.events)
        training_ids = {row["row_id"] for row in self.rows[:-3]}
        revisions = []
        for event in events:
            if event["row_id"] in training_ids and event["label_status"] == "available":
                if event["label"] == 1:
                    revisions.append({**event, "available_from": "2017-01-09", "available_until": None})
                    event["available_until"] = "2017-01-09"
                event.update(label=0, target_price=29)
        events.extend(revisions)
        controls, candidates = [], []
        with self.assertRaisesRegex(ValueError, "Both delayed training classes"):
            self.replay(events=events, control_calls=controls, candidate_calls=candidates)
        self.assertEqual((controls, candidates), ([], []))

    def test_changed_original_training_hash_fails_before_control_and_candidate_fits(self):
        saved = all_references(self.rows, self.events, self.scoring)
        changed = deepcopy(self.rows)
        changed[0]["features"][WTI_FEATURE_COLUMNS[0]] += 1
        controls, candidates = [], []
        with self.assertRaisesRegex(ValueError, "Original training inputs differ"):
            self.replay(rows=changed, saved=saved, control_calls=controls, candidate_calls=candidates)
        self.assertEqual((controls, candidates), ([], []))

    def test_changed_expanding_control_output_stops_before_candidate_fits(self):
        saved = all_references(self.rows, self.events, self.scoring)
        controls, candidates = [], []
        predict = regularization_predictor(controls)

        def wrong_control(training, forecast):
            result = predict(training, forecast)
            result["logistic_wti_inventory_c01"]["probability_up"] = 0.56
            return result

        with self.assertRaisesRegex(ValueError, "Expanding C=0.1 model outputs differ"):
            self.replay(saved=saved, control=wrong_control, candidate_calls=candidates)
        self.assertEqual(len(controls), len(saved["regularization_reference_rows"]))
        self.assertEqual(candidates, [])

    def test_holdout_events_and_origins_fail_before_any_fit(self):
        saved = all_references(self.rows, self.events, self.scoring)
        future = feature_row(date(2023, 1, 5))
        for rows, events, scoring in (
            (self.rows + [future], self.events, self.scoring),
            (self.rows, self.events + event_history(future, 1), self.scoring),
            (self.rows, self.events + [{**self.events[1], "available_from": "2023-01-03"}], self.scoring),
            (self.rows, self.events, self.scoring + [{**self.scoring[-1], "available_from": "2023-01-03"}]),
        ):
            controls, candidates = [], []
            with self.assertRaisesRegex(ValueError, "Holdout"):
                self.replay(rows=rows, events=events, scoring=scoring, saved=saved,
                            control_calls=controls, candidate_calls=candidates)
            self.assertEqual((controls, candidates), ([], []))

    def test_future_interval_ends_cannot_change_or_escape_saved_fit_states(self):
        expected, _ = self.replay()
        changed = deepcopy(self.events)
        for event in changed:
            if event["available_until"] is None:
                event["available_until"] = "2024-01-02"
        actual, _ = self.replay(events=changed)
        self.assertEqual(actual, expected)
        for prediction in actual:
            states = prediction["delayed_training"]["selected_label_states"]
            self.assertTrue(all("available_until" not in state for state in states))
            self.assertEqual(prediction["delayed_training"]["selected_label_states_sha256"], sha256(json_bytes(states)))

    def test_unscored_forecast_and_null_inventory_remain_in_shared_cohort(self):
        missing_id = self.scoring[-1]["row_id"]
        events = [deepcopy(event) for event in self.events
                  if event["row_id"] != missing_id or event["label_status"] != "available"]
        for event in events:
            if event["row_id"] == missing_id:
                event["available_until"] = None
        delayed = deepcopy(self.delayed)
        delayed[-1]["features"].update(dict(zip(INVENTORY_FEATURE_COLUMNS, [None, None, None, None, 3])))
        calls = []
        predictions, summary = self.replay(events=events, scoring=self.scoring[:-1], delayed=delayed,
                                           candidate_calls=calls)
        self.assertEqual((summary["prediction_count"], summary["scored_count"], summary["unscored_count"]), (3, 2, 1))
        self.assertEqual(len(calls), 3)
        self.assertIsNone(predictions[-1]["label"])
        self.assertEqual(predictions[-1]["delayed_information"]["features"][INVENTORY_FEATURE_COLUMNS[-1]], 3)
        self.assertTrue(all(predictions[-1]["delayed_information"]["features"][name] is None
                            for name in INVENTORY_FEATURE_COLUMNS[:4]))
        self.assertEqual({metrics["count"] for metrics in summary["metrics"].values()}, {2})
        paired = summary["paired_availability_delay"]
        self.assertEqual(sum(paired[key] for key in
                             ("both_correct", "both_wrong", "delayed_only_correct", "original_only_correct")), 2)
        for metric, difference in paired["metric_differences_delayed_minus_original"].items():
            delayed_metric = summary["metrics"][DELAYED_MODEL_NAME][metric]
            control_metric = summary["metrics"]["logistic_wti_inventory_c01"][metric]
            self.assertEqual(difference, delayed_metric - control_metric
                             if delayed_metric is not None and control_metric is not None else None)


if __name__ == "__main__":
    unittest.main()

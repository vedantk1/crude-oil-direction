"""Historical training selection excludes future outcomes and withdrawn labels."""

from datetime import date
import unittest

from oil_audit.availability import PointInTimeSeries
from oil_audit.features import FEATURE_SETS, build_features
from oil_audit.labels import build_label_events
from oil_audit.training import training_rows


def row(series, observed, available, value, end=""):
    return {"period_start_date": observed, "realtime_start_date": available,
            "realtime_end_date": end, series: value}


class TrainingTests(unittest.TestCase):
    def setUp(self):
        self.wti = PointInTimeSeries("DCOILWTICO", [
            row("DCOILWTICO", "2020-05-01", "2020-05-01", "28"),
            row("DCOILWTICO", "2020-05-04", "2020-05-05", "30"),
            row("DCOILWTICO", "2020-05-11", "2020-05-13", "31", "2020-05-19"),
            row("DCOILWTICO", "2020-05-11", "2020-05-20", "29", "2020-05-26"),
            row("DCOILWTICO", "2020-05-11", "2020-05-27", "")])
        self.brent = PointInTimeSeries("DCOILBRENTEU", [
            row("DCOILBRENTEU", "2020-05-01", "2020-05-01", "33")])
        self.forecast = date(2020, 5, 4)
        features, provenance, eligibility = build_features(self.wti, self.brent, self.forecast)
        self.feature_rows = [{
            "row_id": "example", "forecast_date": self.forecast.isoformat(),
            "features": features, "provenance": provenance, "eligibility": eligibility,
        }]
        self.events = [{"row_id": "example", **event} for event in build_label_events(
            self.wti, self.forecast, date(2020, 5, 11))]
        self.maximum_as_of = date(2020, 5, 28)

    def select(self, as_of, feature_set="wti"):
        return training_rows(self.feature_rows, self.events, as_of,
                             self.maximum_as_of, feature_set)

    def test_only_active_label_is_used_at_each_training_cutoff(self):
        self.assertEqual(self.select(date(2020, 5, 13)), [])
        self.assertEqual(self.select(date(2020, 5, 14))[0]["label"], 1)
        self.assertEqual(self.select(date(2020, 5, 20))[0]["label"], 1)
        self.assertEqual(self.select(date(2020, 5, 21))[0]["label"], 0)
        self.assertEqual(self.select(date(2020, 5, 27))[0]["label"], 0)
        self.assertEqual(self.select(date(2020, 5, 28)), [])

    def test_future_forecast_and_ineligible_row_are_not_selected(self):
        future = {**self.feature_rows[0], "row_id": "future", "forecast_date": "2020-05-25"}
        ineligible = {**self.feature_rows[0], "row_id": "ineligible",
                      "eligibility": {"wti": False, "wti_brent": False}}
        self.feature_rows.extend((future, ineligible))
        original_events = list(self.events)
        for row_id in ("future", "ineligible"):
            self.events.extend({**event, "row_id": row_id} for event in original_events)
        selected = self.select(date(2020, 5, 14))
        self.assertEqual([item["row_id"] for item in selected], ["example"])
        self.assertEqual(self.select(date(2020, 5, 3)), [])

    def test_feature_allowlist_excludes_outcomes_and_future_provenance(self):
        self.feature_rows[0]["features"].update({
            "target_price": 31, "future_revision_count": 2,
            "vintage_end": "2020-05-19", "label": 1,
        })
        for feature_set in ("wti", "wti_brent"):
            selected = self.select(date(2020, 5, 14), feature_set)[0]
            self.assertEqual(set(selected["features"]), set(FEATURE_SETS[feature_set]))
            self.assertEqual(set(selected), {"row_id", "forecast_date", "features", "label"})
            self.assertIsNone(selected["features"]["wti_change_5_business_days_usd"])

    def test_selection_does_not_depend_on_event_input_order(self):
        expected = self.select(date(2020, 5, 21))
        self.events.reverse()
        self.assertEqual(self.select(date(2020, 5, 21)), expected)

    def test_training_cutoff_cannot_extrapolate_beyond_snapshot(self):
        with self.assertRaisesRegex(ValueError, "exceeds"):
            self.select(date(2020, 5, 29))
        self.assertEqual(self.select(self.maximum_as_of), [])

    def test_unavailable_or_missing_event_history_is_excluded(self):
        self.events = []
        self.assertEqual(self.select(date(2020, 5, 14)), [])


if __name__ == "__main__":
    unittest.main()

"""Training outcomes must follow endpoint availability and later withdrawals."""

from datetime import date
import unittest

from oil_audit.availability import PointInTimeSeries
from oil_audit.labels import build_label_events, select_label


def row(observed, available, end, value):
    return {"period_start_date": observed, "realtime_start_date": available,
            "realtime_end_date": end, "DCOILWTICO": value}


class LabelTests(unittest.TestCase):
    def test_waits_for_both_endpoints_then_updates_and_withdraws(self):
        series = PointInTimeSeries("DCOILWTICO", [
            row("2020-04-20", "2020-04-29", "", "-36.98"),
            row("2020-04-27", "2020-04-28", "2020-05-05", "12.78"),
            row("2020-04-27", "2020-05-06", "2020-05-12", "-40"),
            row("2020-04-27", "2020-05-13", "", "")])
        events = build_label_events(series, date(2020, 4, 20), date(2020, 4, 27))
        self.assertIsNone(select_label(events, date(2020, 4, 19)))
        self.assertEqual(select_label(events, date(2020, 4, 20))["label_status"], "missing_both")
        self.assertEqual(select_label(events, date(2020, 4, 29))["label_status"], "missing_origin")
        self.assertEqual(select_label(events, date(2020, 4, 30))["label"], 1)
        self.assertEqual(select_label(events, date(2020, 5, 6))["label"], 1)
        self.assertEqual(select_label(events, date(2020, 5, 7))["label"], 0)
        withdrawn = select_label(events, date(2020, 5, 14))
        self.assertEqual(withdrawn["label_status"], "missing_target")
        self.assertIsNone(withdrawn["target_price"])
        self.assertIsNone(withdrawn["label"])
        for earlier, later in zip(events, events[1:]):
            self.assertEqual(earlier["available_until"], later["available_from"])

    def test_weekend_revisions_collapse_to_latest_state(self):
        series = PointInTimeSeries("DCOILWTICO", [
            row("2020-05-04", "2020-05-05", "", "30"),
            row("2020-05-11", "2020-05-15", "2020-05-15", "31"),
            row("2020-05-11", "2020-05-16", "2020-05-16", "29"),
            row("2020-05-11", "2020-05-17", "", "32")])
        events = build_label_events(series, date(2020, 5, 4), date(2020, 5, 11))
        monday = select_label(events, date(2020, 5, 18))
        self.assertEqual(monday["target_price"], 32)
        self.assertEqual(monday["target_vintage_start"], "2020-05-17")
        self.assertEqual(len([event for event in events
                              if event["available_from"] == "2020-05-18"]), 1)
        self.assertIsNone(select_label(events, date(2020, 5, 17))["label"])

    def test_same_date_publication_is_usable_on_next_business_day_and_ties_are_zero(self):
        series = PointInTimeSeries("DCOILWTICO", [
            row("2020-05-04", "2020-05-04", "", "30"),
            row("2020-05-11", "2020-05-11", "", "30")])
        events = build_label_events(series, date(2020, 5, 4), date(2020, 5, 11))
        self.assertIsNone(select_label(events, date(2020, 5, 11))["label"])
        self.assertEqual(select_label(events, date(2020, 5, 12))["label"], 0)

    def test_predated_target_stays_quarantined(self):
        series = PointInTimeSeries("DCOILWTICO", [
            row("2012-12-19", "2012-12-20", "", "90"),
            row("2012-12-27", "2012-12-26", "", "91")])
        events = build_label_events(series, date(2012, 12, 19), date(2012, 12, 27))
        later = select_label(events, date(2020, 1, 2))
        self.assertEqual(later["label_status"], "quarantined")
        self.assertIsNone(later["target_price"])
        self.assertIsNone(later["label"])

    def test_missing_exact_target_is_not_replaced_with_next_observation(self):
        series = PointInTimeSeries("DCOILWTICO", [
            row("2020-05-04", "2020-05-05", "", "30"),
            row("2020-05-12", "2020-05-13", "", "31")])
        events = build_label_events(series, date(2020, 5, 4), date(2020, 5, 11))
        self.assertEqual(select_label(events, date(2020, 5, 20))["label_status"], "missing_target")

    def test_future_revision_cannot_change_past_label(self):
        original = [row("2020-05-04", "2020-05-05", "", "30"),
                    row("2020-05-11", "2020-05-13", "", "31")]
        revised = [original[0],
                   row("2020-05-11", "2020-05-13", "2020-05-19", "31"),
                   row("2020-05-11", "2020-05-20", "", "29")]
        earlier = []
        for rows in (original, revised):
            events = build_label_events(PointInTimeSeries("DCOILWTICO", rows),
                                        date(2020, 5, 4), date(2020, 5, 11))
            event = select_label(events, date(2020, 5, 14))
            # The future event boundary is provenance, not a historical input.
            earlier.append({key: value for key, value in event.items()
                            if key != "available_until"})
        self.assertEqual(*earlier)

    def test_finite_final_interval_cannot_leave_label_active_forever(self):
        series = PointInTimeSeries("DCOILWTICO", [
            row("2020-05-04", "2020-05-05", "", "30"),
            row("2020-05-11", "2020-05-13", "2020-05-19", "31")])
        events = build_label_events(series, date(2020, 5, 4), date(2020, 5, 11))
        self.assertEqual(select_label(events, date(2020, 5, 20))["label"], 1)
        self.assertEqual(select_label(events, date(2020, 5, 21))["label_status"], "missing_target")


if __name__ == "__main__":
    unittest.main()

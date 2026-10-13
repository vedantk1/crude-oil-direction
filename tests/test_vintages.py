"""Prevent future leakage and accidental resurrection of deleted values."""

import unittest

from oil_audit.vintages import snapshot


def row(observed, available, end, value):
    return {"period_start_date": observed, "realtime_start_date": available,
            "realtime_end_date": end, "DCOILWTICO": value}


class VintageTests(unittest.TestCase):
    def test_deleted_value_does_not_reappear(self):
        history = [row("2011-07-22", "2011-07-27", "2026-09-15", "99.53"),
                   row("2011-07-22", "2026-09-16", "", "")]
        self.assertEqual(snapshot(history, "2026-09-15")["2011-07-22"]["DCOILWTICO"], "99.53")
        self.assertEqual(snapshot(history, "2026-09-16")["2011-07-22"]["DCOILWTICO"], "")

    def test_predated_observation_stays_quarantined(self):
        history = [row("2012-12-27", "2012-12-26", "", "90.91")]
        self.assertEqual(snapshot(history, "2012-12-26"), {})
        self.assertEqual(snapshot(history, "2012-12-28"), {})
        self.assertIn("2012-12-27", snapshot(history, "2012-12-26", strict=False))

    def test_no_false_history_before_first_vintage(self):
        history = [row("1986-01-02", "2011-04-06", "", "25.56")]
        self.assertEqual(snapshot(history, "2000-01-01"), {})
        self.assertIn("1986-01-02", snapshot(history, "2011-04-06"))

    def test_same_day_observation_waits_for_next_day_in_strict_mode(self):
        history = [row("2020-04-01", "2020-04-01", "", "20.28")]
        self.assertEqual(snapshot(history, "2020-04-01"), {})
        self.assertIn("2020-04-01", snapshot(history, "2020-04-02"))

    def test_overlapping_versions_are_rejected(self):
        history = [row("2020-03-06", "2020-03-11", "2020-04-22", "41.14"),
                   row("2020-03-06", "2020-04-22", "", "45.90")]
        with self.assertRaisesRegex(ValueError, "Overlapping"):
            snapshot(history, "2020-04-22")


if __name__ == "__main__":
    unittest.main()

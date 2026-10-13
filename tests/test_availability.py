"""Date-level archive lookups must not expose later revisions or revive gaps."""

from datetime import date
import unittest

from oil_audit.availability import PointInTimeSeries


def row(observed, available, end, value):
    return {"period_start_date": observed, "realtime_start_date": available,
            "realtime_end_date": end, "DCOILWTICO": value}


class AvailabilityTests(unittest.TestCase):
    def test_later_revision_does_not_change_earlier_value(self):
        original = PointInTimeSeries("DCOILWTICO", [
            row("2020-03-18", "2020-03-25", "", "26.96")])
        revised = PointInTimeSeries("DCOILWTICO", [
            row("2020-03-18", "2020-03-25", "2020-03-31", "26.96"),
            row("2020-03-18", "2020-04-01", "", "20.48")])
        cutoff = date(2020, 3, 30)
        self.assertEqual({day: version.value for day, version in original.usable_state(cutoff).items()},
                         {day: version.value for day, version in revised.usable_state(cutoff).items()})
        self.assertEqual(revised.first_vintage, date(2020, 3, 25))
        self.assertEqual(revised.last_vintage, date(2020, 4, 1))

    def test_missing_revision_never_resurrects_old_price(self):
        series = PointInTimeSeries("DCOILWTICO", [
            row("2011-07-22", "2011-07-27", "2026-09-15", "99.53"),
            row("2011-07-22", "2026-09-16", "", "")])
        self.assertEqual(len(series.usable_state(date(2026, 9, 15))), 1)
        self.assertIsNone(series.state(date(2026, 9, 16))[date(2011, 7, 22)].value)
        self.assertEqual(series.usable_state(date(2026, 9, 16)), {})

    def test_predated_record_is_quarantined_at_every_later_cutoff(self):
        series = PointInTimeSeries("DCOILWTICO", [
            row("2012-12-27", "2012-12-26", "", "90.91")])
        for cutoff in (date(2012, 12, 26), date(2012, 12, 27), date(2020, 1, 2)):
            self.assertTrue(series.state(cutoff)[date(2012, 12, 27)].quarantined)
            self.assertEqual(series.usable_state(cutoff), {})

    def test_same_date_is_allowed_at_archive_cutoff(self):
        series = PointInTimeSeries("DCOILWTICO", [
            row("2020-04-01", "2020-04-01", "", "20.28")])
        self.assertEqual(series.state(date(2020, 3, 31)), {})
        self.assertEqual(series.usable_state(date(2020, 4, 1))[date(2020, 4, 1)].value, 20.28)

    def test_invalid_intervals_fail(self):
        for end, expected in (("2020-04-01", "Overlapping"),
                              ("2020-03-30", "Gap"),
                              ("", "Overlapping")):
            with self.subTest(end=end), self.assertRaisesRegex(ValueError, expected):
                PointInTimeSeries("DCOILWTICO", [
                    row("2020-03-18", "2020-03-25", end, "26.96"),
                    row("2020-03-18", "2020-04-01", "", "20.48")])
        with self.assertRaisesRegex(ValueError, "precedes"):
            PointInTimeSeries("DCOILWTICO", [
                row("2020-03-18", "2020-03-25", "2020-03-24", "26.96")])

    def test_nonfinite_prices_fail(self):
        for value in ("NaN", "inf", "-inf"):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "Non-finite"):
                PointInTimeSeries("DCOILWTICO", [
                    row("2020-03-18", "2020-03-25", "", value)])


if __name__ == "__main__":
    unittest.main()

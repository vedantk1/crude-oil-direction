"""Checks for source defects that could silently corrupt forecasting labels."""

from datetime import date
import unittest

from oil_audit.prices import audit_series, compare_calendars


class PriceAuditTests(unittest.TestCase):
    def test_duplicate_dates_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            audit_series([(date(2020, 4, 20), -36.98), (date(2020, 4, 20), 8.91)])

    def test_unsorted_dates_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "increasing"):
            audit_series([(date(2020, 4, 21), 8.91), (date(2020, 4, 20), -36.98)])

    def test_negative_price_is_preserved_and_gap_is_not_filled(self):
        result = audit_series([
            (date(2020, 4, 17), 18.31),
            (date(2020, 4, 20), -36.98),
            (date(2020, 4, 21), None),
            (date(2020, 4, 22), 13.64),
        ])
        self.assertEqual(result["numeric_observations"], 3)
        self.assertEqual(result["explicit_empty_price_rows"], 1)
        self.assertEqual(result["weekdays_without_price"], ["2020-04-21"])
        self.assertEqual(result["nonpositive_observations"], [{"date": "2020-04-20", "value": -36.98}])
        self.assertEqual(result["largest_absolute_adjacent_changes"][0]["change_usd"], -55.29)

    def test_calendars_keep_asymmetric_missing_dates(self):
        result = compare_calendars({
            "wti": [(date(2023, 12, 22), 73.56), (date(2023, 12, 26), 75.84), (date(2023, 12, 29), 71.89)],
            "brent": [(date(2023, 12, 22), 80.23), (date(2023, 12, 29), 77.69)],
        })
        self.assertEqual(result["both_have_price"], 2)
        self.assertEqual(result["wti_only_dates"], ["2023-12-26"])

    def test_nonfinite_values_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            audit_series([(date(2020, 4, 20), float("nan"))])


if __name__ == "__main__":
    unittest.main()

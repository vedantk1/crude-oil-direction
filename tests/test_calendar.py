"""Check historical holiday transitions and deterministic horizon boundaries."""

import unittest
from datetime import date, datetime

from oil_audit.calendar import (
    add_business_days,
    is_business_day,
    next_business_day,
    previous_business_day,
)


class CalendarTests(unittest.TestCase):
    def test_weekends_and_strict_next_previous(self):
        self.assertFalse(is_business_day(date(2026, 3, 7)))
        self.assertFalse(is_business_day(date(2026, 3, 8)))
        self.assertEqual(next_business_day(date(2026, 3, 6)), date(2026, 3, 9))
        self.assertEqual(previous_business_day(date(2026, 3, 9)), date(2026, 3, 6))
        self.assertEqual(next_business_day(date(2026, 3, 9)), date(2026, 3, 10))

    def test_new_year_observed_in_previous_year(self):
        self.assertFalse(is_business_day(date(2021, 12, 31)))
        self.assertEqual(next_business_day(date(2021, 12, 30)), date(2022, 1, 3))
        self.assertEqual(previous_business_day(date(2022, 1, 3)), date(2021, 12, 30))

    def test_sunday_holiday_observed_monday(self):
        self.assertFalse(is_business_day(date(2023, 1, 2)))
        self.assertEqual(next_business_day(date(2022, 12, 30)), date(2023, 1, 3))

    def test_juneteenth_begins_in_2021(self):
        self.assertTrue(is_business_day(date(2020, 6, 19)))
        self.assertFalse(is_business_day(date(2021, 6, 18)))
        self.assertFalse(is_business_day(date(2022, 6, 20)))
        self.assertFalse(is_business_day(date(2023, 6, 19)))
        self.assertEqual(next_business_day(date(2021, 6, 17)), date(2021, 6, 21))

    def test_opm_2026_recurring_holidays(self):
        # Dates from OPM's 2026 holiday schedule, independent of the formulas.
        holidays = [
            (1, 1), (1, 19), (2, 16), (5, 25), (6, 19), (7, 3),
            (9, 7), (10, 12), (11, 11), (11, 26), (12, 25),
        ]
        for month, day in holidays:
            with self.subTest(month=month, day=day):
                self.assertFalse(is_business_day(date(2026, month, day)))

    def test_federal_proxy_is_not_exchange_or_exceptional_closure_calendar(self):
        self.assertTrue(is_business_day(date(2026, 4, 3)))  # Good Friday
        self.assertTrue(is_business_day(date(2017, 1, 20)))  # Inauguration Day

    def test_five_day_horizon_crossing_holiday(self):
        self.assertEqual(add_business_days(date(2026, 6, 15), 5), date(2026, 6, 23))
        self.assertEqual(add_business_days(date(2026, 6, 23), -5), date(2026, 6, 15))
        self.assertEqual(add_business_days(date(2021, 12, 23), 5), date(2022, 1, 3))

    def test_holiday_start_and_zero_count(self):
        self.assertEqual(add_business_days(date(2026, 7, 3), 1), date(2026, 7, 6))
        self.assertEqual(add_business_days(date(2026, 7, 3), -1), date(2026, 7, 2))
        self.assertEqual(add_business_days(date(2026, 7, 3), 0), date(2026, 7, 3))

    def test_unsupported_dates_and_crossing_range_raise(self):
        for day in (date(2010, 12, 31), date(2101, 1, 1)):
            with self.subTest(day=day), self.assertRaisesRegex(ValueError, "2011-01-01"):
                is_business_day(day)
        with self.assertRaises(ValueError):
            previous_business_day(date(2011, 1, 1))
        with self.assertRaises(ValueError):
            next_business_day(date(2100, 12, 31))

    def test_reject_datetime_and_noninteger_counts(self):
        with self.assertRaises(TypeError):
            is_business_day(datetime(2026, 6, 15, 9))
        for count in (True, 1.5, "5"):
            with self.subTest(count=count), self.assertRaises(TypeError):
                add_business_days(date(2026, 6, 15), count)


if __name__ == "__main__":
    unittest.main()

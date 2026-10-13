"""Calendar-window selection never consults outcomes or removes missing inputs."""

from copy import deepcopy
from datetime import date
import unittest

from oil_eval.window import ROLLING_WINDOW_YEARS, rolling_window_start, select_rolling_training_rows


class RollingWindowTests(unittest.TestCase):
    def test_fixed_calendar_years_and_leap_day_clamp(self):
        self.assertEqual(ROLLING_WINDOW_YEARS, 4)
        self.assertEqual(rolling_window_start(date(2017, 1, 5)), date(2013, 1, 5))
        self.assertEqual(rolling_window_start(date(2020, 2, 29)), date(2016, 2, 29))
        self.assertEqual(rolling_window_start(date(2104, 2, 29)), date(2100, 2, 28))
        self.assertEqual(rolling_window_start(date(2021, 3, 1)), date(2017, 3, 1))

    def test_lower_boundary_inclusive_and_input_order_values_unchanged(self):
        rows = [{"row_id": str(index), "forecast_date": day,
                 "features": {"inventory": None}, "label": index % 2}
                for index, day in enumerate(("2016-12-29", "2013-01-04", "2013-01-05", "2015-08-20"))]
        before = deepcopy(rows)
        result = select_rolling_training_rows(rows, date(2017, 1, 5))
        self.assertEqual(result, [rows[0], rows[2], rows[3]])
        self.assertTrue(all(actual is original for actual, original in zip(result, (rows[0], rows[2], rows[3]))))
        self.assertEqual(rows, before)
        self.assertTrue(all(row["features"]["inventory"] is None for row in result))

    def test_current_or_future_rows_rejected_instead_of_silently_admitted(self):
        for day in ("2017-01-05", "2017-01-06", "2023-01-01"):
            with self.subTest(day=day), self.assertRaisesRegex(ValueError, "strictly earlier"):
                select_rolling_training_rows([{"forecast_date": day}], date(2017, 1, 5))

    def test_does_not_choose_labels_or_override_history_requirements(self):
        # Minimal row objects deliberately carry no labels or features: only
        # replay can determine their historical eligibility before selection.
        rows = [{"forecast_date": "2014-01-01"}, {"forecast_date": "2012-01-01"}]
        self.assertEqual(select_rolling_training_rows(rows, date(2017, 1, 5)), rows[:1])
        self.assertEqual(select_rolling_training_rows(rows, date(2021, 1, 5)), [])
        self.assertEqual(select_rolling_training_rows([], date(2017, 1, 5)), [])


if __name__ == "__main__":
    unittest.main()

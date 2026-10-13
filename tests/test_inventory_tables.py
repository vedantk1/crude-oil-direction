"""Synthetic schema/availability checks; no historical inventory fixtures."""

from copy import deepcopy
import csv
from datetime import date, datetime
import io
import unittest

from oil_audit.inventory_tables import read_table4


def rows():
    return [
        ["STUB_1", "1/7/22", "12/31/21", "Difference", "1/8/21",
         "Percent Change", "1/10/20", "Percent Change"],
        ["Crude Oil", "900.000", "800.000", "100.000", "0", "0", "0", "0"],
        ["Commercial (Excluding SPR)", "400.125", "399.125", "1.000", "0", "0", "0", "0"],
        ["Cushing", "20.250", "21.750", "-1.500", "0", "0", "0", "0"],
    ]


def payload(content=None, encoding="utf-8"):
    output = io.StringIO()
    csv.writer(output).writerows(rows() if content is None else content)
    return output.getvalue().encode(encoding)


class InventoryTableTests(unittest.TestCase):
    def load(self, content=None, **kwargs):
        return read_table4(payload(content), date(2022, 1, 12), **kwargs)

    def test_exact_series_and_duplicate_comparison_headers(self):
        result = self.load(expected_observation_date=date(2022, 1, 7))
        self.assertEqual(result["current_date"], date(2022, 1, 7))
        self.assertEqual(result["previous_date"], date(2021, 12, 31))
        self.assertEqual(set(result["series"]), {"commercial_crude", "cushing"})
        self.assertEqual(result["series"]["commercial_crude"]["current_million_barrels"], 400.125)
        self.assertEqual(result["series"]["cushing"]["published_change_million_barrels"], -1.5)
        self.assertEqual(result["series"]["cushing"]["rounding_residual"], 0)

    def test_utf8_bom_and_legacy_cp1252_notes(self):
        content = rows() + [["Notes: independent rounding – see definitions"]]
        for encoding in ("utf-8", "utf-8-sig", "cp1252"):
            with self.subTest(encoding=encoding):
                self.assertEqual(read_table4(payload(content, encoding), date(2022, 1, 12)), self.load())

    def test_numeric_commas_and_legitimate_rounding_residual(self):
        content = rows()
        content[2][1:4] = ["1,234.567", "1,232.566", "+2.000"]
        result = self.load(content)["series"]["commercial_crude"]
        self.assertEqual(result["current_million_barrels"], 1234.567)
        self.assertEqual(result["rounded_level_change_million_barrels"], 2.001)
        self.assertEqual(result["rounding_residual"], -0.001)

    def test_inconsistent_difference_is_rejected(self):
        content = rows()
        content[2][3] = "1.002"
        with self.assertRaisesRegex(ValueError, "rounding tolerance"):
            self.load(content)

    def test_negative_levels_missing_nonfinite_and_flagged_values_rejected(self):
        for column in (1, 2, 3):
            for value in ("", "NA", "R400.125", "400.125R", "NaN", "Infinity", "1e4", "4,00.125"):
                content = rows()
                content[2][column] = value
                with self.subTest(column=column, value=value), self.assertRaises(ValueError):
                    self.load(content)
        for column in (1, 2):
            content = rows()
            content[2][column] = "-1.000"
            with self.subTest(column=column), self.assertRaisesRegex(ValueError, "nonnegative"):
                self.load(content)

    def test_required_rows_must_be_exact_unique_and_complete(self):
        variants = [rows()[:-1], rows() + [deepcopy(rows()[2])]]
        changed_label = rows()
        changed_label[2][0] = "Commercial (Including SPR)"
        variants.append(changed_label)
        short = rows()
        short[2].pop()
        variants.append(short)
        for content in variants:
            with self.subTest(content=content), self.assertRaises(ValueError):
                self.load(content)

    def test_unknown_header_layout_is_rejected(self):
        for index, value in ((0, "Series"), (3, "Percent Change"), (5, "Change"), (7, "Change")):
            content = rows()
            content[0][index] = value
            with self.subTest(index=index), self.assertRaisesRegex(ValueError, "header layout"):
                self.load(content)
        content = rows()
        content[0].append("Extra")
        with self.assertRaisesRegex(ValueError, "header layout"):
            self.load(content)

    def test_current_and_previous_dates_are_exact_consecutive_fridays(self):
        for current, prior in (("1/6/22", "12/30/21"), ("1/7/22", "12/24/21"),
                               ("12/31/21", "1/7/22"), ("1/99/22", "12/31/21"),
                               ("2022-01-07", "12/31/21")):
            content = rows()
            content[0][1:3] = [current, prior]
            with self.subTest(current=current, prior=prior), self.assertRaises(ValueError):
                self.load(content)

    def test_comparison_dates_are_valid_and_earlier(self):
        for index, value in ((4, "1/7/22"), (6, "1/15/21"), (4, "Year ago")):
            content = rows()
            content[0][index] = value
            with self.subTest(index=index, value=value), self.assertRaises(ValueError):
                self.load(content)

    def test_observation_must_precede_actual_release_and_match_issue(self):
        for released in (date(2022, 1, 7), date(2022, 1, 6)):
            with self.subTest(released=released), self.assertRaisesRegex(ValueError, "precede"):
                read_table4(payload(), released)
        with self.assertRaisesRegex(ValueError, "issue/index"):
            self.load(expected_observation_date=date(2022, 1, 14))
        # Delayed releases are legitimate; never infer a maximum lag or URL date.
        self.assertEqual(read_table4(payload(), date(2022, 1, 26))["current_date"], date(2022, 1, 7))

    def test_empty_malformed_and_wrong_argument_types_rejected(self):
        for data in (b"", b'"unterminated', "not bytes"):
            with self.subTest(data=data), self.assertRaises(ValueError):
                read_table4(data, date(2022, 1, 12))
        for released in ("2022-01-12", datetime(2022, 1, 12)):
            with self.subTest(released=released), self.assertRaises(ValueError):
                read_table4(payload(), released)
        with self.assertRaises(ValueError):
            self.load(expected_observation_date="2022-01-07")


if __name__ == "__main__":
    unittest.main()

"""Only admitted ALFRED values may reach Unicode, CSV or numeric parsing."""

from datetime import date
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from oil_audit.availability import PointInTimeSeries
from oil_audit.bounded_vintages import bounded_records


def write_archive(root, rows, series="DCOILWTICO", part=1, header=None):
    path = root / f"{series}-realtime-part{part}.zip"
    header = header or f"period_start_date,{series},realtime_start_date,realtime_end_date\n".encode()
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("intervals.csv", header + b"".join(row + b"\n" for row in rows))
    return path


class BoundedVintageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_observation_and_vintage_gates_precede_unicode_csv_and_numeric_values(self):
        write_archive(self.root, [
            b"2022-12-28,70,2022-12-29,2026-01-01",
            b'2023-01-03,"\xff,not,a,price,2022-12-29,',
            b'2011-04-05,"\xff,not,a,price,2023-01-04,',
        ])
        records = bounded_records("DCOILWTICO", self.root, date(2022, 12, 31))
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["DCOILWTICO"], "70")
        self.assertEqual(records[0]["realtime_end_date"], "")
        series = PointInTimeSeries("DCOILWTICO", records)
        self.assertEqual(series.last_vintage, date(2022, 12, 29))
        self.assertIsNone(series.versions_by_observation[date(2022, 12, 28)][0].vintage_end)

    def test_entirely_future_rows_never_call_csv_decoder(self):
        write_archive(self.root, [b"2023-01-03,\xff,2023-01-04,"])
        with patch("oil_audit.bounded_vintages.csv.reader", side_effect=AssertionError("must stay gated")):
            self.assertEqual(bounded_records("DCOILWTICO", self.root, date(2022, 12, 31)), [])

    def test_generic_brent_deduplication_sorting_and_inclusive_boundary(self):
        rows = [b'"2022-12-30","80.5","2022-12-31",""',
                b"2022-12-29,79,2022-12-30,2022-12-31"]
        write_archive(self.root, rows, series="DCOILBRENTEU")
        write_archive(self.root, rows[:1], series="DCOILBRENTEU", part=2)
        result = bounded_records("DCOILBRENTEU", self.root, date(2022, 12, 31))
        self.assertEqual([row["period_start_date"] for row in result], ["2022-12-29", "2022-12-30"])
        self.assertEqual(result[0]["realtime_end_date"], "2022-12-31")
        self.assertEqual(result[-1]["DCOILBRENTEU"], "80.5")

    def test_missing_withdrawals_and_quarantine_keep_original_pit_semantics(self):
        write_archive(self.root, [b"2020-04-01,-36.98,2020-04-01,2020-04-02",
                                 b"2020-04-01,.,2020-04-03,",
                                 b"2020-04-06,99,2020-04-02,"])
        series = PointInTimeSeries("DCOILWTICO", bounded_records("DCOILWTICO", self.root, date(2020, 4, 30)))
        self.assertEqual(series.usable_state(date(2020, 4, 2))[date(2020, 4, 1)].value, -36.98)
        self.assertEqual(series.usable_state(date(2020, 4, 7)), {})
        self.assertTrue(series.state(date(2020, 4, 7))[date(2020, 4, 6)].quarantined)

    def test_admitted_corruption_and_bad_schemas_fail_without_filling_values(self):
        for rows, header in (
            ([b"2022-12-28,\xff,2022-12-29,"], None),
            ([b"2022-12-28,70,2022-12-29,2022-12-27"], None),
            ([b"bad-date,70,2022-12-29,"], None),
            ([b"2022-12-28,70,2022-12-29,"], b"wrong,header\n"),
        ):
            write_archive(self.root, rows, header=header)
            with self.assertRaises(ValueError):
                bounded_records("DCOILWTICO", self.root, date(2022, 12, 31))
        write_archive(self.root, [b"2022-12-28,NaN,2022-12-29,"])
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            PointInTimeSeries("DCOILWTICO", bounded_records("DCOILWTICO", self.root, date(2022, 12, 31)))


if __name__ == "__main__":
    unittest.main()

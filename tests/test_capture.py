from datetime import datetime, timezone
import unittest

from oil_audit.capture import capture_bounds


class CaptureTests(unittest.TestCase):
    def test_separate_earliest_file_bound_for_each_series(self):
        bounds = capture_bounds([
            {"path": "W-part1.zip", "retrieved_at_utc": "2020-05-01T12:00:00+00:00"},
            {"path": "W-part2.zip", "retrieved_at_utc": "2020-05-02T12:00:00+00:00"},
            {"path": "B-part1.zip", "retrieved_at_utc": "2020-04-30T12:00:00+00:00"},
        ], ("W", "B"))
        self.assertEqual(bounds["W"], datetime(2020, 5, 1, 12, tzinfo=timezone.utc))
        self.assertEqual(bounds["B"], datetime(2020, 4, 30, 12, tzinfo=timezone.utc))

    def test_date_only_does_not_invent_time_or_timezone(self):
        bounds = capture_bounds([{"path": "W-part1.zip", "retrieval_date": "2020-05-01"}], ("W",))
        self.assertEqual(bounds["W"], datetime(2020, 4, 30, tzinfo=timezone.utc))

    def test_missing_or_naive_provenance_rejected(self):
        for records in ([], [{"path": "W-part1.zip"}],
                        [{"path": "W-part1.zip", "retrieved_at_utc": "2020-05-01T12:00:00"}]):
            with self.assertRaises(ValueError):
                capture_bounds(records, ("W",))

from datetime import date
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from oil_audit.availability import PointInTimeSeries
from oil_audit.dataset import (build, construct_rows, forecast_schedule, jsonl_bytes,
                               read_dataset, sha256, summarize, workspace_path)
from argparse import ArgumentTypeError


def raw(series_id, observed, start, value, end=""):
    return {"period_start_date": observed, "realtime_start_date": start,
            "realtime_end_date": end, series_id: str(value)}


class DatasetTests(unittest.TestCase):
    def test_schedule_merges_weekend_updates_and_observes_holidays(self):
        schedule = forecast_schedule(
            [date(2020, 7, 2), date(2020, 7, 3), date(2020, 7, 4), date(2020, 7, 4)],
            date(2011, 1, 1))
        self.assertEqual(schedule, [(date(2020, 7, 6),
                                    [date(2020, 7, 2), date(2020, 7, 3), date(2020, 7, 4)])])

    def test_first_evaluation_label_survives_withdrawal_but_training_summary_does_not(self):
        wti = PointInTimeSeries("W", [
            raw("W", "2020-04-29", "2020-04-30", 18),
            raw("W", "2020-05-01", "2020-05-06", 20),
            raw("W", "2020-05-08", "2020-05-13", 21, "2020-05-19"),
            raw("W", "2020-05-08", "2020-05-20", ""),
        ])
        brent = PointInTimeSeries("B", [raw("B", "2020-04-28", "2020-04-30", 25)])
        rows, events, evaluation = construct_rows(wti, brent, [date(2020, 4, 30)], date(2011, 1, 1))
        row = rows[0]
        self.assertEqual(row["forecast_date"], "2020-05-01")
        self.assertEqual(row["forecast_timestamp"], "2020-05-01T18:00:00-04:00")
        self.assertEqual(row["information_cutoff_date"], "2020-04-30")
        self.assertEqual(row["target_date"], "2020-05-08")
        self.assertEqual(evaluation[0]["label"], 1)
        self.assertEqual(evaluation[0]["available_from"], "2020-05-14")
        self.assertNotIn("available_until", evaluation[0])
        summary = summarize(rows, events, evaluation, date(2020, 5, 21))
        self.assertEqual(summary["previously_available_now_unavailable"], 1)
        self.assertEqual(summary["training_rows_at_replay_end"]["wti"], 0)
        _, bounded_events, bounded_evaluation = construct_rows(
            wti, brent, [date(2020, 4, 30)], date(2011, 1, 1), date(2020, 5, 7))
        self.assertEqual(bounded_evaluation, [])
        self.assertTrue(all(event["available_from"] <= "2020-05-07" for event in bounded_events))

    def test_five_business_day_target_is_not_shifted_to_existing_prices(self):
        wti = PointInTimeSeries("W", [raw("W", "2020-06-30", "2020-07-01", 20)])
        brent = PointInTimeSeries("B", [raw("B", "2020-06-30", "2020-07-01", 22)])
        rows, events, evaluation = construct_rows(wti, brent, [date(2020, 7, 1)], date(2011, 1, 1))
        self.assertEqual(rows[0]["forecast_date"], "2020-07-02")
        self.assertEqual(rows[0]["target_date"], "2020-07-10")
        self.assertEqual(events[0]["label_status"], "missing_both")
        self.assertEqual(evaluation, [])

    def test_dataset_read_rejects_modified_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = jsonl_bytes([{"sample": 1}])
            (root / "features.jsonl").write_bytes(payload)
            (root / "manifest.json").write_text(json.dumps({"artifacts": {
                "features.jsonl": {"sha256": sha256(payload), "bytes": len(payload)}}}))
            _, data = read_dataset(root)
            self.assertEqual(data["features.jsonl"], [{"sample": 1}])
            (root / "features.jsonl").write_bytes(jsonl_bytes([{"sample": 2}]))
            with self.assertRaisesRegex(ValueError, "checksum"):
                read_dataset(root)

    def test_build_is_repeatable_and_will_not_replace_modified_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "manifest.json").write_text(json.dumps([
                {"path": series_id + "-realtime-part1.zip", "retrieval_date": "2020-05-10"}
                for series_id in ("DCOILWTICO", "DCOILBRENTEU")]))
            with patch("oil_audit.dataset.vintages.verify_manifest"), \
                 patch("oil_audit.dataset.vintages.form_metadata", return_value=(["2020-04-30"], [])), \
                 patch("oil_audit.dataset.vintages.records", side_effect=lambda series_id, _: [
                     raw(series_id, "2020-04-29", "2020-04-30", 20)]):
                first = build(root, root / "datasets", report=root / "report.json")
                second = build(root, root / "datasets", report=root / "report.json")
                self.assertEqual(first, second)
                destination = Path(first["dataset_directory"])
                read_dataset(destination)
                (destination / "features.jsonl").write_text("{}\n")
                with self.assertRaisesRegex(ValueError, "Refusing to replace"):
                    build(root, root / "datasets", report=root / "report.json")

    def test_cli_paths_stay_workspace_relative(self):
        self.assertEqual(workspace_path("data/datasets"), Path("data/datasets"))
        for value in ("/tmp/output", "../output", "data/../../output"):
            with self.assertRaises(ArgumentTypeError):
                workspace_path(value)

    def test_build_cannot_issue_forecast_after_either_series_capture(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "manifest.json").write_text(json.dumps([
                {"path": "DCOILWTICO-realtime-part1.zip", "retrieved_at_utc": "2020-05-02T20:00:00+00:00"},
                {"path": "DCOILBRENTEU-realtime-part1.zip", "retrieved_at_utc": "2020-05-01T21:00:00+00:00"},
            ]))
            with patch("oil_audit.dataset.vintages.verify_manifest"), \
                 patch("oil_audit.dataset.vintages.form_metadata", return_value=(["2020-04-30"], [])), \
                 patch("oil_audit.dataset.vintages.records", side_effect=lambda series_id, _: [
                     raw(series_id, "2020-04-29", "2020-04-30", 20)]):
                with self.assertRaisesRegex(ValueError, "No forecast opportunities"):
                    build(root, root / "datasets", report=root / "report.json")


if __name__ == "__main__":
    unittest.main()

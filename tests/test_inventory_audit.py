"""Synthetic offline inventory snapshots and development availability tests."""

from copy import deepcopy
import csv
from datetime import date, timedelta
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from oil_audit.calendar import next_business_day
from oil_audit.dataset import json_bytes, sha256
from oil_audit.inventory import (audit_config, compare_overlaps, development_coverage, load_snapshot,
                                 run, summarize, _evidence, _probes, _quarantine)
from oil_audit.inventory_sources import (ARCHIVE_INDEX, SUPPLEMENTARY_ISSUES,
                                         _canonical_issues, parse_index)


def csv_payload(observed, current="400.000", previous="399.000", change="1.000", table=4):
    dates = [observed, observed - timedelta(days=7), observed - timedelta(days=364),
             observed - timedelta(days=728)]
    fmt = lambda value: value.strftime("%m/%d/%y")
    header = (["STUB_1", fmt(dates[0]), fmt(dates[1]), "Difference", fmt(dates[2]),
               "Percent Change", fmt(dates[3]), "Percent Change"] if table == 4 else
              ["STUB_1", fmt(dates[0]), fmt(dates[1]), "Difference", "Percent Change",
               fmt(dates[2]), "Difference", "Percent Change"])
    rows = [header, ["Commercial (Excluding SPR)", current, previous, change, "0", "0", "0", "0"],
            ["Cushing", "20.000", "19.000", "1.000", "0", "0", "0", "0"]]
    output = io.StringIO()
    csv.writer(output).writerows(rows)
    return output.getvalue().encode()


def record(issue_id, released, observed, current=400, previous=399, commercial_regime="excludes_lease_stocks"):
    return {
        "issue_id": issue_id, "release_date": released, "observation_date": observed,
        "previous_observation_date": (date.fromisoformat(observed) - timedelta(days=7)).isoformat(),
        "earliest_eligible_issuance_date": next_business_day(date.fromisoformat(released)).isoformat(),
        "series": {name: {
            "current_million_barrels": current, "previous_million_barrels": previous,
            "published_change_million_barrels": current - previous,
            "rounded_level_change_million_barrels": current - previous, "rounding_residual": 0,
            "definition_regime": commercial_regime if name == "commercial_crude" else "cushing_reported",
        } for name in ("commercial_crude", "cushing")},
    }


def forecast(day, target):
    return {"row_id": "wti-direction-" + day.replace("-", ""),
            "forecast_date": day, "target_date": target}


class SnapshotTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix=".test-inventory-", dir=".")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).relative_to(Path.cwd())
        self.snapshot = self.root / "snapshot"
        self.snapshot.mkdir()
        rows = []
        for year in range(2011, 2023):
            observed = date(year, 1, 1)
            observed += timedelta(days=(4 - observed.weekday()) % 7)
            released = observed + timedelta(days=5)
            slug = released.strftime("%Y_%m_%d")
            rows.append(f'<tr><td>January</td><td><a href="{ARCHIVE_INDEX}{year}/{slug}/wpsr_{slug}.php">{released.day}</a></td><td>{observed.day}</td></tr>')
        index = ("<table>" + "".join(rows) + "</table>").encode()
        indexed = parse_index(index)
        issues = _canonical_issues(indexed)
        self.manifest = {
            "schema_version": 1, "complete": True, "all_downloads_succeeded": True,
            "release_years": [2011, 2022], "index": self.capture("archive-index.html", index, ARCHIVE_INDEX),
            "indexed_issue_count": len(indexed), "issues": issues,
            "additional_unindexed_issue_ids": [row["issue_id"] for row in issues
                                                if row["issue_id"] not in {item["issue_id"] for item in indexed}],
            "sources": [], "supplementary_issue_pages": [],
        }
        for issue in issues:
            data = csv_payload(date.fromisoformat(issue["observation_date"]))
            self.manifest["sources"].append({"issue_id": issue["issue_id"], **self.capture(
                issue["issue_id"] + "-table4.csv", data, issue["table_url"])})
        for issue in SUPPLEMENTARY_ISSUES:
            released, observed = map(date.fromisoformat, (issue["release_date"], issue["observation_date"]))
            page = ("<p>Release Date: " + released.strftime("%B %d, %Y")
                    + " Data for week ending " + observed.strftime("%B %d, %Y") + "</p>").encode()
            self.manifest["supplementary_issue_pages"].append({"issue_id": issue["issue_id"], **self.capture(
                issue["issue_id"] + "-issue.html", page, issue["issue_url"])})
        self.save()

    def capture(self, name, payload, url):
        (self.snapshot / name).write_bytes(payload)
        return {"file": name, "requested_url": url, "resolved_url": url, "status": "downloaded",
                "sha256": sha256(payload), "bytes": len(payload)}

    def save(self):
        (self.snapshot / "manifest.json").write_bytes(json_bytes(self.manifest))

    def test_verifies_every_source_and_extracts_release_qualified_records(self):
        identity, issues, records, failures, _ = load_snapshot(self.snapshot)
        self.assertEqual((len(issues), len(records), len(failures)), (15, 15, 0))
        self.assertEqual(len(identity["verified_artifacts"]), 19)
        for row in records:
            self.assertLess(row["observation_date"], row["release_date"])
            self.assertGreater(row["earliest_eligible_issuance_date"], row["release_date"])
        regimes = {row["issue_id"]: row["series"]["commercial_crude"]["definition_regime"] for row in records}
        self.assertEqual(regimes["2011_07_27"], "includes_lease_stocks")
        self.assertEqual(regimes["2016_11_30"], "excludes_lease_stocks")

    def test_checksum_missing_file_and_path_traversal_fail(self):
        source = self.manifest["sources"][0]
        path = self.snapshot / source["file"]
        original = path.read_bytes()
        path.write_bytes(original + b"changed")
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            load_snapshot(self.snapshot)
        path.unlink()
        with self.assertRaisesRegex(ValueError, "Missing declared"):
            load_snapshot(self.snapshot)
        path.write_bytes(original)
        self.manifest["index"]["file"] = "../archive-index.html"
        self.save()
        with self.assertRaises(ValueError):
            load_snapshot(self.snapshot)

    def test_failed_download_and_schema_failures_stay_explicit(self):
        first = self.manifest["sources"][0]
        first["status"] = "download_failed"
        first["http_status"] = 404
        self.manifest["all_downloads_succeeded"] = False
        second = self.manifest["sources"][1]
        bad = b"wrong,schema\n"
        (self.snapshot / second["file"]).write_bytes(bad)
        second.update(bytes=len(bad), sha256=sha256(bad))
        self.save()
        _, issues, records, failures, _ = load_snapshot(self.snapshot)
        self.assertEqual({row["status"] for row in failures}, {"download_failed", "schema_failed"})
        self.assertEqual(len(records), 13)
        _, overlaps = compare_overlaps(records)
        summary = summarize(issues, records, failures, overlaps, [])
        self.assertEqual(summary["failure_count"], 2)
        self.assertIn(issues[0]["observation_date"], summary["unusable_friday_weeks_within_span"])
        self.assertTrue(summary["missing_friday_weeks_within_captured_span"])

    def test_incomplete_identity_scope_and_duplicate_records_rejected(self):
        original = deepcopy(self.manifest)
        for mutate in (
            lambda m: m.update(complete=False),
            lambda m: m.update(release_years=[2011, 2023]),
            lambda m: m["issues"][0].update(release_date="2023-01-04"),
            lambda m: m["sources"].append(deepcopy(m["sources"][0])),
            lambda m: m["sources"].pop(),
            lambda m: m.update(indexed_issue_count=999),
            lambda m: m["sources"][0].update(expected_capture={"sha256": "0" * 64, "bytes": 1}),
        ):
            self.manifest = deepcopy(original)
            mutate(self.manifest)
            self.save()
            with self.subTest(mutation=mutate), self.assertRaises(ValueError):
                load_snapshot(self.snapshot)

    def test_supplemental_page_metadata_must_match_declared_release(self):
        page = self.manifest["supplementary_issue_pages"][0]
        bad = b"<p>Release Date: July 28, 2011</p>"
        (self.snapshot / page["file"]).write_bytes(bad)
        page.update(bytes=len(bad), sha256=sha256(bad))
        self.save()
        with self.assertRaisesRegex(ValueError, "release date"):
            load_snapshot(self.snapshot)

    def test_immutable_rebuild_protects_sources_reference_and_own_artifacts(self):
        reference = {"run_id": "synthetic", "evaluation_directory": str(self.root / "reference")}
        forecasts = [forecast("2017-01-05", "2017-01-12")]
        before = {p.name: p.read_bytes() for p in self.snapshot.iterdir()}
        destination, report = self.root / "audits", self.root / "report.json"
        with patch("oil_audit.inventory._load_saved_run", return_value=(reference, {}, forecasts)), \
             patch("oil_eval.data.load_development_data", side_effect=AssertionError("raw dataset read")), \
             patch("oil_eval.models.Pipeline.fit", side_effect=AssertionError("model fit")):
            first = run(self.snapshot, destination, report)
            self.assertEqual(first, run(self.snapshot, destination, report))
            self.assertEqual(first["development_coverage"]["forecast_count"], 1)
            self.assertEqual(before, {p.name: p.read_bytes() for p in self.snapshot.iterdir()})
            protected = [self.snapshot / "manifest.json", self.snapshot / "archive-index.html", WTI_REPORT]
            for path in protected:
                with self.subTest(path=path), self.assertRaisesRegex(ValueError, "cannot overwrite"):
                    run(self.snapshot, destination, path)
            output = Path(first["audit_directory"]) / "summary.json"
            with self.assertRaisesRegex(ValueError, "cannot overwrite"):
                run(self.snapshot, destination, output)
            with patch("oil_audit.inventory.audit_config", return_value={**audit_config(), "test_identity": "changed"}):
                with self.assertRaisesRegex(ValueError, "cannot overwrite"):
                    run(self.snapshot, destination, output)
                with self.assertRaisesRegex(ValueError, "cannot overwrite"):
                    run(self.snapshot, self.root / "different-audit-root", output)
            output.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "Refusing to replace"):
                run(self.snapshot, destination, report)

    def test_optional_table1_evidence_is_hashed_and_crosschecked(self):
        _, _, records, _, _ = load_snapshot(self.snapshot)
        selected = next(row for row in records if row["issue_id"] == "2011_07_27")
        evidence_root = self.root / "evidence"
        evidence_root.mkdir()
        data = csv_payload(date(2011, 7, 22), table=1)
        source = {"file": "sample.csv", "requested_url": selected["table_url"].replace("table4", "table1"),
                  "sha256": sha256(data), "bytes": len(data)}
        (evidence_root / "sample.csv").write_bytes(data)
        manifest = evidence_root / "manifest.json"
        manifest.write_bytes(json_bytes({"complete": True, "sources": [source]}))
        identity, _, checks = _evidence(manifest, records)
        self.assertIn("sample.csv", identity["verified_artifacts"])
        self.assertTrue(checks[0]["match"])
        changed = deepcopy(records)
        next(row for row in changed if row["issue_id"] == "2011_07_27")["series"]["commercial_crude"]["current_million_barrels"] += 1
        with self.assertRaisesRegex(ValueError, "stocks differ"):
            _evidence(manifest, changed)
        source["requested_url"] = source["requested_url"].replace("/archive/2011/", "/archive/2012/")
        manifest.write_bytes(json_bytes({"complete": True, "sources": [source]}))
        with patch("oil_audit.inventory._verified", side_effect=AssertionError("read source")):
            with self.assertRaisesRegex(ValueError, "canonical issue URL"):
                _evidence(manifest, records)

    def test_future_or_unrecognized_evidence_is_rejected_before_value_file_read(self):
        manifest = self.root / "evidence.json"
        for url in (
            "https://www.eia.gov/petroleum/supply/weekly/archive/2023/2023_01_04/csv/table1.csv",
            "https://www.eia.gov/petroleum/supply/weekly/csv/table4.csv",
            "https://www.eia.gov/pressroom/releases/unknown.php",
        ):
            manifest.write_bytes(json_bytes({"complete": True, "sources": [{"requested_url": url}]}))
            with self.subTest(url=url), patch("oil_audit.inventory._verified", side_effect=AssertionError("read source bytes")):
                with self.assertRaisesRegex(ValueError, "bounded historical scope"):
                    _evidence(manifest, [])

    def test_early_2011_failed_probes_are_bounded_and_qualified(self):
        manifest = self.root / "probes.json"
        source = {"candidate_release_date": "2011-07-20", "file": "2011_07_20-table4.csv",
                  "requested_url": ARCHIVE_INDEX + "2011/2011_07_20/csv/table4.csv",
                  "status": "download_failed", "http_status": 404, "error_type": "HTTPError",
                  "attempts": [{"status": "download_failed", "http_status": 404,
                                "error_type": "HTTPError", "attempted_at_utc": "2026-01-01T00:00:00Z"}]}
        manifest.write_bytes(json_bytes({"sources": [source]}))
        evidence, protected = _probes(manifest, "2011-07-27")
        self.assertEqual(evidence["probe_count"], 1)
        self.assertIn("does not prove", evidence["qualification"])
        self.assertEqual(protected, [manifest])
        with self.assertRaisesRegex(ValueError, "bounded failed"):
            _probes(manifest, "2011-07-20")
        source["attempts"] = []
        manifest.write_bytes(json_bytes({"sources": [source]}))
        with self.assertRaisesRegex(ValueError, "attempts"):
            _probes(manifest, "2011-07-27")

    def test_quarantine_verifies_original_failed_response_without_admitting_values(self):
        issue = self.manifest["issues"][0]
        stale = csv_payload(date.fromisoformat(issue["observation_date"]) - timedelta(days=7))
        source = self.capture("stale.csv", stale, issue["table_url"])
        # The capture helper wrote into the snapshot; quarantine shares that directory.
        manifest = self.snapshot / "quarantine.json"
        manifest.write_bytes(json_bytes({"complete": True, "sources": [source]}))
        failures = [{"issue_id": issue["issue_id"], "status": "download_failed",
                     "captured_response_metadata": {key: source[key] for key in ("bytes", "sha256")}}]
        identity, protected = _quarantine(manifest, [issue], failures)
        item = identity["verified_stale_responses"][0]
        self.assertTrue(item["excluded_from_inventory_records"])
        self.assertNotEqual(item["actual_current_date_header"], item["expected_observation_date"])
        self.assertEqual(len(protected), 2)
        failures[0]["captured_response_metadata"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "original failed capture"):
            _quarantine(manifest, [issue], failures)
        with patch("oil_audit.inventory._verified", side_effect=AssertionError("read source")):
            with self.assertRaisesRegex(ValueError, "failed canonical"):
                _quarantine(manifest, [issue], [])


class ReplayAvailabilityTests(unittest.TestCase):
    def test_definition_break_is_distinct_from_other_value_changes(self):
        before = record("before", "2016-10-05", "2016-09-30", 400, 399, "includes_lease_stocks")
        after = record("after", "2016-10-13", "2016-10-07", 373, 370)
        later = record("later", "2016-10-19", "2016-10-14", 375, 374)
        changes, counts = compare_overlaps([before, after, later])
        crude = [row for row in changes if row["series"] == "commercial_crude"]
        self.assertEqual([row["classification"] for row in crude],
                         ["definition_break_change", "other_archive_value_change"])
        self.assertEqual(counts["commercial_crude"]["definition_break_change"], 1)
        self.assertEqual(counts["cushing"]["definition_break_change"], 0)
        self.assertEqual(counts["cushing"]["other_archive_value_change"], 2)

    def test_release_date_not_issue_name_controls_delayed_availability(self):
        old = record("old", "2022-06-15", "2022-06-10")
        delayed = record("2022_06_17_data", "2022-06-29", "2022-06-17")
        newer = record("2022_06_29", "2022-06-29", "2022-06-24")
        records = [old, delayed, newer]
        dates = [forecast("2022-06-23", "2022-06-30"), forecast("2022-06-29", "2022-07-07"),
                 forecast("2022-06-30", "2022-07-08")]
        coverage = development_coverage(records, records, dates)
        self.assertEqual([row["latest_declared_issue_id"] for row in coverage], ["old", "old", "2022_06_29"])
        self.assertEqual(coverage[-1]["observation_age_calendar_days"], 6)

    def test_four_week_availability_needs_exact_prior_observation_and_same_regime(self):
        old = record("old", "2017-01-05", "2016-12-30", commercial_regime="includes_lease_stocks")
        latest = record("latest", "2017-02-01", "2017-01-27")
        coverage = development_coverage([old, latest], [old, latest], [forecast("2017-02-02", "2017-02-09")])[0]
        self.assertTrue(coverage["series"]["cushing"]["four_week_change_available"])
        self.assertFalse(coverage["series"]["commercial_crude"]["four_week_change_available"])
        self.assertEqual(coverage["series"]["commercial_crude"]["four_week_missing_reason"], "definition_regime_mismatch")
        old["observation_date"] = "2016-12-23"
        old["previous_observation_date"] = "2016-12-16"
        missing = development_coverage([old, latest], [old, latest], [forecast("2017-02-02", "2017-02-09")])[0]
        self.assertFalse(missing["series"]["cushing"]["four_week_change_available"])

    def test_latest_failed_release_is_not_silently_replaced_with_stale_values(self):
        old = record("old", "2017-01-05", "2016-12-30")
        failed = record("failed", "2017-01-11", "2017-01-06")
        row = development_coverage([old, failed], [old], [forecast("2017-01-12", "2017-01-20")])[0]
        self.assertEqual(row["status"], "latest_declared_issue_unusable")
        self.assertFalse(row["series"]["cushing"]["stock_available"])

    def test_future_inventory_does_not_change_earlier_coverage_and_holdout_is_rejected(self):
        old = record("old", "2017-01-05", "2016-12-30")
        later = record("later", "2017-02-01", "2017-01-27")
        rows = [forecast("2017-01-06", "2017-01-13")]
        self.assertEqual(development_coverage([old], [old], rows),
                         development_coverage([old, later], [old, later], rows))
        with self.assertRaisesRegex(ValueError, "development-only"):
            development_coverage([old], [old], [forecast("2023-01-05", "2023-01-12")])


WTI_REPORT = Path("reports/development_evaluation.json")


if __name__ == "__main__":
    unittest.main()

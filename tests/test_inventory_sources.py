"""Synthetic release-index tests contain no downloaded inventory values."""

from datetime import date, timedelta
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
from urllib.error import URLError

from oil_audit.dataset import json_bytes, sha256
from oil_audit import inventory_sources as sources
from oil_audit.inventory_sources import ARCHIVE_INDEX, parse_index


def issue_row(released, ending, *, slug=None, host="", month=None, release_text=None):
    slug = slug or released.strftime("%Y_%m_%d")
    path = f"/petroleum/supply/weekly/archive/{released.year}/{slug}/wpsr_{slug}.php"
    href = host + path
    month = released.strftime("%B") if month is None else month
    release_text = str(released.day) if release_text is None else release_text
    return (f'<tr><th>{month}</th><td><a href="{href}">{release_text}</a></td>'
            f'<td>{ending}</td></tr>')


def minimal_rows():
    rows = []
    for year in range(2011, 2023):
        first = date(year, 1, 1)
        observed = first + timedelta(days=(4 - first.weekday()) % 7)
        released = observed + timedelta(days=5)
        rows.append(issue_row(released, str(observed.day)))
    return rows


def document(rows):
    return ("<html><table>" + "".join(rows) + "</table></html>").encode("utf-8")


class InventoryIndexTests(unittest.TestCase):
    def test_minimal_index_covers_all_twelve_requested_years(self):
        parsed = parse_index(document(minimal_rows()))
        self.assertEqual(len(parsed), 12)
        self.assertEqual([row["release_date"][:4] for row in parsed],
                         [str(year) for year in range(2011, 2023)])
        for row in parsed:
            self.assertLess(row["observation_date"], row["release_date"])
            self.assertTrue(row["issue_url"].startswith(ARCHIVE_INDEX))
            self.assertTrue(row["table_url"].endswith("/csv/table4.csv"))

    def test_same_day_delayed_issues_use_displayed_release_not_folder_date(self):
        rows = minimal_rows() + [
            issue_row(date(2022, 6, 29), "24"),
            issue_row(date(2022, 6, 29), "17", slug="2022_06_17_data", month=""),
        ]
        parsed = [row for row in parse_index(document(rows)) if row["release_date"] == "2022-06-29"]
        self.assertEqual([row["observation_date"] for row in parsed], ["2022-06-17", "2022-06-24"])
        self.assertEqual(parsed[0]["issue_id"], "2022_06_17_data")
        self.assertEqual(parsed[0]["table_url"], ARCHIVE_INDEX + "2022/2022_06_17_data/csv/table4.csv")

    def test_month_and_year_crossing_observation_dates(self):
        rows = minimal_rows() + [
            issue_row(date(2022, 2, 2), "1/28"),
            issue_row(date(2022, 1, 5), "12/31"),
        ]
        by_release = {row["release_date"]: row for row in parse_index(document(rows))}
        self.assertEqual(by_release["2022-02-02"]["observation_date"], "2022-01-28")
        self.assertEqual(by_release["2022-01-05"]["observation_date"], "2021-12-31")

    def test_future_links_are_ignored_before_date_or_host_parsing(self):
        future = issue_row(date(2023, 1, 5), "not a date", release_text="999",
                           host="https://irrelevant.example")
        baseline = parse_index(document(minimal_rows()))
        self.assertEqual(parse_index(document(minimal_rows() + [future])), baseline)

    def test_duplicate_issue_link_is_rejected(self):
        rows = minimal_rows()
        with self.assertRaisesRegex(ValueError, "Duplicate linked archive issue"):
            parse_index(document(rows + [rows[-1]]))

    def test_different_issues_for_same_observation_week_are_rejected(self):
        rows = minimal_rows() + [
            issue_row(date(2022, 3, 2), "2/25"),
            issue_row(date(2022, 3, 3), "2/25"),
        ]
        with self.assertRaisesRegex(ValueError, "Duplicate indexed observation weeks"):
            parse_index(document(rows))

    def test_inconsistent_or_invalid_dates_are_rejected(self):
        for released, ending, release_text in (
            (date(2022, 2, 2), "2", None),
            (date(2022, 2, 2), "4", None),
            (date(2022, 2, 2), "1/27", None),
            (date(2022, 2, 2), "2/30", None),
            (date(2022, 2, 2), "1/28", "31"),
            (date(2022, 2, 2), "unknown", None),
            (date(2022, 2, 2), "1/28", "unknown"),
        ):
            with self.subTest(ending=ending, release_text=release_text), self.assertRaises(ValueError):
                parse_index(document(minimal_rows() + [issue_row(
                    released, ending, release_text=release_text)]))

    def test_missing_requested_year_or_empty_index_is_rejected(self):
        for rows in (minimal_rows()[:-1], []):
            with self.subTest(row_count=len(rows)), self.assertRaisesRegex(ValueError, "every requested year"):
                parse_index(document(rows))

    def test_unexpected_host_for_in_scope_issue_is_rejected(self):
        rows = minimal_rows() + [issue_row(date(2022, 2, 2), "1/28", host="https://other.example")]
        with self.assertRaisesRegex(ValueError, "Unexpected archive-index host"):
            parse_index(document(rows))

    def test_official_absolute_url_and_html_extension_are_supported(self):
        rows = minimal_rows()
        rows[0] = rows[0].replace('href="/', 'href="https://www.eia.gov/').replace(".php", ".html")
        parsed = parse_index(document(rows))
        self.assertTrue(parsed[0]["issue_url"].startswith(ARCHIVE_INDEX))
        self.assertTrue(parsed[0]["issue_url"].endswith(".html"))

    def test_missing_opening_row_tag_preserves_november_release_cells(self):
        rows = minimal_rows() + [
            issue_row(date(2016, 11, 23), "18"),
            issue_row(date(2016, 11, 30), "25", month="").replace("<tr>", ""),
        ]
        parsed = parse_index(document(rows))
        recovered = next(row for row in parsed if row["issue_id"] == "2016_11_30")
        self.assertEqual(recovered["release_date"], "2016-11-30")
        self.assertEqual(recovered["observation_date"], "2016-11-25")

    def test_issue_link_outside_table_rows_cannot_be_silently_omitted(self):
        link = ('<p><a href="/petroleum/supply/weekly/archive/2022/2022_02_02/'
                'wpsr_2022_02_02.php">February 2</a></p>')
        with self.assertRaisesRegex(ValueError, "omitted linked inventory issues"):
            parse_index(document(minimal_rows() + [link]))

    def test_unrecoverable_unclosed_issue_row_cannot_be_silently_omitted(self):
        unclosed = issue_row(date(2022, 2, 2), "1/28").replace("</tr>", "")
        with self.assertRaisesRegex(ValueError, "omitted linked inventory issues"):
            parse_index(document(minimal_rows() + [unclosed]))


class FakeResponse(io.BytesIO):
    def __init__(self, url, payload):
        super().__init__(payload)
        self.url = url
        self.headers = {"Content-Type": "text/plain", "Last-Modified": "test-source-time"}


def capture_record(url, filename, payload):
    return {"requested_url": url, "resolved_url": url, "file": filename,
            "status": "downloaded", "bytes": len(payload), "sha256": sha256(payload),
            "retrieved_at_utc": "2026-10-13T00:00:00+00:00"}


class InventoryFetchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.index = document(minimal_rows())
        self.indexed = parse_index(self.index)
        self.issues = sources._canonical_issues(self.indexed)
        self.responses = {ARCHIVE_INDEX: self.index}
        for issue in self.issues:
            observed = date.fromisoformat(issue["observation_date"])
            self.responses[issue["table_url"]] = (
                '"STUB_1","' + observed.strftime("%m/%d/%y") + '","prior"\n'
            ).encode()
        for issue in sources.SUPPLEMENTARY_ISSUES:
            released = date.fromisoformat(issue["release_date"]).strftime("%B %d, %Y")
            observed = date.fromisoformat(issue["observation_date"]).strftime("%B %d, %Y")
            self.responses[issue["issue_url"]] = (
                f"<p>Data for week ending {observed} | Release Date: {released}</p>"
            ).encode()
        self.calls = []

    def client(self, request, *, timeout):
        self.assertEqual(timeout, 20)
        self.calls.append(request.full_url)
        return FakeResponse(request.full_url, self.responses[request.full_url])

    def seed(self):
        index_record = capture_record(ARCHIVE_INDEX, "archive-index.html", self.index)
        (self.directory / "archive-index.html").write_bytes(self.index)
        manifest = {"schema_version": 1, "started_at_utc": "2026-10-13T00:00:00+00:00",
                    "release_years": [2011, 2022], "complete": False,
                    "index": index_record, "issues": self.indexed, "sources": []}
        self.save(manifest)
        return manifest

    def save(self, manifest):
        (self.directory / "manifest.json").write_bytes(json_bytes(manifest))

    def run_fetch(self, resume=None):
        with patch.object(sources, "urlopen", side_effect=self.client), \
                patch.object(sources, "RequestThrottle") as throttle, \
                patch("builtins.print"):
            path = sources.fetch(resume=resume or self.directory, max_workers=3)
        self.assertEqual(throttle.return_value.wait.call_count, len(self.calls))
        return json.loads(path.read_bytes())

    def test_resume_skips_verified_bytes_retries_failures_and_recovers_orphans(self):
        manifest = self.seed()
        complete, failed, orphan, missing = self.indexed[:4]
        for issue in (complete, missing):
            record = {"issue_id": issue["issue_id"], **capture_record(
                issue["table_url"], issue["issue_id"] + "-table4.csv", self.responses[issue["table_url"]])}
            manifest["sources"].append(record)
            if issue == complete:
                (self.directory / record["file"]).write_bytes(self.responses[issue["table_url"]])
        prior_failure = {"issue_id": failed["issue_id"], "requested_url": failed["table_url"],
                         "file": failed["issue_id"] + "-table4.csv", "status": "download_failed",
                         "error_type": "HTTPError", "http_status": 503,
                         "attempted_at_utc": "2026-10-13T00:01:00+00:00"}
        manifest["sources"].append(prior_failure)
        (self.directory / (orphan["issue_id"] + "-table4.csv")).write_bytes(self.responses[orphan["table_url"]])
        self.save(manifest)
        with patch.object(sources, "_atomic_manifest", wraps=sources._atomic_manifest) as persist:
            result = self.run_fetch()
        records = {row["issue_id"]: row for row in result["sources"]}
        self.assertNotIn(ARCHIVE_INDEX, self.calls)
        self.assertNotIn(complete["table_url"], self.calls)
        self.assertEqual(records[complete["issue_id"]], manifest["sources"][0])
        self.assertEqual(records[failed["issue_id"]]["attempts"][0]["http_status"], 503)
        self.assertTrue(records[orphan["issue_id"]]["recovered_existing_file"])
        self.assertEqual(records[missing["issue_id"]]["sha256"], manifest["sources"][1]["sha256"])
        self.assertEqual(len(result["supplementary_issue_pages"]), 3)
        self.assertEqual(len(result["issues"]), 15)
        self.assertEqual(len(result["sources"]), 15)
        self.assertTrue(result["complete"])
        self.assertTrue(result["all_downloads_succeeded"])
        self.assertGreaterEqual(persist.call_count, len(self.calls) + 1)
        self.assertFalse((self.directory / "manifest.json.tmp").exists())

    def test_completed_resume_from_manifest_makes_no_requests(self):
        self.seed()
        first = self.run_fetch()
        self.calls.clear()
        with patch.object(sources, "urlopen", side_effect=AssertionError("Unexpected network call")):
            path = sources.fetch(resume=self.directory / "manifest.json", max_workers=3)
        self.assertEqual(json.loads(path.read_bytes()), first)

    def test_tampered_completed_capture_fails_before_any_network_or_overwrite(self):
        self.seed()
        manifest = self.run_fetch()
        source = self.directory / manifest["sources"][0]["file"]
        source.write_bytes(b"changed")
        with patch.object(sources, "urlopen", side_effect=AssertionError("Unexpected network call")):
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                sources.fetch(resume=self.directory)
        self.assertEqual(source.read_bytes(), b"changed")

    def test_conflicting_orphan_is_preserved_and_capture_finishes_with_failure(self):
        self.seed()
        issue = self.indexed[0]
        orphan = self.directory / (issue["issue_id"] + "-table4.csv")
        orphan.write_bytes(b"previous captured bytes")
        result = self.run_fetch()
        record = next(row for row in result["sources"] if row["issue_id"] == issue["issue_id"])
        self.assertEqual(orphan.read_bytes(), b"previous captured bytes")
        self.assertEqual(record["status"], "download_failed")
        self.assertIn("refusing to overwrite", record["error_reason"])
        self.assertTrue(result["complete"])
        self.assertFalse(result["all_downloads_succeeded"])

    def test_missing_recorded_file_cannot_be_restored_with_changed_source_bytes(self):
        manifest = self.seed()
        issue = self.indexed[0]
        record = {"issue_id": issue["issue_id"], **capture_record(
            issue["table_url"], issue["issue_id"] + "-table4.csv", b"original unavailable bytes")}
        manifest["sources"].append(record)
        self.save(manifest)
        result = self.run_fetch()
        failed = next(row for row in result["sources"] if row["issue_id"] == issue["issue_id"])
        self.assertEqual(failed["status"], "download_failed")
        self.assertIn("recorded checksum", failed["error_reason"])
        self.assertFalse((self.directory / record["file"]).exists())
        self.assertEqual(failed["attempts"][0]["sha256"], record["sha256"])
        # A failed recovery must not erase the original checksum constraint.
        self.calls.clear()
        repeated = self.run_fetch()
        still_failed = next(row for row in repeated["sources"] if row["issue_id"] == issue["issue_id"])
        self.assertEqual(still_failed["status"], "download_failed")
        self.assertEqual(still_failed["expected_capture"]["sha256"], record["sha256"])
        self.assertEqual(len(still_failed["attempts"]), 3)
        self.assertFalse((self.directory / record["file"]).exists())

    def test_wrong_supplementary_release_page_fails_before_table_requests(self):
        self.seed()
        issue = sources.SUPPLEMENTARY_ISSUES[0]
        self.responses[issue["issue_url"]] = b"<p>Released: July 28, 2011</p>"
        with self.assertRaisesRegex(ValueError, "supplementary inventory release"):
            self.run_fetch()
        self.assertEqual(self.calls, [issue["issue_url"]])
        manifest = json.loads((self.directory / "manifest.json").read_bytes())
        self.assertFalse(manifest["complete"])
        self.assertEqual(manifest["supplementary_issue_pages"][0]["status"], "download_failed")

    def test_download_retries_are_throttled_and_preserve_both_attempts(self):
        issue = self.indexed[0]
        throttle = Mock()
        with patch.object(sources, "urlopen", side_effect=[
                URLError("synthetic transport failure"),
                FakeResponse(issue["table_url"], self.responses[issue["table_url"]])]), \
                patch.object(sources.time, "sleep"):
            record = sources._download(issue["table_url"], self.directory / "table4.csv", throttle=throttle)
        self.assertEqual(throttle.wait.call_count, 2)
        self.assertEqual([row["status"] for row in record["attempts"]], ["download_failed", "downloaded"])

    def test_wrong_table_observation_is_not_accepted(self):
        self.seed()
        issue = self.indexed[0]
        self.responses[issue["table_url"]] = b'"STUB_1","01/01/22","prior"\n'
        result = self.run_fetch()
        record = next(row for row in result["sources"] if row["issue_id"] == issue["issue_id"])
        self.assertEqual(record["status"], "download_failed")
        self.assertIn("observation week", record["error_reason"])

    def test_out_of_scope_or_unofficial_resource_is_rejected_before_request(self):
        urls = (
            "https://www.eia.gov/petroleum/supply/weekly/archive/2023/2023_01_05/csv/table4.csv",
            "https://other.example/petroleum/supply/weekly/archive/2022/2022_01_05/csv/table4.csv",
            "https://www.eia.gov/petroleum/supply/weekly/csv/table4.csv",
        )
        with patch.object(sources, "urlopen") as client:
            for url in urls:
                with self.subTest(url=url), self.assertRaises(ValueError):
                    sources._download(url, self.directory / "forbidden.csv")
        client.assert_not_called()


class RequestThrottleTests(unittest.TestCase):
    def test_shared_slots_are_at_least_a_quarter_second_apart(self):
        now, waits, starts = [10.0], [], []

        def sleep(duration):
            waits.append(duration)
            now[0] += duration

        throttle = sources.RequestThrottle(clock=lambda: now[0], sleep=sleep)
        for _ in range(5):
            throttle.wait()
            starts.append(now[0])
        self.assertEqual(starts, [10.0, 10.25, 10.5, 10.75, 11.0])
        self.assertEqual(waits, [0.25] * 4)


if __name__ == "__main__":
    unittest.main()

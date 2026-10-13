"""Capture only pre-2023 EIA inventory releases with release-index evidence."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import date, datetime, timezone
from html import unescape
from html.parser import HTMLParser
import io
import json
import os
from pathlib import Path
import re
from threading import Lock
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlsplit
from urllib.request import Request, urlopen

from .dataset import json_bytes, sha256, workspace_path


ARCHIVE_INDEX = "https://www.eia.gov/petroleum/supply/weekly/archive/"
FIRST_YEAR, LAST_YEAR = 2011, 2022
MONTHS = {name: index for index, name in enumerate((
    "January", "February", "March", "April", "May", "June", "July", "August",
    "September", "October", "November", "December"), 1)}
ISSUE_PATH = re.compile(r"^/petroleum/supply/weekly/archive/(\d{4})/([^/]+)/wpsr_[^/]+\.(?:html|php)$")
TABLE_PATH = re.compile(r"^/petroleum/supply/weekly/archive/(\d{4})/[^/]+/csv/table4\.csv$")
DEFAULT_WORKERS = 12


def _supplement(released: str, observed: str, extension: str) -> dict:
    year, slug = released[:4], released.replace("-", "_")
    return {"issue_id": slug, "release_date": released, "observation_date": observed,
            "issue_url": ARCHIVE_INDEX + f"{year}/{slug}/wpsr_{slug}.{extension}",
            "table_url": ARCHIVE_INDEX + f"{year}/{slug}/csv/table4.csv"}


# July 2011 is absent from the index. The November issues occur in malformed
# index rows; retaining their issue pages independently confirms their dates.
SUPPLEMENTARY_ISSUES = (
    _supplement("2011-07-27", "2011-07-22", "html"),
    _supplement("2016-11-30", "2016-11-25", "php"),
    _supplement("2017-11-29", "2017-11-24", "php"),
)


class _IndexRows(HTMLParser):
    def __init__(self):
        super().__init__()
        self.rows, self.row, self.cell = [], None, None
        self.all_links = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            # Track links even when malformed markup prevents row recovery.
            self.all_links.append(dict(attrs).get("href", ""))
        if tag == "tr":
            self.row = []
        elif tag in ("td", "th"):
            # The official November 2016/2017 row omits its opening <tr>.
            if self.row is None:
                self.row = []
            self.cell = {"text": "", "links": []}
        elif tag == "a" and self.cell is not None:
            self.cell["links"].append(dict(attrs).get("href", ""))

    def handle_data(self, text):
        if self.cell is not None:
            self.cell["text"] += text

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self.cell is not None:
            self.cell["text"] = " ".join(self.cell["text"].split())
            self.row.append(self.cell)
            self.cell = None
        elif tag == "tr" and self.row is not None:
            self.rows.append(self.row)
            self.row = None


def parse_index(payload: bytes) -> list[dict]:
    """Read displayed release/data-ending dates, never a directory-name date.

    One release date can contain two issues (June 29, 2022). Only the linked
    issue year and its displayed calendar cells enter this bounded audit.
    """
    parser = _IndexRows()
    parser.feed(payload.decode("utf-8"))
    linked_paths = set()
    for href in parser.all_links:
        parts = urlsplit(href)
        match = ISSUE_PATH.fullmatch(parts.path)
        if not match or not FIRST_YEAR <= int(match[1]) <= LAST_YEAR:
            continue
        if parts.netloc not in ("", "www.eia.gov"):
            raise ValueError("Unexpected archive-index host")
        linked_paths.add(parts.path)
    month, issues, seen = None, [], set()
    for row in parser.rows:
        if row and row[0]["text"] in MONTHS:
            month = MONTHS[row[0]["text"]]
        for index, cell in enumerate(row):
            for href in cell["links"]:
                path = urlsplit(href).path
                match = ISSUE_PATH.fullmatch(path)
                if not match or not FIRST_YEAR <= int(match[1]) <= LAST_YEAR:
                    continue
                if urlsplit(href).netloc not in ("", "www.eia.gov"):
                    raise ValueError("Unexpected archive-index host")
                year, slug = int(match[1]), match[2]
                if month is None or not cell["text"].isdigit() or index + 1 >= len(row):
                    raise ValueError("Missing displayed archive release date")
                released = date(year, month, int(cell["text"]))
                ending_text = row[index + 1]["text"]
                if re.fullmatch(r"\d{1,2}/\d{1,2}", ending_text):
                    ending_month, ending_day = map(int, ending_text.split("/"))
                    ending_year = year - (ending_month > month)
                elif ending_text.isdigit():
                    ending_month, ending_day, ending_year = month, int(ending_text), year
                else:
                    raise ValueError("Missing displayed data-ending date")
                ending = date(ending_year, ending_month, ending_day)
                if ending >= released or ending.weekday() != 4:
                    raise ValueError("Archive release/data-ending dates disagree")
                if path in seen:
                    raise ValueError("Duplicate linked archive issue")
                seen.add(path)
                issues.append({"issue_id": slug, "release_date": released.isoformat(),
                               "observation_date": ending.isoformat(),
                               "issue_url": urljoin(ARCHIVE_INDEX, path),
                               "table_url": urljoin(ARCHIVE_INDEX, f"{year}/{slug}/csv/table4.csv")})
    if seen != linked_paths:
        raise ValueError("Archive row reconstruction omitted linked inventory issues")
    if not issues or {date.fromisoformat(row["release_date"]).year for row in issues} != set(range(FIRST_YEAR, LAST_YEAR + 1)):
        raise ValueError("Archive index does not cover every requested year")
    issues.sort(key=lambda row: (row["release_date"], row["observation_date"]))
    observations = [row["observation_date"] for row in issues]
    if len(observations) != len(set(observations)):
        raise ValueError("Duplicate indexed observation weeks")
    return issues


class RequestThrottle:
    """Share one request-start slot every quarter second across all workers."""

    def __init__(self, *, clock=time.monotonic, sleep=time.sleep):
        self.clock, self.sleep = clock, sleep
        self.lock, self.next_start = Lock(), 0.0

    def wait(self):
        with self.lock:
            remaining = self.next_start - self.clock()
            if remaining > 0:
                self.sleep(remaining)
            self.next_start = self.clock() + 0.25


def _allowed_url(url: str) -> None:
    parts = urlsplit(url)
    if parts.scheme != "https" or parts.netloc != "www.eia.gov" or parts.query or parts.fragment:
        raise ValueError("Only official bounded EIA archive resources are allowed")
    if url == ARCHIVE_INDEX:
        return
    match = ISSUE_PATH.fullmatch(parts.path) or TABLE_PATH.fullmatch(parts.path)
    if match is None or not FIRST_YEAR <= int(match[1]) <= LAST_YEAR:
        raise ValueError("Resource is outside the pre-2023 inventory archive")


def validate_issue_page(payload: bytes, issue: dict) -> None:
    """Check displayed publication dates; the folder name is not evidence."""
    text = " ".join(unescape(re.sub(r"<[^>]+>", " ", payload.decode("cp1252"))).split())
    matches = re.findall(r"(?:Released|Release Date)\s*:\s*([A-Za-z]+) (\d{1,2}), (\d{4})", text)
    try:
        released = [date(int(year), MONTHS[month], int(day)).isoformat() for month, day, year in matches]
    except (KeyError, ValueError) as exc:
        raise ValueError("Supplementary issue page contains an invalid release date") from exc
    if released != [issue["release_date"]]:
        raise ValueError("Supplementary issue page does not confirm its release date")
    endings = re.findall(r"Data for week ending ([A-Za-z]+) (\d{1,2}), (\d{4})", text)
    try:
        observed = [date(int(year), MONTHS[month], int(day)).isoformat() for month, day, year in endings]
    except (KeyError, ValueError) as exc:
        raise ValueError("Supplementary issue page contains an invalid observation date") from exc
    if observed and observed != [issue["observation_date"]]:
        raise ValueError("Supplementary issue page disagrees with its observation week")
    if issue["release_date"] != "2011-07-27" and not observed:
        raise ValueError("Supplementary issue page omits its observation week")


def validate_table_week(payload: bytes, issue: dict) -> None:
    try:
        header = next(csv.reader(io.StringIO(payload.decode("cp1252"))))
        observed = datetime.strptime(header[1], "%m/%d/%y").date().isoformat()
    except (StopIteration, IndexError, ValueError) as exc:
        raise ValueError("Inventory table has no valid observation header") from exc
    if header[0] != "STUB_1" or observed != issue["observation_date"]:
        raise ValueError("Inventory table observation week differs from its release evidence")


def _download(url: str, destination: Path, *, throttle=None, expected=None, validator=None) -> dict:
    """Capture bytes immutably, including each failed transport attempt."""
    _allowed_url(url)
    throttle = throttle or RequestThrottle()
    attempts = []
    identity = {"requested_url": url, "file": destination.name}
    if expected is not None:
        identity["expected_capture"] = {key: expected[key] for key in ("sha256", "bytes")}
    for attempt in range(2):
        throttle.wait()
        details = {"attempted_at_utc": datetime.now(timezone.utc).isoformat()}
        try:
            request = Request(url, headers={"User-Agent": "oil-inventory-audit/0.1"})
            with urlopen(request, timeout=20) as response:
                if response.url.rstrip("/") != url.rstrip("/"):
                    raise ValueError("Unexpected source redirect")
                payload = response.read(2_000_001)
                if not payload or len(payload) > 2_000_000:
                    raise ValueError("Empty or oversized source")
                details.update({"resolved_url": response.url,
                                "content_type": response.headers.get("Content-Type", ""),
                                "http_last_modified": response.headers.get("Last-Modified"),
                                "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
                                "bytes": len(payload), "sha256": sha256(payload)})
            if expected is not None and (details["sha256"] != expected["sha256"]
                                         or details["bytes"] != expected["bytes"]):
                raise ValueError("Recaptured source differs from its recorded checksum")
            if destination.exists() and destination.read_bytes() != payload:
                raise ValueError("Captured source bytes differ; refusing to overwrite")
            if validator is not None:
                validator(payload)
            if destination.exists():
                details["recovered_existing_file"] = True
            else:
                # Exclusive creation also protects against competing fetchers.
                with destination.open("xb") as stream:
                    stream.write(payload)
            details["status"] = "downloaded"
            attempts.append(details)
            return {**identity, **details, "attempts": attempts}
        except (HTTPError, URLError, TimeoutError, ValueError) as exc:
            details.update({"status": "download_failed", "error_type": type(exc).__name__,
                            "http_status": getattr(exc, "code", None)})
            if isinstance(exc, ValueError):
                details["error_reason"] = str(exc)
            attempts.append(details)
            if attempt == 0 and not isinstance(exc, ValueError) and getattr(exc, "code", None) != 404:
                time.sleep(1)
                continue
            return {**identity, **details, "attempts": attempts}
    raise AssertionError("Unreachable download state")


def _atomic_manifest(path: Path, manifest: dict) -> None:
    temporary = path.with_suffix(".json.tmp")
    with temporary.open("wb") as stream:
        stream.write(json_bytes(manifest))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _verified(record, destination: Path, url: str) -> bool:
    if record is None:
        return False
    if record.get("requested_url") != url or record.get("file") != destination.name:
        raise ValueError("Captured source identity differs from its canonical resource")
    if record.get("status") != "downloaded":
        return False
    if not destination.exists():
        return False
    payload = destination.read_bytes()
    if len(payload) != record.get("bytes") or sha256(payload) != record.get("sha256"):
        raise ValueError("Captured source checksum mismatch: " + destination.name)
    return True


def _merge_attempts(previous, current):
    if previous is None:
        return current
    history = previous.get("attempts")
    if history is None:
        history = [{key: value for key, value in previous.items()
                    if key not in ("issue_id", "requested_url", "file")}]
    return {**current, "attempts": history + current["attempts"]}


def _capture_record(previous, url, destination, throttle, validator=None):
    expected = (previous if previous and previous.get("status") == "downloaded" else
                previous.get("expected_capture") if previous else None)
    return _merge_attempts(previous, _download(
        url, destination, throttle=throttle, expected=expected, validator=validator))


def _canonical_issues(indexed):
    by_id = {issue["issue_id"]: issue for issue in indexed}
    for issue in SUPPLEMENTARY_ISSUES:
        if issue["issue_id"] in by_id and by_id[issue["issue_id"]] != issue:
            raise ValueError("Supplementary issue differs from the official index")
        by_id[issue["issue_id"]] = issue
    issues = sorted(by_id.values(), key=lambda row: (row["release_date"], row["observation_date"]))
    if len({row["observation_date"] for row in issues}) != len(issues):
        raise ValueError("Conflicting inventory observation weeks")
    return issues


def _records_by_id(records, allowed):
    result = {}
    for record in records:
        identity = record.get("issue_id")
        if identity not in allowed or identity in result:
            raise ValueError("Unexpected or duplicate captured inventory issue")
        result[identity] = record
    return result


def fetch(output_root: Path = Path("data/audit/inventories"), *, resume: Path | None = None,
          max_workers: int = DEFAULT_WORKERS) -> Path:
    """Capture or resume bounded releases, reusing only verified saved bytes."""
    if type(max_workers) is not int or not 1 <= max_workers <= DEFAULT_WORKERS:
        raise ValueError("Inventory worker count must be between 1 and 12")
    throttle = RequestThrottle()
    if resume is not None:
        manifest_path = resume / "manifest.json" if resume.is_dir() else resume
        destination = manifest_path.parent
        manifest = json.loads(manifest_path.read_bytes())
        if manifest.get("schema_version") != 1 or manifest.get("release_years") != [FIRST_YEAR, LAST_YEAR]:
            raise ValueError("Incompatible inventory capture manifest")
    else:
        started = datetime.now(timezone.utc)
        destination = output_root / started.strftime("%Y%m%dT%H%M%S%fZ")
        destination.mkdir(parents=True, exist_ok=False)
        manifest_path = destination / "manifest.json"
        manifest = {"schema_version": 1, "started_at_utc": started.isoformat(),
                    "release_years": [FIRST_YEAR, LAST_YEAR], "complete": False,
                    "index": None, "issues": [], "sources": []}
    index_path = destination / "archive-index.html"
    if not _verified(manifest.get("index"), index_path, ARCHIVE_INDEX):
        manifest["index"] = _capture_record(manifest.get("index"), ARCHIVE_INDEX, index_path, throttle)
        _atomic_manifest(manifest_path, manifest)
        if manifest["index"]["status"] != "downloaded":
            raise ValueError("Could not capture the official archive index")
    indexed = parse_index(index_path.read_bytes())
    issues = _canonical_issues(indexed)
    by_id = {issue["issue_id"]: issue for issue in issues}
    previous_issues = manifest.get("issues", [])
    if len({row["issue_id"] for row in previous_issues}) != len(previous_issues) or any(
            issue != by_id.get(issue["issue_id"]) for issue in previous_issues):
        raise ValueError("Saved inventory issues differ from the captured release evidence")
    records = _records_by_id(manifest.get("sources", []), by_id)
    pages = _records_by_id(manifest.get("supplementary_issue_pages", []),
                           {issue["issue_id"] for issue in SUPPLEMENTARY_ISSUES})
    # Validate every completed capture before starting any new requests.
    finished = {issue_id for issue_id, record in records.items() if _verified(
        record, destination / (issue_id + "-table4.csv"), by_id[issue_id]["table_url"])}
    verified_pages = {issue["issue_id"] for issue in SUPPLEMENTARY_ISSUES if _verified(
        pages.get(issue["issue_id"]), destination / (issue["issue_id"] + "-issue.html"), issue["issue_url"])}
    manifest.update({"issues": issues, "complete": False, "all_downloads_succeeded": False,
                     "indexed_issue_count": len(indexed),
                     "additional_unindexed_issue_ids": [issue["issue_id"] for issue in issues
                                                        if issue["issue_id"] not in {row["issue_id"] for row in indexed}],
                     "capture_settings": {"max_workers": max_workers, "minimum_request_interval_seconds": 0.25,
                                          "request_timeout_seconds": 20, "maximum_attempts_per_fetch": 2}})

    def persist():
        manifest["sources"] = sorted(records.values(), key=lambda row: row["issue_id"])
        manifest["supplementary_issue_pages"] = sorted(pages.values(), key=lambda row: row["issue_id"])
        _atomic_manifest(manifest_path, manifest)

    persist()
    for issue in SUPPLEMENTARY_ISSUES:
        issue_id = issue["issue_id"]
        path = destination / (issue_id + "-issue.html")
        if issue_id not in verified_pages:
            record = _capture_record(pages.get(issue_id), issue["issue_url"], path, throttle,
                                     validator=lambda payload, issue=issue: validate_issue_page(payload, issue))
            pages[issue_id] = {"issue_id": issue_id, **record}
            persist()
            if record["status"] != "downloaded":
                raise ValueError("Could not verify supplementary inventory release evidence")
        validate_issue_page(path.read_bytes(), issue)
        if issue_id in finished:
            validate_table_week((destination / (issue_id + "-table4.csv")).read_bytes(), issue)
    pending = [issue for issue in issues if issue["issue_id"] not in finished]
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(
            _capture_record, records.get(issue["issue_id"]), issue["table_url"],
            destination / (issue["issue_id"] + "-table4.csv"), throttle,
            lambda payload, issue=issue: validate_table_week(payload, issue)): issue for issue in pending}
        for future in as_completed(futures):
            issue = futures[future]
            records[issue["issue_id"]] = {"issue_id": issue["issue_id"], **future.result()}
            persist()
            if len(records) % 25 == 0:
                print(f"Attempted {len(records)}/{len(issues)} inventory tables", flush=True)
    manifest["complete"] = len(records) == len(issues)
    manifest["all_downloads_succeeded"] = all(row["status"] == "downloaded" for row in records.values())
    persist()
    return manifest_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=workspace_path, default=Path("data/audit/inventories"))
    parser.add_argument("--resume", type=workspace_path,
                        help="Resume an existing snapshot directory or its manifest.json")
    parser.add_argument("--max-workers", type=int, default=DEFAULT_WORKERS)
    args = parser.parse_args()
    print(fetch(args.output_root, resume=args.resume, max_workers=args.max_workers).as_posix())


if __name__ == "__main__":
    main()

"""Offline, qualified replay audit of bounded EIA inventory archive releases.

No inventory features or price models are fitted here. Release-dated archived
files support an as-of convention; they do not prove unchanged first releases
or historical intraday publication times.
"""

import argparse
from collections import Counter, defaultdict
import csv
from datetime import date, datetime, timedelta
import io
import json
from pathlib import Path
import re
import statistics
from urllib.parse import urlsplit

from .calendar import CALENDAR_ID, next_business_day
from .dataset import json_bytes, jsonl_bytes, sha256, workspace_path
from .inventory_sources import (ARCHIVE_INDEX, FIRST_YEAR, LAST_YEAR,
                                SUPPLEMENTARY_ISSUES, _canonical_issues,
                                parse_index, validate_issue_page)
from .inventory_tables import read_table4, _number
from oil_eval.benchmarks import _load_saved_run


DEFINITION_CHANGE = date(2016, 10, 13)
SERIES = ("commercial_crude", "cushing")
WTI_REPORT = Path("reports/development_evaluation.json")
SOURCE_FIELDS = ("issue_id", "release_date", "observation_date", "issue_url", "table_url")
DEFINITION_AND_RELEASE_NOTICES = {
    "https://www.eia.gov/todayinenergy/detail.php?id=28292",
    "https://www.eia.gov/pressroom/releases/press506.php",
    "https://www.eia.gov/pressroom/releases/press507.php",
}


def audit_config():
    return {
        "schema_version": 1, "release_years": [FIRST_YEAR, LAST_YEAR],
        "calendar_id": CALENDAR_ID,
        "availability": "release date strictly before forecast date; earliest issuance is next US business day",
        "commercial_definition_change_release_date": DEFINITION_CHANGE.isoformat(),
        "definition_before": "includes_lease_stocks", "definition_after": "excludes_lease_stocks",
        "coverage_selection": "latest observation in a release available before each saved development forecast",
        "four_week_availability": "exact observation minus 28 calendar days, already released, same definition regime",
        "status": "qualified archive replay; unchanged first releases and intraday availability not proved",
        "scope": "inventory availability audit only; no feature construction, model fitting or holdout evaluation",
    }


def _day(value):
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        raise ValueError("Expected an ISO inventory date")
    return date.fromisoformat(value)


def _basename(root, name):
    if not isinstance(name, str) or not name or Path(name).name != name or name in (".", ".."):
        raise ValueError("Inventory source files must be relative basenames")
    path = root / name
    if path.resolve().parent != root.resolve():
        raise ValueError("Inventory source path escapes its snapshot")
    return path


def _verified(root, record, expected_url, expected_file=None):
    if record.get("requested_url") != expected_url or (
            expected_file is not None and record.get("file") != expected_file):
        raise ValueError("Inventory source identity differs from release evidence")
    path = _basename(root, record.get("file"))
    if record.get("status", "downloaded") != "downloaded":
        if record.get("status") != "download_failed":
            raise ValueError("Unknown inventory download status")
        return path, None
    if record.get("resolved_url", expected_url) != expected_url:
        raise ValueError("Inventory source resolved URL differs from its request")
    if (type(record.get("bytes")) is not int or record["bytes"] <= 0
            or not re.fullmatch(r"[0-9a-f]{64}", record.get("sha256", ""))):
        raise ValueError("Incomplete inventory source checksum metadata")
    if "expected_capture" in record and record["expected_capture"] != {
            key: record[key] for key in ("sha256", "bytes")}:
        raise ValueError("Recovered inventory source differs from its original capture identity")
    try:
        payload = path.read_bytes()
    except FileNotFoundError as exc:
        raise ValueError("Missing declared inventory source: " + path.name) from exc
    if len(payload) != record["bytes"] or sha256(payload) != record["sha256"]:
        raise ValueError("Inventory source checksum mismatch: " + path.name)
    return path, payload


def _by_id(records, expected):
    if not isinstance(records, list):
        raise ValueError("Missing inventory source list")
    result = {}
    for record in records:
        identity = record.get("issue_id")
        if identity not in expected or identity in result:
            raise ValueError("Unexpected or duplicate inventory source identity")
        result[identity] = record
    if set(result) != set(expected):
        raise ValueError("Missing declared inventory source identity")
    return result


def _regime(series, released):
    if series == "cushing":
        return "cushing_reported"
    return "includes_lease_stocks" if released < DEFINITION_CHANGE else "excludes_lease_stocks"


def load_snapshot(snapshot_dir: Path):
    """Authenticate all declared bytes before parsing any inventory values."""
    manifest_path = snapshot_dir / "manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if (manifest.get("schema_version") != 1 or manifest.get("complete") is not True
            or manifest.get("release_years") != [FIRST_YEAR, LAST_YEAR]):
        raise ValueError("Inventory snapshot must be complete and bounded to 2011–2022")
    protected = [manifest_path]
    index_path, index_payload = _verified(snapshot_dir, manifest["index"], ARCHIVE_INDEX, "archive-index.html")
    protected.append(index_path)
    if index_payload is None:
        raise ValueError("Inventory archive index was not captured")
    indexed = parse_index(index_payload)
    issues = _canonical_issues(indexed)
    if manifest.get("issues") != issues:
        raise ValueError("Inventory issues differ from authenticated index and supplemental evidence")
    if manifest.get("indexed_issue_count") != len(indexed):
        raise ValueError("Inventory indexed issue count differs from saved index")
    indexed_ids = {row["issue_id"] for row in indexed}
    extra_ids = [row["issue_id"] for row in issues if row["issue_id"] not in indexed_ids]
    if manifest.get("additional_unindexed_issue_ids") != extra_ids:
        raise ValueError("Inventory supplementary issue identities differ")
    for issue in issues:
        released, observed = _day(issue["release_date"]), _day(issue["observation_date"])
        if not FIRST_YEAR <= released.year <= LAST_YEAR or observed >= released or observed.weekday() != 4:
            raise ValueError("Inventory issue is future-dated or outside its release scope")
    sources = _by_id(manifest.get("sources"), {row["issue_id"] for row in issues})
    pages = _by_id(manifest.get("supplementary_issue_pages"),
                   {row["issue_id"] for row in SUPPLEMENTARY_ISSUES})
    verified = {index_path.name: {key: manifest["index"][key] for key in ("bytes", "sha256")}}
    for issue in SUPPLEMENTARY_ISSUES:
        record = pages[issue["issue_id"]]
        path, payload = _verified(snapshot_dir, record, issue["issue_url"], issue["issue_id"] + "-issue.html")
        protected.append(path)
        if payload is None:
            raise ValueError("Missing supplementary inventory release evidence")
        validate_issue_page(payload, issue)
        verified[path.name] = {key: record[key] for key in ("bytes", "sha256")}
    payloads, failures = {}, []
    for issue in issues:
        record = sources[issue["issue_id"]]
        path, payload = _verified(snapshot_dir, record, issue["table_url"], issue["issue_id"] + "-table4.csv")
        protected.append(path)
        if payload is None:
            failures.append({"issue_id": issue["issue_id"], "release_date": issue["release_date"],
                             "observation_date": issue["observation_date"], "status": "download_failed",
                             "http_status": record.get("http_status"), "error_type": record.get("error_type"),
                             "reason": record.get("error_reason"),
                             "captured_response_metadata": {key: record[key] for key in
                                 ("sha256", "bytes", "http_last_modified", "retrieved_at_utc") if key in record},
                             "response_bytes_saved_in_primary_snapshot": False})
        else:
            verified[path.name] = {key: record[key] for key in ("bytes", "sha256")}
            payloads[issue["issue_id"]] = payload
    if manifest.get("all_downloads_succeeded") is not (len(failures) == 0):
        raise ValueError("Inventory download success flag disagrees with source records")
    records = []
    for issue in issues:
        payload = payloads.get(issue["issue_id"])
        if payload is None:
            continue
        released, observed = _day(issue["release_date"]), _day(issue["observation_date"])
        try:
            parsed = read_table4(payload, released, observed)
        except ValueError as exc:
            failures.append({"issue_id": issue["issue_id"], "release_date": issue["release_date"],
                             "observation_date": issue["observation_date"], "status": "schema_failed",
                             "reason": str(exc)})
            continue
        records.append({
            **{key: issue[key] for key in SOURCE_FIELDS},
            "previous_observation_date": parsed["previous_date"].isoformat(),
            "earliest_eligible_issuance_date": next_business_day(released).isoformat(),
            "source_file": sources[issue["issue_id"]]["file"],
            "source_sha256": sources[issue["issue_id"]]["sha256"],
            "series": {name: {**values, "definition_regime": _regime(name, released)}
                       for name, values in parsed["series"].items()},
        })
    identity = {"manifest_sha256": sha256(manifest_bytes), "verified_artifacts": verified,
                "indexed_issue_count": len(indexed), "additional_unindexed_issue_ids": extra_ids}
    return identity, issues, records, failures, protected


def compare_overlaps(records):
    """Compare repeated current/prior observations in successive saved issues."""
    histories = defaultdict(list)
    for record in records:
        for name in SERIES:
            series = record["series"][name]
            for role, key, observed in (
                ("current", "current_million_barrels", record["observation_date"]),
                ("previous", "previous_million_barrels", record["previous_observation_date"]),
            ):
                histories[(name, observed)].append({
                    "issue_id": record["issue_id"], "release_date": record["release_date"],
                    "issue_observation_date": record["observation_date"], "column": role,
                    "million_barrels": series[key], "definition_regime": series["definition_regime"],
                })
    changes, counts = [], {name: Counter() for name in SERIES}
    for (name, observed), occurrences in sorted(histories.items()):
        occurrences.sort(key=lambda row: (row["release_date"], row["issue_observation_date"]))
        counts[name]["observations"] += 1
        counts[name]["occurrences"] += len(occurrences)
        for before, after in zip(occurrences, occurrences[1:]):
            counts[name]["overlap_comparisons"] += 1
            regime_changed = before["definition_regime"] != after["definition_regime"]
            if regime_changed:
                counts[name]["definition_transitions"] += 1
            if before["million_barrels"] != after["million_barrels"]:
                classification = "definition_break_change" if regime_changed else "other_archive_value_change"
                counts[name][classification] += 1
                changes.append({"series": name, "observation_date": observed,
                                "classification": classification, "before": before, "after": after})
    fields = ("observations", "occurrences", "overlap_comparisons", "definition_transitions",
              "definition_break_change", "other_archive_value_change")
    return changes, {name: {field: counter[field] for field in fields} for name, counter in counts.items()}


def development_coverage(issues, records, forecasts):
    """Check availability only; do not calculate new inventory model features."""
    result, by_id = [], {row["issue_id"]: row for row in records}
    for forecast in forecasts:
        issued = _day(forecast["forecast_date"])
        target = _day(forecast["target_date"])
        if not date(2017, 1, 1) <= issued < date(2023, 1, 1) or not issued < target < date(2023, 1, 1):
            raise ValueError("Inventory coverage requires development-only forecasts and targets")
        eligible = [row for row in issues if _day(row["release_date"]) < issued]
        latest = max(eligible, key=lambda row: (row["observation_date"], row["release_date"])) if eligible else None
        record = by_id.get(latest["issue_id"]) if latest else None
        coverage = {"row_id": forecast["row_id"], "forecast_date": issued.isoformat(),
                    "target_date": target.isoformat(),
                    "latest_declared_issue_id": latest["issue_id"] if latest else None,
                    "status": "available" if record else "latest_declared_issue_unusable" if latest else "no_earlier_release",
                    "series": {}}
        if record:
            if _day(record["earliest_eligible_issuance_date"]) > issued:
                raise ValueError("Inventory issuance precedes the conservative business-day availability")
            coverage.update({"release_date": record["release_date"],
                             "observation_date": record["observation_date"],
                             "observation_age_calendar_days": (issued - _day(record["observation_date"])).days,
                             "release_age_calendar_days": (issued - _day(record["release_date"])).days})
        for name in SERIES:
            reference_day = _day(record["observation_date"]) - timedelta(days=28) if record else None
            occurrences = []
            for historical in records:
                if _day(historical["release_date"]) >= issued:
                    continue
                if reference_day and reference_day.isoformat() in (
                        historical["observation_date"], historical["previous_observation_date"]):
                    occurrences.append(historical)
            reference = max(occurrences, key=lambda row: (row["release_date"], row["observation_date"])) if occurrences else None
            same_regime = bool(record and reference and
                               record["series"][name]["definition_regime"] == reference["series"][name]["definition_regime"])
            coverage["series"][name] = {
                "stock_available": record is not None, "weekly_change_available": record is not None,
                "four_week_change_available": same_regime,
                "four_week_reference_observation_date": reference_day.isoformat() if reference_day else None,
                "four_week_reference_issue_id": reference["issue_id"] if reference else None,
                "four_week_missing_reason": None if same_regime else
                    "latest_stock_unavailable" if record is None else
                    "definition_regime_mismatch" if reference else "no_available_reference_observation",
            }
        result.append(coverage)
    return result


def _fridays(start, end):
    current = start + timedelta(days=(4 - start.weekday()) % 7)
    while current <= end:
        yield current.isoformat()
        current += timedelta(days=7)


def _range(values):
    return {"minimum": min(values), "median": statistics.median(values), "maximum": max(values)} if values else None


def summarize(issues, records, failures, overlap_summary, coverage):
    observed = {row["observation_date"] for row in issues}
    usable = {row["observation_date"] for row in records}
    first, last = min(observed), max(observed)
    releases = [row["release_date"] for row in issues]
    groups = defaultdict(list)
    for row in issues:
        groups[row["release_date"]].append(row["issue_id"])
    return {
        "status": audit_config()["status"], "holdout_evaluated": False,
        "issue_count": len(issues), "parsed_issue_count": len(records), "failure_count": len(failures),
        "failures": failures, "release_date_range": [min(releases), max(releases)],
        "current_observation_date_range": [first, last],
        "previous_observation_date_range": [min(row["previous_observation_date"] for row in records),
                                             max(row["previous_observation_date"] for row in records)] if records else None,
        "release_issue_counts_by_year": dict(sorted(Counter(day[:4] for day in releases).items())),
        "parsed_issue_counts_by_year": dict(sorted(Counter(row["release_date"][:4] for row in records).items())),
        "missing_friday_weeks_within_captured_span": sorted(set(_fridays(_day(first), _day(last))) - observed),
        "unusable_friday_weeks_within_span": sorted(set(_fridays(_day(first), _day(last))) - usable),
        "leading_2011_coverage_limit": {
            "no_saved_release_evidence_from": "2011-01-01",
            "no_saved_release_evidence_through": (_day(min(releases)) - timedelta(days=1)).isoformat(),
            "first_saved_release_date": min(releases),
            "unrepresented_current_observation_fridays": list(_fridays(date(2011, 1, 1), _day(first) - timedelta(days=1))),
            "qualification": "missing from this captured release evidence; not proof that every older file is unavailable; prior-week values cannot be backfilled before their release",
        },
        "release_lag_calendar_days_counts": dict(sorted(Counter(
            str((_day(row["release_date"]) - _day(row["observation_date"])).days) for row in issues).items())),
        "nonstandard_release_lags": [{"issue_id": row["issue_id"], "release_date": row["release_date"],
                                      "observation_date": row["observation_date"],
                                      "lag_calendar_days": (_day(row["release_date"]) - _day(row["observation_date"])).days}
                                     for row in issues if (_day(row["release_date"]) - _day(row["observation_date"])).days != 5],
        "lag_interpretation": "observed calendar lags; no automatic attribution to holidays or disruptions",
        "multiple_issues_on_one_release_date": {day: ids for day, ids in groups.items() if len(ids) > 1},
        "rounding_residuals": {name: {
            "nonzero_count": sum(row["series"][name]["rounding_residual"] != 0 for row in records),
            "maximum_absolute_million_barrels": max((abs(row["series"][name]["rounding_residual"]) for row in records), default=None),
        } for name in SERIES},
        "series_parsed_issue_counts": {name: sum(name in row["series"] for row in records) for name in SERIES},
        "repeated_observation_comparisons": overlap_summary,
        "revisions_qualification": "compares current/prior-week values across currently saved archive issues; unchanged overlaps do not prove the files are original first releases",
        "development_coverage": {
            "forecast_count": len(coverage),
            "forecast_date_range": [coverage[0]["forecast_date"], coverage[-1]["forecast_date"]] if coverage else None,
            "status_counts": dict(Counter(row["status"] for row in coverage)),
            "observation_age_calendar_days": _range([row["observation_age_calendar_days"] for row in coverage if row["status"] == "available"]),
            "release_age_calendar_days": _range([row["release_age_calendar_days"] for row in coverage if row["status"] == "available"]),
            "by_series": {name: {field: sum(row["series"][name][field] for row in coverage)
                                  for field in ("stock_available", "weekly_change_available", "four_week_change_available")}
                          for name in SERIES},
        },
    }


def _evidence(manifest_path, records):
    """Hash auxiliary release/definition evidence and cross-check Table 1 stock rows."""
    if manifest_path is None:
        return None, [], []
    payload = manifest_path.read_bytes()
    manifest = json.loads(payload)
    if manifest.get("complete") is not True:
        raise ValueError("Inventory evidence snapshot is incomplete")
    protected, verified, checks = [manifest_path], {}, []
    by_id = {row["issue_id"]: row for row in records}
    for source in manifest["sources"]:
        url = source["requested_url"]
        parts = urlsplit(url)
        if parts.scheme != "https" or parts.netloc != "www.eia.gov":
            raise ValueError("Auxiliary evidence must use official EIA sources")
        archive = re.fullmatch(r"/petroleum/supply/weekly/archive/(\d{4})/([^/]+)/(?:csv/table1\.csv|pdf/appendixb\.pdf)", parts.path)
        if url not in DEFINITION_AND_RELEASE_NOTICES:
            if (not archive or not FIRST_YEAR <= int(archive[1]) <= LAST_YEAR
                    or archive[2] not in by_id or parts.query or parts.fragment):
                raise ValueError("Auxiliary inventory evidence crosses its bounded historical scope")
            prefix = by_id[archive[2]]["table_url"].removesuffix("csv/table4.csv")
            if url not in (prefix + "csv/table1.csv", prefix + "pdf/appendixb.pdf"):
                raise ValueError("Auxiliary inventory evidence differs from its canonical issue URL")
        path, raw = _verified(manifest_path.parent, source, url)
        if raw is None or path.name in verified:
            raise ValueError("Missing or duplicate inventory evidence source")
        protected.append(path)
        verified[path.name] = {key: source[key] for key in ("requested_url", "sha256", "bytes")}
        match = re.fullmatch(r"/petroleum/supply/weekly/archive/(\d{4})/([^/]+)/csv/table1\.csv", parts.path)
        if match:
            record = by_id.get(match[2])
            if record is None:
                raise ValueError("Table 1 cross-check has no usable Table 4 issue")
            rows = list(csv.reader(io.StringIO(raw.decode("cp1252")), strict=True))
            header = rows[0]
            if len(header) != 8 or [header[index] for index in (0, 3, 4, 6, 7)] != ["STUB_1", "Difference", "Percent Change", "Difference", "Percent Change"]:
                raise ValueError("Unsupported Table 1 stock-section schema")
            if (datetime.strptime(header[1], "%m/%d/%y").date().isoformat() != record["observation_date"]
                    or datetime.strptime(header[2], "%m/%d/%y").date().isoformat() != record["previous_observation_date"]):
                raise ValueError("Table 1 and Table 4 stock dates differ")
            selected = [row for row in rows if row and row[0] == "Commercial (Excluding SPR)"]
            if len(selected) != 1 or len(selected[0]) != 8:
                raise ValueError("Table 1 commercial stock row is missing or ambiguous")
            keys = ("current_million_barrels", "previous_million_barrels", "published_change_million_barrels")
            for index, key in enumerate(keys, 1):
                if float(_number(selected[0][index], key)) != record["series"]["commercial_crude"][key]:
                    raise ValueError("Table 1 and Table 4 commercial stocks differ")
            checks.append({"issue_id": record["issue_id"], "matched_fields": list(keys), "match": True})
    return {"manifest_sha256": sha256(payload), "verified_artifacts": verified}, protected, checks


def _probes(manifest_path, first_release_date):
    """Preserve bounded failed-resource probes without treating them as issues."""
    if manifest_path is None:
        return None, []
    payload = manifest_path.read_bytes()
    manifest = json.loads(payload)
    sources = manifest.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("Historical inventory probe evidence is empty")
    seen = set()
    for source in sources:
        candidate = _day(source["candidate_release_date"])
        slug = candidate.strftime("%Y_%m_%d")
        expected_url = f"{ARCHIVE_INDEX}2011/{slug}/csv/table4.csv"
        if (candidate.year != FIRST_YEAR or candidate >= _day(first_release_date)
                or source.get("requested_url") != expected_url or candidate in seen
                or source.get("status") != "download_failed" or source.get("http_status") != 404
                or source.get("error_type") != "HTTPError"):
            raise ValueError("Inventory probes must be bounded failed early-2011 resource requests")
        _basename(manifest_path.parent, source.get("file"))
        attempts = source.get("attempts")
        if not isinstance(attempts, list) or not attempts or any(
                row.get("status") != "download_failed" or row.get("http_status") != 404
                or row.get("error_type") != "HTTPError" or not row.get("attempted_at_utc") for row in attempts):
            raise ValueError("Inventory probe attempts do not substantiate the declared HTTP failures")
        seen.add(candidate)
    return {"manifest_sha256": sha256(payload), "probe_count": len(sources), "sources": sources,
            "qualification": "These requested modern archive paths returned HTTP 404 at capture time; this does not prove that all older historical inventory files are unavailable."}, [manifest_path]


def _quarantine(manifest_path, issues, failures):
    """Authenticate stale canonical CSV responses without admitting their values."""
    if manifest_path is None:
        return None, []
    payload = manifest_path.read_bytes()
    manifest = json.loads(payload)
    if manifest.get("complete") is not True or not manifest.get("sources"):
        raise ValueError("Inventory quarantine evidence is incomplete")
    failed = {row["issue_id"]: row for row in failures if row["status"] == "download_failed"}
    allowed = {row["table_url"]: row for row in issues if row["issue_id"] in failed}
    protected, records, seen = [manifest_path], [], set()
    for source in manifest["sources"]:
        issue = allowed.get(source.get("requested_url"))
        if issue is None or issue["issue_id"] in seen:
            raise ValueError("Quarantine evidence must correspond to a unique failed canonical Table 4 source")
        expected = failed[issue["issue_id"]]["captured_response_metadata"]
        if any(source.get(key) != expected.get(key) or key not in expected for key in ("bytes", "sha256")):
            raise ValueError("Quarantined response differs from the original failed capture")
        path, raw = _verified(manifest_path.parent, source, issue["table_url"])
        if raw is None:
            raise ValueError("Quarantine evidence bytes were not captured")
        try:
            header = next(csv.reader(io.StringIO(raw.decode("cp1252")), strict=True))
            current = datetime.strptime(header[1], "%m/%d/%y").date()
            previous = datetime.strptime(header[2], "%m/%d/%y").date()
        except (StopIteration, IndexError, ValueError, csv.Error) as exc:
            raise ValueError("Quarantine CSV does not contain readable date headers") from exc
        if current.isoformat() == issue["observation_date"]:
            raise ValueError("Quarantine CSV does not demonstrate a stale observation header")
        records.append({"issue_id": issue["issue_id"], "requested_url": issue["table_url"],
                        "file": path.name, "sha256": sha256(raw), "bytes": len(raw),
                        "expected_observation_date": issue["observation_date"],
                        "actual_current_date_header": current.isoformat(),
                        "actual_previous_date_header": previous.isoformat(),
                        "excluded_from_inventory_records": True})
        protected.append(path)
        seen.add(issue["issue_id"])
    return {"manifest_sha256": sha256(payload), "verified_stale_responses": records}, protected


def run(snapshot_dir: Path, output_root: Path = Path("data/audits/inventories"),
        report: Path = Path("reports/inventory_audit.json"), *, reference_report: Path = WTI_REPORT,
        evidence_manifest: Path | None = None, probe_manifest: Path | None = None,
        quarantine_manifest: Path | None = None):
    snapshot_dir, output_root, report, reference_report = [workspace_path(str(path)) for path in
                                                         (snapshot_dir, output_root, report, reference_report)]
    if evidence_manifest is not None:
        evidence_manifest = workspace_path(str(evidence_manifest))
    if probe_manifest is not None:
        probe_manifest = workspace_path(str(probe_manifest))
    if quarantine_manifest is not None:
        quarantine_manifest = workspace_path(str(quarantine_manifest))
    source, issues, records, failures, protected = load_snapshot(snapshot_dir)
    reference, _, forecasts = _load_saved_run(reference_report, "wti")
    evidence, evidence_paths, checks = _evidence(evidence_manifest, records)
    probes, probe_paths = _probes(probe_manifest, min(row["release_date"] for row in issues))
    quarantine, quarantine_paths = _quarantine(quarantine_manifest, issues, failures)
    protected += evidence_paths + probe_paths + quarantine_paths + [reference_report, WTI_REPORT, Path("reports/brent_comparison.json"),
                                   Path("reports/development_benchmarks.json")]
    protected += [Path(reference["evaluation_directory"]) / name for name in
                  ("manifest.json", "summary.json", "predictions.jsonl")]
    changes, overlap_summary = compare_overlaps(records)
    coverage = development_coverage(issues, records, forecasts)
    summary = summarize(issues, records, failures, overlap_summary, coverage)
    summary["table1_commercial_stock_crosschecks"] = checks
    summary["bounded_early_2011_resource_probes"] = probes
    summary["quarantined_source_evidence"] = quarantine
    code_paths = ("oil_audit/inventory.py", "oil_audit/inventory_tables.py", "oil_audit/inventory_sources.py",
                  "oil_audit/calendar.py", "oil_audit/dataset.py", "oil_eval/benchmarks.py", "oil_eval/data.py",
                  "oil_eval/reference.py", "oil_eval/protocol.py", "oil_eval/models.py", "oil_eval/metrics.py",
                  "oil_audit/features.py")
    identity = {"source_snapshot": source, "reference_run": reference, "auxiliary_evidence": evidence,
                "bounded_resource_probe_evidence": probes,
                "quarantined_source_evidence": quarantine,
                "config": audit_config(),
                "audit_code_sha256": {path: sha256(Path(path).read_bytes()) for path in code_paths}}
    audit_id = sha256(json_bytes(identity))[:20]
    destination = output_root / audit_id
    outputs = {"records.jsonl": jsonl_bytes(records), "overlap_changes.jsonl": jsonl_bytes(changes),
               "development_coverage.jsonl": jsonl_bytes(coverage), "summary.json": json_bytes(summary)}
    manifest = {"audit_id": audit_id, **identity,
                "artifacts": {name: {"sha256": sha256(raw), "bytes": len(raw)} for name, raw in outputs.items()}}
    outputs["manifest.json"] = json_bytes(manifest)
    protected += [destination / name for name in outputs]
    if (report.resolve() in {path.resolve() for path in protected}
            or any(report.resolve().is_relative_to(root.resolve()) for root in
                   (output_root, Path("data/audits/inventories")))
            or (report.name in outputs and (report.parent / "manifest.json").exists())):
        raise ValueError("Inventory audit report cannot overwrite inputs, original reports or immutable artifacts")
    for name, raw in outputs.items():
        path = destination / name
        if path.exists() and path.read_bytes() != raw:
            raise ValueError("Refusing to replace a different inventory audit artifact: " + name)
    destination.mkdir(parents=True, exist_ok=True)
    for name, raw in outputs.items():
        if not (destination / name).exists():
            (destination / name).write_bytes(raw)
    result = {"audit_id": audit_id, "audit_directory": str(destination),
              "source_snapshot_directory": str(snapshot_dir), "reference_run": reference,
              "auxiliary_evidence": evidence, **summary}
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_bytes(json_bytes(result))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-dir", type=workspace_path, required=True)
    parser.add_argument("--reference-report", type=workspace_path, default=WTI_REPORT)
    parser.add_argument("--evidence-manifest", type=workspace_path)
    parser.add_argument("--probe-manifest", type=workspace_path)
    parser.add_argument("--quarantine-manifest", type=workspace_path)
    parser.add_argument("--output-root", type=workspace_path, default=Path("data/audits/inventories"))
    parser.add_argument("--report", type=workspace_path, default=Path("reports/inventory_audit.json"))
    args = parser.parse_args()
    result = run(args.snapshot_dir, args.output_root, args.report, reference_report=args.reference_report,
                 evidence_manifest=args.evidence_manifest, probe_manifest=args.probe_manifest,
                 quarantine_manifest=args.quarantine_manifest)
    print(json.dumps({key: result[key] for key in ("audit_id", "audit_directory", "issue_count",
                                                 "parsed_issue_count", "failure_count", "development_coverage",
                                                 "holdout_evaluated")}, indent=2))


if __name__ == "__main__":
    main()

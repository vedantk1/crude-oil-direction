"""Frozen, release-qualified inventory inputs for development-only forecasts.

Archived releases support the documented conservative date convention, not a
claim that today's archive files are unchanged original releases. All feature
values are constructed at their own historical issuance, before model fitting.
"""

from collections import Counter
from copy import deepcopy
from datetime import date, timedelta
import json
import math
from pathlib import Path
import re

from oil_audit.calendar import CALENDAR_ID, is_business_day, next_business_day
from oil_audit.dataset import json_bytes, jsonl_bytes, sha256
from oil_audit.features import WTI_FEATURE_COLUMNS


INVENTORY_FEATURE_COLUMNS = (
    "commercial_inventory_change_1_week_million_barrels",
    "commercial_inventory_change_4_weeks_million_barrels",
    "cushing_inventory_change_1_week_million_barrels",
    "cushing_inventory_change_4_weeks_million_barrels",
    "inventory_report_age_calendar_days",
)
INVENTORY_MODEL_COLUMNS = WTI_FEATURE_COLUMNS + INVENTORY_FEATURE_COLUMNS
_SERIES = ("commercial_crude", "cushing")
_START, _END = date(2011, 1, 1), date(2023, 1, 1)
_DEFINITION_CHANGE = date(2016, 10, 13)
_ARTIFACTS = ("records.jsonl", "overlap_changes.jsonl", "development_coverage.jsonl", "summary.json")
_AUDIT_CODE = {
    "oil_audit/inventory.py", "oil_audit/inventory_tables.py", "oil_audit/inventory_sources.py",
    "oil_audit/calendar.py", "oil_audit/dataset.py", "oil_eval/benchmarks.py", "oil_eval/data.py",
    "oil_eval/reference.py", "oil_eval/protocol.py", "oil_eval/models.py", "oil_eval/metrics.py",
    "oil_audit/features.py",
}


def _day(value):
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        raise ValueError("Expected an ISO inventory feature date")
    return date.fromisoformat(value)


def _checksum(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate inventory reference JSON key")
        result[key] = value
    return result


def _invalid_constant(value):
    raise ValueError("Non-finite inventory reference JSON value")


def _decode(payload):
    try:
        result = json.loads(payload, object_pairs_hook=_object,
                            parse_constant=_invalid_constant)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError("Invalid inventory reference JSON") from exc
    if not isinstance(result, dict):
        raise ValueError("Inventory reference must be a JSON object")
    return result


def _local_path(value):
    if not isinstance(value, (str, Path)) or not str(value):
        raise ValueError("Inventory reference paths must be workspace-relative")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or not path.resolve().is_relative_to(Path.cwd().resolve()):
        raise ValueError("Inventory reference paths must stay inside the workspace")
    return path


def _checked_bytes(path, expected):
    if (not isinstance(expected, dict) or set(expected) != {"sha256", "bytes"}
            or not _checksum(expected.get("sha256")) or type(expected.get("bytes")) is not int
            or expected["bytes"] < 0):
        raise ValueError("Invalid inventory artifact checksum metadata")
    raw = _local_path(path).read_bytes()
    if len(raw) != expected["bytes"] or sha256(raw) != expected["sha256"]:
        raise ValueError("Inventory artifact checksum mismatch: " + path.name)
    return raw


def load_inventory_reference(audit_report: Path):
    """Authenticate a closed historical audit and re-extract its primary sources.

    Every saved audit artifact is checked before any inventory values are
    admitted. The original snapshot's index and every declared successful source
    are independently authenticated by ``load_snapshot``. Recorded audit code
    hashes remain historical provenance and need not match today's code.
    Auxiliary evidence identities are inherited from the authenticated audit;
    their bytes are not model inputs and their locations are not in its manifest.
    """
    # Lazy import prevents models -> inventory_features -> inventory -> models.
    from oil_audit.inventory import (audit_config, compare_overlaps, development_coverage,
                                     load_snapshot, summarize)

    report_path = _local_path(audit_report)
    report_bytes = report_path.read_bytes()
    report = _decode(report_bytes)
    directory = _local_path(report.get("audit_directory"))
    snapshot = _local_path(report.get("source_snapshot_directory"))
    manifest_bytes = _local_path(directory / "manifest.json").read_bytes()
    manifest = _decode(manifest_bytes)
    keys = {"audit_id", "source_snapshot", "reference_run", "auxiliary_evidence",
            "bounded_resource_probe_evidence", "quarantined_source_evidence", "config",
            "audit_code_sha256", "artifacts"}
    if set(manifest) != keys:
        raise ValueError("Unexpected inventory audit manifest fields")
    identity = {key: value for key, value in manifest.items() if key not in ("audit_id", "artifacts")}
    if (sha256(json_bytes(identity))[:20] != manifest.get("audit_id")
            or report.get("audit_id") != manifest["audit_id"] or directory.name != manifest["audit_id"]):
        raise ValueError("Inventory audit identity does not match its manifest")
    if manifest.get("config") != audit_config() or manifest["config"].get("calendar_id") != CALENDAR_ID:
        raise ValueError("Inventory audit differs from the closed historical availability scope")
    code = manifest.get("audit_code_sha256")
    if not isinstance(code, dict) or set(code) != _AUDIT_CODE or not all(map(_checksum, code.values())):
        raise ValueError("Invalid historical inventory audit code provenance")
    if not isinstance(manifest.get("artifacts"), dict) or set(manifest["artifacts"]) != set(_ARTIFACTS):
        raise ValueError("Inventory audit requires exactly four expected artifacts")
    payloads = {name: _checked_bytes(directory / name, manifest["artifacts"][name]) for name in _ARTIFACTS}
    summary = _decode(payloads["summary.json"])
    expected_report = {"audit_id": manifest["audit_id"], "audit_directory": str(directory),
                       "source_snapshot_directory": str(snapshot), "reference_run": manifest["reference_run"],
                       "auxiliary_evidence": manifest["auxiliary_evidence"], **summary}
    if report != expected_report or summary.get("holdout_evaluated") is not False:
        raise ValueError("Inventory report does not match its closed-holdout audit")
    # Source release bounds are checked before any Table 4 values are parsed.
    source, issues, records, failures, _ = load_snapshot(snapshot)
    if source != manifest["source_snapshot"]:
        raise ValueError("Inventory primary source identity differs from the saved audit")
    if jsonl_bytes(records) != payloads["records.jsonl"]:
        raise ValueError("Inventory records differ from authenticated primary sources")
    changes, overlaps = compare_overlaps(records)
    if jsonl_bytes(changes) != payloads["overlap_changes.jsonl"]:
        raise ValueError("Inventory overlap artifact differs from authenticated primary sources")
    coverage = []
    for line in payloads["development_coverage.jsonl"].splitlines():
        # Reject held-out dates before decoding an artifact record.
        gates = re.findall(rb'"forecast_date"\s*:\s*"(\d{4}-\d{2}-\d{2})"', line)
        if len(gates) != 1 or not date(2017, 1, 1) <= _day(gates[0].decode()) < _END:
            raise ValueError("Inventory audit coverage crosses the development boundary")
        coverage.append(_decode(line))
    if (len({row["forecast_date"] for row in coverage}) != len(coverage)
            or [row["forecast_date"] for row in coverage] != sorted(row["forecast_date"] for row in coverage)
            or development_coverage(issues, records, coverage) != coverage):
        raise ValueError("Inventory audit coverage differs from authenticated sources")
    reconstructed = summarize(issues, records, failures, overlaps, coverage)
    if any(summary.get(key) != value for key, value in reconstructed.items()):
        raise ValueError("Inventory audit summary differs from authenticated sources")
    if (summary.get("bounded_early_2011_resource_probes") != manifest["bounded_resource_probe_evidence"]
            or summary.get("quarantined_source_evidence") != manifest["quarantined_source_evidence"]):
        raise ValueError("Inventory auxiliary evidence identities differ")
    reference = {
        "audit_id": manifest["audit_id"], "audit_directory": str(directory),
        "report_sha256": sha256(report_bytes), "manifest_sha256": sha256(manifest_bytes),
        "artifacts": manifest["artifacts"], "source_snapshot_directory": str(snapshot),
        "source_snapshot": source, "reference_run": manifest["reference_run"],
        "audit_code_sha256": code, "config": manifest["config"],
        "auxiliary_evidence": manifest["auxiliary_evidence"],
        "bounded_resource_probe_evidence": manifest["bounded_resource_probe_evidence"],
        "quarantined_source_evidence": manifest["quarantined_source_evidence"],
    }
    return reference, issues, records


def _prepare(issues, records):
    declared, usable = {}, {}
    for issue in issues:
        released, observed = _day(issue.get("release_date")), _day(issue.get("observation_date"))
        issue_id = issue.get("issue_id")
        if (not isinstance(issue_id, str) or not issue_id or issue_id in declared
                or not _START <= released < _END or observed >= released or observed.weekday() != 4):
            raise ValueError("Invalid, duplicate or out-of-scope declared inventory issue")
        declared[issue_id] = issue
    for record in records:
        issue_id = record.get("issue_id")
        issue = declared.get(issue_id)
        if issue is None or issue_id in usable or any(record.get(key) != issue.get(key) for key in
                                                     ("release_date", "observation_date")):
            raise ValueError("Inventory record differs from its declared issue")
        released, observed = _day(record["release_date"]), _day(record["observation_date"])
        if (_day(record.get("earliest_eligible_issuance_date")) != next_business_day(released)
                or _day(record.get("previous_observation_date")) != observed - timedelta(days=7)
                or not _checksum(record.get("source_sha256"))):
            raise ValueError("Invalid inventory record availability or source provenance")
        if not isinstance(record.get("series"), dict) or set(record["series"]) != set(_SERIES):
            raise ValueError("Inventory record must contain both stock series")
        for name in _SERIES:
            values = record["series"][name]
            regime = ("cushing_reported" if name == "cushing" else
                      "includes_lease_stocks" if released < _DEFINITION_CHANGE else "excludes_lease_stocks")
            if values.get("definition_regime") != regime:
                raise ValueError("Inventory stock definition differs from its release regime")
            for field in ("current_million_barrels", "previous_million_barrels", "published_change_million_barrels"):
                value = values.get(field)
                if type(value) not in (int, float) or not math.isfinite(value):
                    raise ValueError("Inventory input must be a finite numeric value")
        usable[issue_id] = record
    return declared, usable


def _source(issue, record=None, series=None, column=None):
    if issue is None:
        return None
    result = {key: issue[key] for key in ("issue_id", "release_date", "observation_date")}
    result["source_sha256"] = record["source_sha256"] if record else None
    result["earliest_eligible_issuance_date"] = next_business_day(_day(issue["release_date"])).isoformat()
    if record is not None:
        result["previous_observation_date"] = record["previous_observation_date"]
    if series is not None and record is not None:
        result["definition_regime"] = record["series"][series]["definition_regime"]
    if column:
        result["column"] = column
        result["value_observation_date"] = record[
            "previous_observation_date" if column == "previous" else "observation_date"]
    return result


def attach_inventory_features(feature_rows, issues, records):
    """Return copied rows and missingness counts, without dropping any forecast.

    An unusable latest declared report blocks all stock changes. Its release
    date still establishes report age. Earlier rows never see later revisions;
    four-week reference values may use either current or prior columns of the
    latest version already released at that row's own issuance.
    """
    declared, usable = _prepare(issues, records)
    result, status_counts = [], Counter()
    available = Counter({name: 0 for name in INVENTORY_FEATURE_COLUMNS})
    reasons = {name: Counter() for name in INVENTORY_FEATURE_COLUMNS}
    for original in feature_rows:
        issued = _day(original.get("forecast_date"))
        target = _day(original.get("target_date"))
        if not _START <= issued < target < _END or not is_business_day(issued):
            raise ValueError("Inventory features require development-only business-day forecasts")
        if "inventory_provenance" in original or any(name in original.get("features", {}) for name in INVENTORY_FEATURE_COLUMNS):
            raise ValueError("Inventory features have already been frozen for this forecast")
        cutoff = issued - timedelta(days=1)
        if ("information_cutoff_date" in original
                and original["information_cutoff_date"] != cutoff.isoformat()):
            raise ValueError("Inventory information cutoff differs from the preceding date")
        candidates = [issue for issue in declared.values()
                      if _day(issue["release_date"]) < issued
                      and next_business_day(_day(issue["release_date"])) <= issued]
        latest = max(candidates, key=lambda item: (item["observation_date"], item["release_date"])) if candidates else None
        current = usable.get(latest["issue_id"]) if latest else None
        status = "available" if current else "latest_declared_issue_unusable" if latest else "no_earlier_release"
        values, feature_provenance = dict.fromkeys(INVENTORY_FEATURE_COLUMNS), {}
        for column in INVENTORY_FEATURE_COLUMNS:
            feature_provenance[column] = {"status": "missing", "missing_reason": status,
                                          "current": _source(latest, current), "reference": None}
        if latest:
            age = INVENTORY_FEATURE_COLUMNS[-1]
            values[age] = (issued - _day(latest["release_date"])).days
            feature_provenance[age].update(status="available", missing_reason=None)
        if current:
            reference_date = _day(current["observation_date"]) - timedelta(days=28)
            occurrences = []
            for historical in usable.values():
                if (_day(historical["release_date"]) >= issued
                        or _day(historical["earliest_eligible_issuance_date"]) > issued):
                    continue
                for role, key in (("current", "observation_date"), ("previous", "previous_observation_date")):
                    if historical[key] == reference_date.isoformat():
                        occurrences.append((historical, role))
            reference, role = max(occurrences, key=lambda pair: (
                pair[0]["release_date"], pair[0]["observation_date"])) if occurrences else (None, None)
            for index, name in enumerate(_SERIES):
                weekly, four_week = INVENTORY_FEATURE_COLUMNS[index * 2:index * 2 + 2]
                stock = current["series"][name]
                values[weekly] = stock["published_change_million_barrels"]
                feature_provenance[weekly].update(status="available", missing_reason=None,
                                                  current=_source(latest, current, name, "current"))
                same_regime = reference and stock["definition_regime"] == reference["series"][name]["definition_regime"]
                if same_regime:
                    values[four_week] = stock["current_million_barrels"] - reference["series"][name][role + "_million_barrels"]
                feature_provenance[four_week].update(
                    status="available" if same_regime else "missing",
                    missing_reason=None if same_regime else "definition_regime_mismatch" if reference else "no_available_reference_observation",
                    current=_source(latest, current, name, "current"),
                    reference=_source(reference, reference, name, role),
                    reference_observation_date=reference_date.isoformat())
        row = deepcopy(original)
        row.setdefault("features", {}).update(values)
        row["inventory_provenance"] = {"information_cutoff_date": cutoff.isoformat(), "status": status,
                                        "latest_declared_issue": _source(latest, current),
                                        "features": feature_provenance}
        result.append(row)
        status_counts[status] += 1
        for name, value in values.items():
            if value is None:
                reasons[name][feature_provenance[name]["missing_reason"]] += 1
            else:
                available[name] += 1
    summary = {
        "row_count": len(result), "latest_issue_status_counts": dict(sorted(status_counts.items())),
        "feature_available_counts": dict(available),
        "feature_missing_counts": {name: len(result) - available[name] for name in INVENTORY_FEATURE_COLUMNS},
        "feature_missing_reasons": {name: dict(sorted(counts.items())) for name, counts in reasons.items()},
        "information_cutoff": "release date strictly before each historical forecast; earliest eligible issuance no later than forecast",
        "frozen_at_historical_issuance": True,
    }
    return result, summary

"""Authenticated, pre-holdout source views for one extra business-day delay."""

from collections import Counter
from copy import deepcopy
from datetime import date, datetime, time, timedelta
import hashlib
import json
from pathlib import Path
import re
from zoneinfo import ZoneInfo

from oil_audit.availability import PointInTimeSeries
from oil_audit.bounded_vintages import bounded_records
from oil_audit.calendar import add_business_days, is_business_day, next_business_day, previous_business_day
from oil_audit.capture import capture_bounds
from oil_audit.dataset import json_bytes, sha256
from oil_audit.features import WTI_FEATURE_COLUMNS, _one_series
from oil_audit import vintages
from .inventory_features import INVENTORY_FEATURE_COLUMNS, INVENTORY_MODEL_COLUMNS, attach_inventory_features
from .protocol import HOLDOUT_START, SCORING_AS_OF, TRAIN_START


DELAY_BUSINESS_DAYS = 1
DELAY_POLICY_ID = "additional-one-us-business-day-v1"
WTI_SERIES_ID = "DCOILWTICO"


def _date(value):
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        raise ValueError("Expected an ISO availability-delay date")
    return date.fromisoformat(value)


def _local_path(value):
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or not path.resolve().is_relative_to(Path.cwd().resolve()):
        raise ValueError("Availability-delay source paths must stay workspace-relative")
    return path


def delay_information_dates(forecast_date: date) -> tuple[date, date]:
    """Return delayed label-knowledge date and inclusive source cutoff.

    Labels already carry next-business-day availability, so use their state
    at P, the previous business day. Source versions/releases use P minus one
    calendar day. The actual forecast, its target, and feature ages stay at D.
    """
    if type(forecast_date) is not date or not TRAIN_START <= forecast_date < HOLDOUT_START:
        raise ValueError("Availability delay requires a pre-holdout forecast date")
    if not is_business_day(forecast_date):
        raise ValueError("Availability delay requires a US business-day forecast")
    knowledge = previous_business_day(forecast_date)
    return knowledge, knowledge - timedelta(days=1)


def load_delay_wti_source(source_dir: Path, expected_source_manifest_sha256: str):
    """Authenticate original archive bytes, then admit only pre-2023 WTI data."""
    root = _local_path(source_dir)
    manifest_bytes = (root / "manifest.json").read_bytes()
    if sha256(manifest_bytes) != expected_source_manifest_sha256:
        raise ValueError("ALFRED source manifest differs from the authenticated dataset")
    records = json.loads(manifest_bytes)
    if not isinstance(records, list) or not records:
        raise ValueError("ALFRED source manifest must contain source records")
    verified = {}
    for record in records:
        path = _local_path(record["path"])
        if path.parent != root or str(path) in verified:
            raise ValueError("Unexpected or duplicate ALFRED source manifest path")
        checksum, count = hashlib.sha256(), 0
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                checksum.update(block)
                count += len(block)
        if checksum.hexdigest() != record.get("sha256") or count != record.get("bytes"):
            raise ValueError("ALFRED source checksum mismatch: " + path.name)
        verified[str(path)] = {"sha256": checksum.hexdigest(), "bytes": count}
    parts = sorted(root.glob(WTI_SERIES_ID + "-realtime-part*.zip"))
    page = root / (WTI_SERIES_ID + "-download-page.html")
    if not parts or not all(str(path) in verified for path in [*parts, page]):
        raise ValueError("ALFRED manifest omits a required WTI input")
    bound = capture_bounds(records, (WTI_SERIES_ID,))[WTI_SERIES_ID]
    if bound < datetime.combine(SCORING_AS_OF, time(18), tzinfo=ZoneInfo("America/New_York")):
        raise ValueError("ALFRED capture does not cover the development period")
    listed, _ = vintages.form_metadata(WTI_SERIES_ID, root)
    listed_pre_holdout = {_date(day) for day in listed if _date(day) < HOLDOUT_START}
    retained = bounded_records(WTI_SERIES_ID, root, SCORING_AS_OF)
    wti = PointInTimeSeries(WTI_SERIES_ID, retained)
    starts = {version.vintage_start for values in wti.versions_by_observation.values() for version in values}
    if not starts.issubset(listed_pre_holdout) or not listed_pre_holdout or (
            min(starts) != min(listed_pre_holdout) or max(starts) != max(listed_pre_holdout)):
        raise ValueError("Bounded WTI archive differs from its captured vintage-date coverage")
    provenance = {"source_manifest_sha256": sha256(manifest_bytes), "source_directory": str(root),
                  "verified_source_artifacts": verified, "series_id": WTI_SERIES_ID,
                  "source_value_boundary_exclusive": HOLDOUT_START.isoformat(),
                  "future_interval_ends": "masked before numeric parsing",
                  "retained_interval_count": len(retained), "retained_intervals_sha256": sha256(json_bytes(retained))}
    return provenance, wti


def _validate_rows(rows):
    seen = set()
    for row in rows:
        issued, target = _date(row["forecast_date"]), _date(row["target_date"])
        if not TRAIN_START <= issued < target < HOLDOUT_START or not is_business_day(issued):
            raise ValueError("Delay features require bounded historical forecast and target dates")
        if target != add_business_days(issued, 5) or row["row_id"] != "wti-direction-" + issued.strftime("%Y%m%d"):
            raise ValueError("Delay feature identity differs from the fixed original target")
        if row["row_id"] in seen:
            raise ValueError("Duplicate delay forecast identity")
        seen.add(row["row_id"])


def _wti_features(wti, issued, cutoff):
    values, provenance, _ = _one_series(wti, cutoff, issued)
    features = {"wti_" + name: value for name, value in values.items()}
    return features, provenance, {"wti": features["wti_latest_price"] is not None,
                                 "complete_wti_features": all(features[name] is not None for name in WTI_FEATURE_COLUMNS)}


def reconstruct_baseline_wti_rows(original_rows, wti):
    """Require every original WTI input/eligibility to reproduce before delay."""
    _validate_rows(original_rows)
    reconstructed, expected, actual = [], [], []
    for original in original_rows:
        issued = _date(original["forecast_date"])
        features, provenance, eligibility = _wti_features(wti, issued, issued - timedelta(days=1))
        expected.append({"row_id": original["row_id"], "features": {
            name: original["features"][name] for name in WTI_FEATURE_COLUMNS}, "eligible": original["eligibility"]["wti"]})
        actual.append({"row_id": original["row_id"], "features": features, "eligible": eligibility["wti"]})
        if actual[-1] != expected[-1]:
            raise ValueError("Baseline WTI archive reconstruction differs at " + original["forecast_date"])
        row = deepcopy(original)
        row["features"] = features
        row["provenance"] = {"wti": provenance}
        row["eligibility"] = eligibility
        reconstructed.append(row)
    return reconstructed, {"row_count": len(actual), "all_wti_features_and_eligibility_identical": True,
                           "original_wti_projection_sha256": sha256(json_bytes(expected)),
                           "reconstructed_wti_projection_sha256": sha256(json_bytes(actual))}


def build_delayed_feature_rows(original_rows, wti, issues, records):
    """Freeze exact-date delayed inputs at each unchanged original issuance."""
    _validate_rows(original_rows)
    # These metadata checks precede any inventory value lookup. The caller's
    # authenticated inventory loader also verifies source bounds and values.
    for item in [*issues, *records]:
        released, observed = _date(item["release_date"]), _date(item["observation_date"])
        if not date(2011, 1, 1) <= released < HOLDOUT_START or observed >= HOLDOUT_START:
            raise ValueError("Delayed inventory view cannot admit holdout sources")
    result, statuses, changes = [], Counter(), Counter()
    for original in original_rows:
        issued = _date(original["forecast_date"])
        knowledge, cutoff = delay_information_dates(issued)
        features, provenance, eligibility = _wti_features(wti, issued, cutoff)
        base = {key: deepcopy(value) for key, value in original.items() if key != "inventory_provenance"}
        base.update(features=features, provenance={"wti": provenance}, eligibility=eligibility,
                    information_cutoff_date=(issued - timedelta(days=1)).isoformat())
        eligible_issues = [issue for issue in issues
                           if next_business_day(_date(issue["release_date"])) <= knowledge]
        eligible_ids = {issue["issue_id"] for issue in eligible_issues}
        eligible_records = [record for record in records if record["issue_id"] in eligible_ids]
        attached, _ = attach_inventory_features([base], eligible_issues, eligible_records)
        row = attached[0]
        row["information_cutoff_date"] = cutoff.isoformat()
        row["inventory_provenance"]["information_cutoff_date"] = cutoff.isoformat()
        row["inventory_provenance"]["label_knowledge_date"] = knowledge.isoformat()
        changed = [name for name in INVENTORY_MODEL_COLUMNS if name in original["features"]
                   and original["features"][name] != row["features"][name]]
        row["availability_delay_provenance"] = {
            "policy_id": DELAY_POLICY_ID, "additional_business_days": DELAY_BUSINESS_DAYS,
            "original_information_cutoff_date": (issued - timedelta(days=1)).isoformat(),
            "label_knowledge_date": knowledge.isoformat(), "information_cutoff_date": cutoff.isoformat(),
            "feature_age_reference_date": original["forecast_date"],
            "eligible_declared_inventory_issue_count": len(eligible_issues),
            "eligible_usable_inventory_issue_count": len(eligible_records),
            "wti_status": "available" if eligibility["wti"] else "no_numeric_observation_at_delayed_cutoff",
            "changed_feature_columns_vs_original": changed,
        }
        result.append(row)
        statuses[row["availability_delay_provenance"]["wti_status"]] += 1
        changes.update(changed)
    return result, {"row_count": len(result), "policy_id": DELAY_POLICY_ID,
                    "additional_business_days": DELAY_BUSINESS_DAYS,
                    "wti_status_counts": dict(statuses),
                    "feature_change_counts_vs_original": {name: changes[name] for name in INVENTORY_MODEL_COLUMNS},
                    "feature_missing_counts": {name: sum(row["features"][name] is None for row in result)
                                               for name in INVENTORY_MODEL_COLUMNS},
                    "frozen_at_original_forecast": True}

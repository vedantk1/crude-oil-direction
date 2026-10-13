"""Verify dataset bytes and admit only rows allowed by the development lock.

Checksums cover the entire immutable source. Individual JSON records are gated
by dates before JSON decoding, so the evaluation never materializes held-out
features or outcomes. Future interval ends are masked to prevent a later
withdrawal from influencing the development replay.
"""

from collections import defaultdict
from datetime import date, datetime, time, timedelta
import hashlib
import json
import math
from pathlib import Path
import re
from zoneinfo import ZoneInfo

from oil_audit.calendar import CALENDAR_ID, add_business_days, is_business_day
from oil_audit.features import WTI_FEATURE_COLUMNS
from .protocol import HOLDOUT_START, SCORING_AS_OF, TRAIN_START


ARTIFACTS = ("features.jsonl", "label_events.jsonl", "evaluation_labels.jsonl")
STATUSES = {"available", "missing_both", "missing_origin", "missing_target", "quarantined"}
LABEL_FIELDS = (
    "row_id", "forecast_date", "target_date", "available_from", "label_status",
    "label", "origin_price", "target_price", "origin_vintage_start", "target_vintage_start",
)
_DATE_PATTERNS = {
    name: re.compile(rb'"' + name.encode() + rb'"\s*:\s*"(\d{4}-\d{2}-\d{2})"')
    for name in ("forecast_date", "available_from")
}


def _date(value, field: str) -> date:
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        raise ValueError("Expected an ISO date: " + field)
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError("Invalid date: " + field) from exc


def _gate_date(line: bytes, field: str) -> date:
    matches = _DATE_PATTERNS[field].findall(line)
    if len(matches) != 1:
        raise ValueError("Expected exactly one unescaped date gate: " + field)
    return _date(matches[0].decode("ascii"), field)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON object key: " + key)
        result[key] = value
    return result


def _decode(payload: bytes) -> dict:
    try:
        result = json.loads(payload, object_pairs_hook=_unique_object)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError("Invalid development JSON record") from exc
    if not isinstance(result, dict):
        raise ValueError("Expected a JSON object")
    return result


def _gated_records(path: Path, labels: bool = False):
    with path.open("rb") as stream:
        for line in stream:
            if not line.strip():
                raise ValueError("Blank JSONL record: " + path.name)
            forecast = _gate_date(line, "forecast_date")
            if forecast >= HOLDOUT_START:
                continue
            if forecast < TRAIN_START:
                raise ValueError("Forecast precedes the locked training start")
            if labels and _gate_date(line, "available_from") >= HOLDOUT_START:
                continue
            row = _decode(line)
            if _date(row.get("forecast_date"), "forecast_date") != forecast:
                raise ValueError("Forecast gate differs from decoded record")
            yield row


def _finite(value, field: str, nullable: bool = True):
    if value is None and nullable:
        return
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("Expected a finite numeric value: " + field)


def _validate_identity(row: dict) -> tuple[date, date]:
    forecast = _date(row.get("forecast_date"), "forecast_date")
    target = _date(row.get("target_date"), "target_date")
    if row.get("row_id") != "wti-direction-" + forecast.strftime("%Y%m%d"):
        raise ValueError("Row ID does not match forecast date")
    if not is_business_day(forecast) or target != add_business_days(forecast, 5):
        raise ValueError("Forecast or target differs from the fixed five-business-day calendar")
    return forecast, target


def _validate_label(row: dict, feature: dict, event: bool) -> None:
    if not all(key in row for key in LABEL_FIELDS):
        raise ValueError("Incomplete label record")
    forecast, target = _validate_identity(row)
    if any(row[key] != feature[key] for key in ("row_id", "forecast_date", "target_date")):
        raise ValueError("Label and feature identities differ")
    available = _date(row["available_from"], "available_from")
    if available < forecast or available > SCORING_AS_OF or not is_business_day(available):
        raise ValueError("Invalid development label availability date")
    status = row["label_status"]
    if status not in STATUSES:
        raise ValueError("Unknown label status")
    for name in ("origin_price", "target_price"):
        _finite(row[name], name, nullable=status != "available")
    for name in ("origin_vintage_start", "target_vintage_start"):
        if row[name] is not None and _date(row[name], name) >= available:
            raise ValueError("Label endpoint vintage was unavailable at its event")
    if status == "available":
        if target >= available or forecast >= available:
            raise ValueError("Label is available before both endpoint dates have passed")
        if type(row["label"]) is not int or row["label"] not in (0, 1):
            raise ValueError("Available labels must be binary integers")
        if row["label"] != int(row["target_price"] > row["origin_price"]):
            raise ValueError("Label does not match its endpoint prices")
        for name, observed in (("origin_vintage_start", forecast), ("target_vintage_start", target)):
            if row[name] is None or _date(row[name], name) < observed:
                raise ValueError("Available label uses a quarantined endpoint")
    elif row["label"] is not None:
        raise ValueError("Unavailable label must be null")
    if event:
        if "available_until" not in row:
            raise ValueError("Missing label interval end")
        if row["available_until"] is not None:
            until = _date(row["available_until"], "available_until")
            if until <= available:
                raise ValueError("Label interval is empty or reversed")
            if until >= HOLDOUT_START:
                row["available_until"] = None
    elif status != "available":
        raise ValueError("Scoring labels must be available")


def _verify_artifacts(dataset_dir: Path, manifest: dict) -> dict:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != set(ARTIFACTS):
        raise ValueError("Dataset must contain exactly the three expected source artifacts")
    verified = {}
    for name in ARTIFACTS:
        checksum, count = hashlib.sha256(), 0
        with (dataset_dir / name).open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                checksum.update(block)
                count += len(block)
        expected = artifacts[name]
        if checksum.hexdigest() != expected.get("sha256") or count != expected.get("bytes"):
            raise ValueError("Dataset checksum mismatch: " + name)
        verified[name] = {"sha256": checksum.hexdigest(), "bytes": count}
    return verified


def _validate_config(config) -> None:
    expected = {
        "schema_version": 1,
        "calendar_id": CALENDAR_ID,
        "horizon_business_days": 5,
        "forecast_time": "18:00",
        "forecast_timezone": "America/New_York",
        "information_cutoff": "preceding calendar date, inclusive",
    }
    if not isinstance(config, dict):
        raise ValueError("Missing source dataset configuration")
    for key, value in expected.items():
        if type(config.get(key)) is not type(value) or config[key] != value:
            raise ValueError("Incompatible source dataset configuration: " + key)
    feature_sets = config.get("feature_sets")
    if not isinstance(feature_sets, dict) or feature_sets.get("wti") != list(WTI_FEATURE_COLUMNS):
        raise ValueError("Incompatible source dataset configuration: feature_sets.wti")


def load_development_data(dataset_dir: Path) -> tuple[dict, list[dict], list[dict], list[dict]]:
    """Return source identity, WTI features, event history and frozen scoring labels.

    Boundary-crossing feature rows are retained for the runner's explicit purge.
    No held-out JSON row is decoded. Source hashes cover held-out bytes without
    interpreting their contents and are used only for integrity verification.
    """
    manifest_bytes = (dataset_dir / "manifest.json").read_bytes()
    manifest = _decode(manifest_bytes)
    _validate_config(manifest.get("config"))
    artifacts = _verify_artifacts(dataset_dir, manifest)
    maximum = _date(manifest["config"]["maximum_supported_as_of_date"], "maximum_supported_as_of_date")
    if maximum < SCORING_AS_OF:
        raise ValueError("Dataset snapshot does not cover the locked development scoring cutoff")
    identity = {
        "dataset_id": manifest["dataset_id"],
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "source_manifest_sha256": manifest["source_manifest_sha256"],
        "artifacts": artifacts,
        "maximum_as_of_date": SCORING_AS_OF.isoformat(),
    }
    features, by_id = [], {}
    previous = None
    for row in _gated_records(dataset_dir / "features.jsonl"):
        forecast, _ = _validate_identity(row)
        if previous is not None and forecast <= previous:
            raise ValueError("Feature rows must have unique dates in chronological order")
        previous = forecast
        if _date(row.get("information_cutoff_date"), "information_cutoff_date") != forecast - timedelta(days=1):
            raise ValueError("Feature cutoff must be the preceding calendar date")
        expected_timestamp = datetime.combine(
            forecast, time(18), tzinfo=ZoneInfo("America/New_York")).isoformat()
        if row.get("forecast_timestamp") != expected_timestamp:
            raise ValueError("Forecast timestamp must be 18:00 America/New_York on its forecast date")
        values = row.get("features")
        if not isinstance(values, dict) or not all(column in values for column in WTI_FEATURE_COLUMNS):
            raise ValueError("Missing WTI feature columns")
        for column in WTI_FEATURE_COLUMNS:
            _finite(values[column], column)
        row["features"] = {column: values[column] for column in WTI_FEATURE_COLUMNS}
        eligibility = row.get("eligibility")
        if not isinstance(eligibility, dict) or type(eligibility.get("wti")) is not bool:
            raise ValueError("Missing WTI eligibility flag")
        if eligibility["wti"] != (values["wti_latest_price"] is not None):
            raise ValueError("WTI eligibility does not match available price")
        features.append(row)
        by_id[row["row_id"]] = row
    if not features:
        raise ValueError("Dataset contains no admissible development features")

    events, histories, previous = [], defaultdict(list), None
    for row in _gated_records(dataset_dir / "label_events.jsonl", labels=True):
        if row.get("row_id") not in by_id:
            continue
        _validate_label(row, by_id[row["row_id"]], event=True)
        key = (row["forecast_date"], row["available_from"])
        if previous is not None and key <= previous:
            raise ValueError("Label events must be unique and chronologically grouped")
        previous = key
        histories[row["row_id"]].append(row)
        events.append(row)
    for feature in features:
        history = histories[feature["row_id"]]
        if not history or history[0]["available_from"] != feature["forecast_date"]:
            raise ValueError("Label history must start at its forecast date")
        for left, right in zip(history, history[1:]):
            if left["available_until"] != right["available_from"]:
                raise ValueError("Label history contains a gap or overlap")
        if history[-1]["available_until"] is not None:
            raise ValueError("Label history must remain open at the development cutoff")

    scoring, seen, previous = [], set(), None
    for row in _gated_records(dataset_dir / "evaluation_labels.jsonl", labels=True):
        if row.get("row_id") not in by_id:
            continue
        _validate_label(row, by_id[row["row_id"]], event=False)
        if row["row_id"] in seen or (previous is not None and row["forecast_date"] <= previous):
            raise ValueError("Scoring labels must be unique and chronologically ordered")
        previous = row["forecast_date"]
        seen.add(row["row_id"])
        first = next((event for event in histories[row["row_id"]]
                      if event["label_status"] == "available"), None)
        if first is None or any(row[key] != first[key] for key in LABEL_FIELDS):
            raise ValueError("Scoring label differs from its first available event")
        scoring.append(row)
    expected = {row_id for row_id, history in histories.items()
                if any(event["label_status"] == "available" for event in history)}
    if seen != expected:
        raise ValueError("Missing first-available scoring labels")
    return identity, features, events, scoring

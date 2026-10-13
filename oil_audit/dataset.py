"""Build an immutable, offline dataset from verified full ALFRED archives."""

import argparse
from collections import Counter, defaultdict
from datetime import date, datetime, time, timedelta
import hashlib
import json
import math
from pathlib import Path
import statistics
from zoneinfo import ZoneInfo

from .availability import PointInTimeSeries
from .calendar import CALENDAR_ID, CALENDAR_NOTES, add_business_days, next_business_day
from .capture import CAPTURE_POLICY, capture_bounds
from .features import FEATURE_COLUMNS, FEATURE_SETS, build_features
from .labels import build_label_events, select_label
from .training import training_rows
from . import vintages


def json_bytes(value) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def jsonl_bytes(rows) -> bytes:
    return "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n"
                   for row in rows).encode()


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def forecast_schedule(vintage_dates: list[date], start: date) -> list[tuple[date, list[date]]]:
    """One issuance per unique next-business-day WTI update opportunity."""
    schedule = defaultdict(list)
    for vintage in sorted(set(vintage_dates)):
        if vintage >= start:
            schedule[next_business_day(vintage)].append(vintage)
    return sorted(schedule.items())


def construct_rows(wti: PointInTimeSeries, brent: PointInTimeSeries,
                   vintage_dates: list[date], start: date, maximum_as_of: date | None = None):
    feature_rows, events, evaluation = [], [], []
    for forecast, triggers in forecast_schedule(vintage_dates, start):
        if maximum_as_of is not None and forecast > maximum_as_of:
            continue
        target = add_business_days(forecast, 5)
        row_id = "wti-direction-" + forecast.strftime("%Y%m%d")
        features, provenance, eligibility = build_features(wti, brent, forecast)
        feature_rows.append({
            "row_id": row_id,
            "forecast_date": forecast.isoformat(),
            "forecast_timestamp": datetime.combine(
                forecast, time(18), tzinfo=ZoneInfo("America/New_York")).isoformat(),
            "information_cutoff_date": (forecast - timedelta(days=1)).isoformat(),
            "trigger_wti_vintage_dates": [day.isoformat() for day in triggers],
            "target_date": target.isoformat(),
            "features": features, "provenance": provenance, "eligibility": eligibility,
        })
        label_history = build_label_events(wti, forecast, target)
        if maximum_as_of is not None:
            label_history = [event for event in label_history
                             if event["available_from"] <= maximum_as_of.isoformat()]
            # Open only within this bounded export, not a future-availability claim.
            label_history[-1]["available_until"] = None
        for event in label_history:
            events.append({"row_id": row_id, **event})
        first = next((event for event in label_history if event["label_status"] == "available"), None)
        if first is not None:
            # Frozen first-jointly-available outcomes are for evaluation only.
            # Training always reconstructs active (possibly withdrawn) labels.
            evaluation.append({"row_id": row_id, **{
                key: value for key, value in first.items() if key != "available_until"
            }})
    return feature_rows, events, evaluation


def _distribution(values) -> dict:
    values = sorted(value for value in values if value is not None)
    if not values:
        return {"count": 0}
    return {"count": len(values), "min": min(values), "median": statistics.median(values),
            "p95_nearest_rank": values[math.ceil(0.95 * len(values)) - 1], "max": max(values)}


def summarize(feature_rows, events, evaluation, maximum_as_of: date) -> dict:
    histories = defaultdict(list)
    for event in events:
        histories[event["row_id"]].append(event)
    current = [select_label(histories[row["row_id"]], maximum_as_of) for row in feature_rows]
    first_by_id = {row["row_id"]: row for row in evaluation}
    pending = sum(row["target_date"] >= maximum_as_of.isoformat() for row in feature_rows)
    return {
        "forecast_rows": len(feature_rows),
        "forecast_date_range": [feature_rows[0]["forecast_date"], feature_rows[-1]["forecast_date"]],
        "target_date_range": [feature_rows[0]["target_date"], feature_rows[-1]["target_date"]],
        "label_event_rows": len(events),
        "first_available_evaluation_labels": len(evaluation),
        "latest_label_status_counts": dict(Counter(event["label_status"] for event in current)),
        "latest_available_direction_counts": dict(Counter(
            "up" if event["label"] == 1 else "not_up" for event in current
            if event["label_status"] == "available")),
        "first_available_direction_counts": dict(Counter(
            "up" if row["label"] == 1 else "not_up" for row in evaluation)),
        "first_available_ties": sum(row["target_price"] == row["origin_price"] for row in evaluation),
        "target_not_yet_before_replay_end": pending,
        "previously_available_now_unavailable": sum(
            row["row_id"] in first_by_id and event["label_status"] != "available"
            for row, event in zip(feature_rows, current)),
        "direction_changed_between_available_versions": sum(
            len({event["label"] for event in history if event["label_status"] == "available"}) > 1
            for history in histories.values()),
        "feature_missing_counts": {column: sum(row["features"][column] is None for row in feature_rows)
                                   for column in FEATURE_COLUMNS},
        "eligibility_counts": {key: sum(row["eligibility"][key] for row in feature_rows)
                               for key in feature_rows[0]["eligibility"]},
        "data_age_calendar_days": {key: _distribution(row["features"][key] for row in feature_rows)
                                   for key in ("wti_age_calendar_days", "brent_age_calendar_days",
                                               "spread_age_calendar_days")},
        "training_rows_at_replay_end": {name: len(training_rows(
            feature_rows, events, maximum_as_of, maximum_as_of, name)) for name in FEATURE_SETS},
    }


def build(data_dir: Path = Path("data/audit/vintages"),
          output_root: Path = Path("data/datasets"), start: date = date(2011, 1, 1),
          report: Path = Path("reports/dataset_summary.json")) -> dict:
    vintages.verify_manifest(data_dir)
    source_manifest = (data_dir / "manifest.json").read_bytes()
    source_records = json.loads(source_manifest)
    series, listed_dates = {}, {}
    for series_id in vintages.SERIES:
        listed, _ = vintages.form_metadata(series_id, data_dir)
        listed_dates[series_id] = sorted(date.fromisoformat(day) for day in listed)
        series[series_id] = PointInTimeSeries(series_id, vintages.records(series_id, data_dir))
        starts = {version.vintage_start for versions in series[series_id].versions_by_observation.values()
                  for version in versions}
        if not starts.issubset(set(listed_dates[series_id])):
            raise ValueError("Interval starts missing from the captured vintage-date list")
        if min(starts) != min(listed_dates[series_id]) or max(starts) != max(listed_dates[series_id]):
            raise ValueError("Captured interval coverage differs from listed vintage bounds")
    wti, brent = (series[series_id] for series_id in vintages.SERIES)
    bounds = capture_bounds(source_records, vintages.SERIES)
    schedule = forecast_schedule(listed_dates[wti.series_id], start)
    safe_dates = [forecast for forecast, _ in schedule if all(
        datetime.combine(forecast, time(18), tzinfo=ZoneInfo("America/New_York")) <= bound
        for bound in bounds.values())]
    if not safe_dates:
        raise ValueError("No forecast opportunities in the requested period")
    maximum_as_of = max(safe_dates)
    feature_rows, events, evaluation = construct_rows(
        wti, brent, listed_dates[wti.series_id], start, maximum_as_of)
    config = {
        "schema_version": 1, "requested_start_vintage_date": start.isoformat(),
        "forecast_schedule": "next US business day after each WTI listed vintage; collisions merged",
        "forecast_time": "18:00", "forecast_timezone": "America/New_York",
        "information_cutoff": "preceding calendar date, inclusive",
        "calendar_id": CALENDAR_ID, "calendar_notes": CALENDAR_NOTES,
        "horizon_business_days": 5,
        "target": "WTI price on target date > WTI price on forecast date; ties map to 0",
        "training_labels": "active endpoint versions as of preceding calendar date; withdrawals honored",
        "evaluation_labels": "frozen first jointly available numeric endpoint pair; evaluation only",
        "maximum_supported_as_of_date": maximum_as_of.isoformat(),
        "source_capture_policy": CAPTURE_POLICY,
        "capture_lower_bounds_utc": {key: value.isoformat() for key, value in bounds.items()},
        "feature_sets": FEATURE_SETS,
    }
    source_code = {str(path): sha256(path.read_bytes()) for path in sorted(Path("oil_audit").glob("*.py"))}
    identity = {"config": config, "source_manifest_sha256": sha256(source_manifest),
                "source_code_sha256": source_code}
    dataset_id = sha256(json_bytes(identity))[:20]
    destination = output_root / dataset_id
    outputs = {"features.jsonl": jsonl_bytes(feature_rows), "label_events.jsonl": jsonl_bytes(events),
               "evaluation_labels.jsonl": jsonl_bytes(evaluation)}
    summary = summarize(feature_rows, events, evaluation, maximum_as_of)
    summary["issuances_excluded_by_capture_bound"] = len(schedule) - len(safe_dates)
    summary["series_checks"] = {
        series_id: {
            "first_listed_vintage": min(listed_dates[series_id]).isoformat(),
            "last_listed_vintage": max(listed_dates[series_id]).isoformat(),
            "listed_vintages": len(listed_dates[series_id]),
            "quarantined_versions": sum(version.quarantined for versions in item.versions_by_observation.values()
                                        for version in versions),
        } for series_id, item in series.items()
    }
    manifest = {"dataset_id": dataset_id, **identity, "sources": source_records,
                "artifacts": {name: {"sha256": sha256(payload), "bytes": len(payload)}
                              for name, payload in outputs.items()}, "summary": summary}
    outputs["manifest.json"] = json_bytes(manifest)
    destination.mkdir(parents=True, exist_ok=True)
    for name, payload in outputs.items():
        path = destination / name
        if path.exists():
            if path.read_bytes() != payload:
                raise ValueError("Refusing to replace a different dataset artifact: " + str(path))
        else:
            path.write_bytes(payload)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_bytes(json_bytes({"dataset_id": dataset_id, "dataset_directory": str(destination),
                                  "config": config, **summary}))
    return {"dataset_directory": str(destination), **summary}


def read_dataset(dataset_dir: Path):
    manifest = json.loads((dataset_dir / "manifest.json").read_text())
    result = {}
    for name, expected in manifest["artifacts"].items():
        payload = (dataset_dir / name).read_bytes()
        if sha256(payload) != expected["sha256"] or len(payload) != expected["bytes"]:
            raise ValueError("Dataset checksum mismatch: " + name)
        result[name] = [json.loads(line) for line in payload.splitlines()]
    return manifest, result


def workspace_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise argparse.ArgumentTypeError("Use a workspace-relative path without parent traversal")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build_parser = subparsers.add_parser("build", help="Build offline from a saved full vintage snapshot")
    build_parser.add_argument("--data-dir", type=workspace_path, default=Path("data/audit/vintages"))
    build_parser.add_argument("--output-root", type=workspace_path, default=Path("data/datasets"))
    build_parser.add_argument("--start", type=date.fromisoformat, default=date(2011, 1, 1))
    build_parser.add_argument("--report", type=workspace_path, default=Path("reports/dataset_summary.json"))
    view_parser = subparsers.add_parser("training-view", help="Export a historical training view; fits no model")
    view_parser.add_argument("--dataset-dir", type=workspace_path, required=True)
    view_parser.add_argument("--as-of", type=date.fromisoformat, required=True)
    view_parser.add_argument("--feature-set", choices=FEATURE_SETS, default="wti")
    view_parser.add_argument("--output", type=workspace_path, required=True)
    args = parser.parse_args()
    if args.command == "build":
        print(json.dumps(build(args.data_dir, args.output_root, args.start, args.report), indent=2))
    else:
        manifest, data = read_dataset(args.dataset_dir)
        rows = training_rows(data["features.jsonl"], data["label_events.jsonl"], args.as_of,
                             date.fromisoformat(manifest["config"]["maximum_supported_as_of_date"]),
                             args.feature_set)
        if args.output.exists():
            raise ValueError("Choose a new output path to preserve existing training views")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(jsonl_bytes(rows))
        print(json.dumps({"rows": len(rows), "as_of": args.as_of.isoformat(),
                          "feature_set": args.feature_set, "output": str(args.output)}))


if __name__ == "__main__":
    main()

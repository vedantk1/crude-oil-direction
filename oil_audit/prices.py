"""Audit EIA daily observations without imputing, deleting, or training."""

import hashlib
import json
from datetime import date, timedelta
from math import isfinite
from pathlib import Path

import xlrd


EXPECTED_KEYS = {"wti": "RWTC", "brent": "RBRTE"}


def read_eia(path: Path, expected_key: str) -> list[tuple[date, float | None]]:
    workbook = xlrd.open_workbook(path)
    sheet = workbook.sheet_by_name("Data 1")
    if sheet.cell_value(1, 1) != expected_key or sheet.cell_value(2, 0) != "Date":
        raise ValueError("Unexpected EIA series or workbook schema")
    if "Dollars per Barrel" not in sheet.cell_value(2, 1):
        raise ValueError("Unexpected EIA units")
    rows = []
    for index in range(3, sheet.nrows):
        cell = sheet.cell(index, 0)
        if cell.ctype != xlrd.XL_CELL_DATE:
            raise ValueError(f"Invalid observation date in row {index + 1}")
        observed_on = xlrd.xldate_as_datetime(cell.value, workbook.datemode).date()
        value = sheet.cell(index, 1)
        if value.ctype in {xlrd.XL_CELL_EMPTY, xlrd.XL_CELL_BLANK}:
            price = None
        elif value.ctype == xlrd.XL_CELL_NUMBER:
            price = float(value.value)
        else:
            raise ValueError(f"Unexpected price marker in row {index + 1}: {value.value!r}")
        rows.append((observed_on, price))
    return rows


def weekdays(start: date, end: date) -> list[date]:
    result = []
    day = start
    while day <= end:
        if day.weekday() < 5:
            result.append(day)
        day += timedelta(days=1)
    return result


def audit_series(rows: list[tuple[date, float | None]]) -> dict:
    if not rows:
        raise ValueError("Empty source")
    dates = [d for d, _ in rows]
    if len(set(dates)) != len(dates):
        raise ValueError("Duplicate observation dates; do not silently select a value")
    if dates != sorted(dates):
        raise ValueError("Observation dates are not increasing")
    valid = [(d, value) for d, value in rows if value is not None]
    if not valid:
        raise ValueError("No numeric observations")
    if any(not isfinite(value) for _, value in valid):
        raise ValueError("Non-finite price")
    values = dict(valid)
    first, last = valid[0][0], valid[-1][0]
    missing = [d for d in weekdays(first, last) if d not in values]
    changes = [
        {"previous_date": d0.isoformat(), "date": d1.isoformat(),
         "previous_price": p0, "price": p1, "change_usd": round(p1 - p0, 2),
         "calendar_days_between": (d1 - d0).days}
        for (d0, p0), (d1, p1) in zip(valid, valid[1:])
    ]
    longest = max(changes, key=lambda row: row["calendar_days_between"], default=None)
    yearly = []
    for year in range(first.year, last.year + 1):
        lower, upper = max(first, date(year, 1, 1)), min(last, date(year, 12, 31))
        yearly.append({"year": year, "observations": sum(d.year == year for d in values),
                       "weekdays_without_price": sum(d.year == year for d in missing),
                       "weekdays_in_bounded_period": len(weekdays(lower, upper))})
    return {
        "first_observation": first.isoformat(), "last_observation": last.isoformat(),
        "source_rows": len(rows), "numeric_observations": len(valid),
        "explicit_empty_price_rows": len(rows) - len(valid),
        "duplicate_dates": 0, "dates_increasing": True,
        "weekend_observations": sum(d.weekday() >= 5 for d in values),
        "weekday_grid_size": len(weekdays(first, last)),
        "weekdays_without_price_count": len(missing),
        "missing_date_interpretation": "Weekday grid only; includes holidays and unclassified omissions. Not a missing-trading-day count.",
        "minimum": {"date": min(valid, key=lambda row: row[1])[0].isoformat(), "value": min(values.values())},
        "maximum": {"date": max(valid, key=lambda row: row[1])[0].isoformat(), "value": max(values.values())},
        "nonpositive_observations": [{"date": d.isoformat(), "value": p} for d, p in valid if p <= 0],
        "unchanged_adjacent_observations": sum(row["change_usd"] == 0 for row in changes),
        "largest_absolute_adjacent_changes": sorted(changes, key=lambda row: abs(row["change_usd"]), reverse=True)[:10],
        "longest_calendar_gap": longest,
        "yearly_coverage": yearly,
        "weekdays_without_price": [d.isoformat() for d in missing],
    }


def compare_calendars(series: dict[str, list[tuple[date, float | None]]]) -> dict:
    valid = {key: {d: p for d, p in rows if p is not None} for key, rows in series.items()}
    first = max(min(prices) for prices in valid.values())
    last = min(max(prices) for prices in valid.values())
    wti = {d for d in valid["wti"] if first <= d <= last}
    brent = {d for d in valid["brent"] if first <= d <= last}
    missing_both = set(weekdays(first, last)) - wti - brent
    return {
        "start": first.isoformat(), "end": last.isoformat(),
        "both_have_price": len(wti & brent),
        "wti_only_count": len(wti - brent), "brent_only_count": len(brent - wti),
        "wti_only_dates": sorted(d.isoformat() for d in wti - brent),
        "brent_only_dates": sorted(d.isoformat() for d in brent - wti),
        "neither_has_price_weekdays": len(missing_both),
        "join_rule": "Keep WTI target calendar. Brent features require separate availability alignment; do not inner-join first and redefine the target horizon.",
    }


def run_audit(manifest_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text())
    if not manifest.get("complete"):
        raise ValueError("Source snapshot is incomplete")
    series = {}
    provenance = []
    for source in manifest["sources"]:
        if source["source_id"] not in EXPECTED_KEYS:
            continue
        path = manifest_path.parent / source["file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != source["sha256"]:
            raise ValueError(f"Source checksum mismatch: {source['source_id']}")
        series[source["source_id"]] = read_eia(path, EXPECTED_KEYS[source["source_id"]])
        provenance.append({key: source[key] for key in ("source_id", "requested_url", "retrieved_at_utc", "sha256", "bytes")})
    if set(series) != set(EXPECTED_KEYS):
        raise ValueError("Both price series are required")
    return {
        "snapshot_manifest": manifest_path.as_posix(),
        "scope": "Current EIA download quality only. Observation dates are not availability dates. This artifact does not reconstruct vintages or train models.",
        "source_provenance": provenance,
        "series": {key: audit_series(rows) for key, rows in series.items()},
        "calendar_comparison": compare_calendars(series),
    }

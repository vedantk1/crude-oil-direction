"""Audit public ALFRED archives; run from the workspace root with Python 3.

No credentials or third-party packages are needed. Default execution is offline.
Use --download to retrieve sources into a new timestamped subdirectory.
"""

import argparse
import collections
import csv
import datetime as dt
import hashlib
import io
import json
from pathlib import Path
import re
import statistics
import urllib.parse
import urllib.request
import zipfile


ROOT = Path("data/audit/vintages")
SERIES = ("DCOILWTICO", "DCOILBRENTEU")
MAX_VINTAGES_PER_REQUEST = 450
SAMPLE_DATES = ("2011-04-06", "2011-04-13", "2020-04-22", "2020-04-29")


def save_download(filename, payload, url, form_fields=None):
    path = ROOT / filename
    if path.exists():
        raise ValueError("Refusing to replace source evidence: " + str(path))
    path.write_bytes(payload)
    manifest_path = ROOT / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else []
    manifest.append({
        "path": str(path), "source_url": url,
        "sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload),
        "retrieved_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "form_fields": form_fields,
    })
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


def verify_manifest(data_dir=None):
    """Verify saved evidence; never manufacture new retrieval timestamps."""
    root = Path(data_dir) if data_dir is not None else ROOT
    manifest = json.loads((root / "manifest.json").read_text())
    for record in manifest:
        path = Path(record["path"])
        payload = path.read_bytes()
        if len(payload) != record["bytes"] or hashlib.sha256(payload).hexdigest() != record["sha256"]:
            raise ValueError("Source checksum mismatch: " + str(path))
    required = {path for series in SERIES for path in root.glob(series + "-realtime-part*.zip")}
    required |= {root / (series + suffix) for series in SERIES
                 for suffix in ("-sample-vintages.zip", "-download-page.html")}
    if not required.issubset({Path(row["path"]) for row in manifest}):
        raise ValueError("Manifest omits required input files")


def form_url(series):
    return "https://alfred.stlouisfed.org/series/downloaddata?seid=" + series


def form_metadata(series, data_dir=None):
    root = Path(data_dir) if data_dir is not None else ROOT
    html = (root / (series + "-download-page.html")).read_text()
    selector = re.search(
        r'<select id="form_selected_vintage_dates".*?</select>', html, re.S
    ).group()
    dates = re.findall(r'<option value="([0-9-]+)"', selector)
    limits = [re.search(r'id="form_obs_' + key +
                       r'_date"[^>]+value="([^"]+)"', html).group(1)
              for key in ("start", "end")]
    return dates, limits


def request_zip(series, dates, output_type, filename):
    listed, limits = form_metadata(series)
    fields = [("form[units]", "lin"), ("form[obs_start_date]", limits[0]),
              ("form[obs_end_date]", limits[1]),
              ("form[file_type]", str(output_type)),
              ("form[file_format]", "csv"), ("form[download_data]", "")]
    fields += [("form[selected_vintage_dates][]", date)
               for date in dates if date in listed]
    other_dates = [date for date in dates if date not in listed]
    if other_dates:
        fields.append(("form[entered_vintage_dates]", " ".join(other_dates)))
    request = urllib.request.Request(
        form_url(series), urllib.parse.urlencode(fields).encode())
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = response.read()
    if not zipfile.is_zipfile(io.BytesIO(payload)):
        raise ValueError("ALFRED returned no ZIP for " + series)
    save_download(filename, payload, form_url(series), fields)


def download():
    ROOT.mkdir(parents=True, exist_ok=True)
    for series in SERIES:
        with urllib.request.urlopen(form_url(series), timeout=60) as response:
            save_download(series + "-download-page.html", response.read(), form_url(series))
        dates, _ = form_metadata(series)
        for start in range(0, len(dates), MAX_VINTAGES_PER_REQUEST):
            part = start // MAX_VINTAGES_PER_REQUEST + 1
            request_zip(series, dates[start:start + MAX_VINTAGES_PER_REQUEST],
                        1, series + "-realtime-part" + str(part) + ".zip")
        request_zip(series, SAMPLE_DATES, 2, series + "-sample-vintages.zip")


def zip_rows(path):
    with zipfile.ZipFile(path) as archive:
        csv_name = next(name for name in archive.namelist() if name.endswith(".csv"))
        return list(csv.DictReader(io.StringIO(archive.read(csv_name).decode())))


def records(series, data_dir=None):
    # The batch downloads overlap in observation dates: deduplicate whole records.
    root = Path(data_dir) if data_dir is not None else ROOT
    unique = {}
    for path in sorted(root.glob(series + "-realtime-part*.zip")):
        for row in zip_rows(path):
            unique[tuple(row.items())] = row
    return sorted(unique.values(), key=lambda row: (
        row["period_start_date"], row["realtime_start_date"]))


def snapshot(rows, date, strict=True):
    """Get ALFRED's state on a date, optionally rejecting future observations.

    This is a date-level archive lookup, not a claim about intraday availability.
    Missing values are deliberately retained; an old value must not survive a
    later deletion merely because it was once present.
    """
    result = {}
    for row in rows:
        if row["realtime_start_date"] <= date and (
                not row["realtime_end_date"] or date <= row["realtime_end_date"]):
            if strict and (row["period_start_date"] >= date or
                           row["realtime_start_date"] < row["period_start_date"]):
                continue
            if row["period_start_date"] in result:
                raise ValueError("Overlapping intervals")
            result[row["period_start_date"]] = row
    return result


def days_between(start, end):
    return (dt.date.fromisoformat(end) - dt.date.fromisoformat(start)).days


def audit(series):
    rows = records(series)
    vintage_dates, limits = form_metadata(series)
    groups = collections.defaultdict(list)
    for row in rows:
        groups[row["period_start_date"]].append(row)
    for versions in groups.values():
        for previous, current in zip(versions, versions[1:]):
            end = previous["realtime_end_date"]
            if not end or end >= current["realtime_start_date"]:
                raise ValueError("Overlapping archive intervals")
            if days_between(end, current["realtime_start_date"]) != 1:
                raise ValueError("Gap in archive intervals")
    initial = []
    for observation_date, versions in groups.items():
        numeric = [row for row in versions if row[series]]
        if numeric:
            initial.append(numeric[0])
    lags = [(days_between(row["period_start_date"], row["realtime_start_date"]), row)
            for row in initial if row["realtime_start_date"] > vintage_dates[0]]
    anomalies = [row for row in rows if row[series] and
                 row["realtime_start_date"] <= row["period_start_date"]]
    invalid = [row for row in anomalies
               if row["realtime_start_date"] < row["period_start_date"]]
    transitions = []
    for observation_date, versions in groups.items():
        for previous, current in zip(versions, versions[1:]):
            if previous[series] == current[series]:
                continue
            kind = ("numeric_to_numeric" if previous[series] and current[series]
                    else "numeric_to_missing" if previous[series]
                    else "missing_to_numeric")
            transitions.append({"observation_date": observation_date,
                                "vintage_date": current["realtime_start_date"],
                                "before": previous[series], "after": current[series],
                                "kind": kind})
    comparisons = []
    for column in [key for key in zip_rows(ROOT / (series + "-sample-vintages.zip"))[0]
                   if key != "observation_date"]:
        date = dt.datetime.strptime(column.rsplit("_", 1)[1], "%Y%m%d").date().isoformat()
        state = snapshot(rows, date, strict=False)
        for sample in zip_rows(ROOT / (series + "-sample-vintages.zip")):
            observed = state.get(sample["observation_date"], {}).get(series, "")
            assert observed == sample[column], (series, date, sample["observation_date"])
        present = sorted(day for day, row in state.items() if row[series])
        comparisons.append({"vintage_date": date, "nonmissing_count": len(present),
                            "latest_observation": present[-1],
                            "latest_value": state[present[-1]][series]})
    # Specific integrity checks on actual archive quirks.
    assert "2012-12-27" not in snapshot(rows, "2012-12-26", strict=True)
    assert not snapshot(rows, "2011-04-05", strict=True)
    for row in invalid:
        assert row["period_start_date"] not in snapshot(
            rows, row["realtime_start_date"], strict=True)
        later = (dt.date.fromisoformat(row["period_start_date"]) + dt.timedelta(days=1)).isoformat()
        assert row["period_start_date"] not in snapshot(rows, later, strict=True)
    subsets = {}
    for lower in (limits[0], "2000-01-01", "2011-04-06", "2012-01-01", "2015-01-01"):
        selected = {day: values for day, values in groups.items() if day >= lower}
        subsets[lower] = {
            "current_nonmissing": sum(any(row[series] and not row["realtime_end_date"]
                                           for row in values) for values in selected.values()),
            "distinct_dates_with_multiple_numeric_values": sum(
                len({row[series] for row in values if row[series]}) > 1
                for values in selected.values()),
            "numeric_to_missing_on_2026_09_16": sum(
                change["vintage_date"] == "2026-09-16" and
                change["kind"] == "numeric_to_missing" and
                change["observation_date"] >= lower for change in transitions),
        }
    return {
        "series_id": series, "first_vintage": vintage_dates[0],
        "last_vintage": vintage_dates[-1], "listed_vintage_count": len(vintage_dates),
        "observation_limits": limits, "unique_realtime_records": len(rows),
        "ever_numeric_observation_count": len(initial),
        "date_level_reconstruction_verified_against_samples": comparisons,
        "first_numeric_lag_after_initial_vintage": {
            "count": len(lags), "unit": "calendar days",
            "minimum": min(lag for lag, _ in lags),
            "median": statistics.median(lag for lag, _ in lags),
            "maximum": max(lag for lag, _ in lags),
            "distribution": dict(sorted(collections.Counter(lag for lag, _ in lags).items())),
            "longest_examples": [{"lag": lag, **row} for lag, row in
                                 sorted(lags, key=lambda item: item[0], reverse=True)[:10]],
            "caution": "Includes subsequently removed numeric holiday records; initial archive load excluded.",
        },
        "same_or_future_observation_date_records": anomalies,
        "impossible_future_observation_date_records": invalid,
        "subsets_by_observation_start": subsets,
        "transition_counts_by_vintage_date": {
            kind: dict(sorted(collections.Counter(change["vintage_date"]
                            for change in transitions if change["kind"] == kind).items()))
            for kind in ("numeric_to_numeric", "numeric_to_missing", "missing_to_numeric")},
        "event_examples": [change for change in transitions
                           if change["observation_date"] in (
                               "1986-01-31", "1987-12-21", "2019-11-29", "2019-12-25",
                               "2020-03-06", "2020-03-18", "2020-04-20", "2024-05-31")],
        "checks_passed": ["Four independent vintage columns match reconstructed records",
                          "No overlapping or gapped intervals across every observation history",
                          "Future observations and incorrectly predated records rejected by strict as-of lookup",
                          "No false reconstruction before initial archive date"],
    }


def main():
    global ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--capture-only", action="store_true",
                        help="With --download, save and hash bytes without decoding observation values")
    parser.add_argument("--data-dir", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=Path("reports/vintage_audit.json"))
    args = parser.parse_args()
    if args.capture_only and not args.download:
        parser.error("--capture-only requires --download")
    if args.data_dir.is_absolute() or args.output.is_absolute():
        parser.error("Use workspace-relative paths")
    ROOT = args.data_dir
    if args.download:
        ROOT = ROOT / dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        download()
    verify_manifest()
    if args.capture_only:
        print("Source evidence:", ROOT.as_posix())
        return
    results = {series: audit(series) for series in SERIES}
    from .report_output import write_audit_report
    write_audit_report("vintage_audit.json", (json.dumps(results, indent=2) + "\n").encode(), args.output)
    print("Source evidence:", ROOT.as_posix())
    for series, result in results.items():
        print(series, result["first_vintage"], result["listed_vintage_count"],
              "vintages; integrity checks passed")


if __name__ == "__main__":
    main()

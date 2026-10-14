"""Cross-check saved EIA prices, ALFRED intervals, and sampled weekly releases."""

import argparse
import csv
from datetime import date
import hashlib
import io
import json
from pathlib import Path

from .prices import EXPECTED_KEYS, read_eia
from .sources import ARCHIVE_ISSUES
from . import vintages


def read_weekly_prices(path: Path, issue_date: date) -> dict:
    """Read the inspected Table 11 CSV layout; reject unfamiliar layouts."""
    rows = list(csv.reader(io.StringIO(path.read_text(encoding="cp1252").replace("\x1a", ""))))
    header_index = next(index for index, row in enumerate(rows)
                        if row[:3] == ["STUB_1", "STUB_2", "STUB_3"]
                        and len(row) == 15 and row[3].startswith("Fri "))
    header = rows[header_index]
    # Columns 3 and 4 are weekly averages. Only columns 5 onward are daily.
    days = []
    for label in header[5:]:
        month, day = map(int, label.split()[1].split("/"))
        year = issue_date.year - (month > issue_date.month)
        observed = date(year, month, day)
        if observed >= issue_date or observed.strftime("%a") != label.split()[0]:
            raise ValueError("Archive issue and daily header disagree")
        days.append(observed.isoformat())
    output = {}
    for row in rows[header_index + 1:]:
        if row and row[0] == "STUB_1":
            break
        if len(row) != 15 or row[0] != "Crude Oil":
            continue
        series = {"WTI - Cushing": "wti", "Brent": "brent"}.get(row[1].strip())
        if not series:
            continue
        output[series] = {day: None if value.strip() in {"", "–", "NA"} else float(value)
                          for day, value in zip(days, row[5:])}
    if set(output) != {"wti", "brent"}:
        raise ValueError("Missing crude series in archive")
    return output


def reconcile(manifest_path: Path, release_dir: Path) -> dict:
    manifest = json.loads(manifest_path.read_text())
    eia = {}
    for source in manifest["sources"]:
        key = source["source_id"]
        if key in EXPECTED_KEYS:
            path = manifest_path.parent / source["file"]
            if hashlib.sha256(path.read_bytes()).hexdigest() != source["sha256"]:
                raise ValueError("EIA checksum mismatch")
            eia[key] = {day.isoformat(): value for day, value in read_eia(path, EXPECTED_KEYS[key]) if value is not None}
    vintages.verify_manifest()
    comparisons = {}
    for key, series_id in (("wti", "DCOILWTICO"), ("brent", "DCOILBRENTEU")):
        rows = vintages.records(series_id)
        current = {row["period_start_date"]: float(row[series_id]) for row in rows
                   if not row["realtime_end_date"] and row[series_id]}
        mismatches = [{"date": day, "eia": eia[key].get(day), "alfred": current.get(day)}
                      for day in sorted(set(eia[key]) | set(current))
                      if eia[key].get(day) != current.get(day)]
        comparisons[key] = {"eia_count": len(eia[key]), "alfred_count": len(current),
                            "mismatch_count": len(mismatches), "mismatches": mismatches}
    release_manifest = json.loads((release_dir / "manifest.json").read_text())
    if "sources" in release_manifest and not release_manifest.get("complete"):
        raise ValueError("Incomplete archived release download")
    release_manifest = release_manifest.get("files", release_manifest.get("sources"))
    for record in release_manifest:
        path = Path(record.get("path", record.get("file", "")))
        if not path.exists():
            path = release_dir / path
        if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError("Archived release checksum mismatch")
    archives = []
    required = [release_dir / ("table11-" + issue + ".csv") for issue in ARCHIVE_ISSUES]
    for path in required:
        issue_date = date.fromisoformat(path.stem.removeprefix("table11-"))
        daily = read_weekly_prices(path, issue_date)
        for key, values in daily.items():
            mismatches = [{"observation_date": day, "archived_price": value,
                           "current_price": eia[key].get(day)}
                          for day, value in values.items() if value != eia[key].get(day)]
            archives.append({"issue_date": issue_date.isoformat(), "series": key,
                             "daily_columns": len(values),
                             "numeric_prices": sum(value is not None for value in values.values()),
                             "first_observation": min(values), "last_observation": max(values),
                             "mismatches": mismatches})
    return {"current_eia_vs_alfred": comparisons, "sampled_weekly_releases": archives,
            "scope": "Current entire series compared; only four EIA weekly issues sampled. No intraday availability proof."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--vintage-dir", type=Path, default=Path("data/audit/vintages"))
    parser.add_argument("--release-dir", type=Path, default=Path("data/audit/releases"))
    parser.add_argument("--output", type=Path, default=Path("reports/source_reconciliation.json"))
    args = parser.parse_args()
    if any(path.is_absolute() for path in (args.manifest, args.output, args.vintage_dir, args.release_dir)):
        parser.error("Use workspace-relative paths")
    vintages.ROOT = args.vintage_dir
    result = reconcile(args.manifest, args.release_dir)
    from .report_output import write_audit_report
    write_audit_report("source_reconciliation.json", (json.dumps(result, indent=2) + "\n").encode(), args.output)
    print(args.output.as_posix())


if __name__ == "__main__":
    main()

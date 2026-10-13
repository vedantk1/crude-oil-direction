"""Generate invented source files and exercise the real offline research workflow.

This is a software demonstration, not a market experiment. All values, archive
content and capture times are synthetic; the EIA URL shapes only exercise the
production source validator and do not claim a download occurred.
"""

import argparse
from bisect import bisect_left
import csv
from datetime import date, timedelta
import io
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

from oil_audit.calendar import is_business_day, next_business_day
from oil_audit.dataset import json_bytes, sha256, workspace_path
from oil_audit.inventory_sources import ARCHIVE_INDEX, MONTHS, SUPPLEMENTARY_ISSUES, _canonical_issues, parse_index
from oil_audit.vintages import SERIES
from .workflow import fresh_directory, run as replay_sources


DEMO_NOTICE = "SYNTHETIC: invented values and archive evidence; software verification only, no market meaning."
CAPTURE_BOUND = "2022-12-31T23:59:59+00:00"
MONTH_NAMES = {number: name for name, number in MONTHS.items()}


def _csv(rows):
    stream = io.StringIO(newline="")
    csv.writer(stream, lineterminator="\n").writerows(rows)
    return stream.getvalue().encode()


def _zip(name, payload):
    buffer = io.BytesIO()
    # Fixed metadata avoids embedding local time or user information.
    info = ZipInfo(name, date_time=(2000, 1, 1, 0, 0, 0))
    info.compress_type = ZIP_DEFLATED
    info.external_attr = 0o600 << 16
    with ZipFile(buffer, "w") as archive:
        archive.writestr(info, payload)
    return buffer.getvalue()


def _first_wednesday(year, month):
    start = date(year, month, 1)
    return start + timedelta(days=(2 - start.weekday()) % 7)


def _price_sources(root):
    root.mkdir(parents=True)
    vintages = [date(2011, 4, 6) + timedelta(weeks=index) for index in range(160)]
    vintages += [_first_wednesday(year, 2) for year in range(2017, 2023)]
    vintages += [date(2022, 12, 21)]
    days = []
    observed = date(2011, 3, 1)
    while observed <= vintages[-1]:
        if is_business_day(observed):
            days.append(observed)
        observed += timedelta(days=1)
    manifest = []
    for series_index, series in enumerate(SERIES):
        records = []
        for day in days:
            released = vintages[bisect_left(vintages, day)]
            cents = 4000 + (day.toordinal() * 37) % 2000 + series_index * (500 + day.day)
            records.append([day.isoformat(), released.isoformat(), "", f"{cents / 100:.2f}"])
        numeric_records = list(records)
        records += [["2022-12-30", "2023-01-04", "", "FUTURE_VALUE_MUST_NOT_BE_DECODED"],
                    ["2023-01-03", "2023-01-04", "", "FUTURE_VALUE_MUST_NOT_BE_DECODED"]]
        form = ("<!-- " + DEMO_NOTICE + " -->\n<select id=\"form_selected_vintage_dates\">"
                + "".join(f'<option value="{day.isoformat()}">{day.isoformat()}</option>' for day in vintages)
                + '<option value="2023-01-04">Synthetic future gate sentinel</option>'
                + '</select><input id="form_obs_start_date" value="2011-03-01">'
                + '<input id="form_obs_end_date" value="2022-12-21">').encode()
        table = _csv([["period_start_date", series, "realtime_start_date", "realtime_end_date"],
                      *[[row[0], row[3], row[1], row[2]] for row in records]])
        # The builder authenticates this companion archive even though it reads
        # the interval table; make its single vintage column independently valid.
        sample = _csv([["observation_date", series + "_20221221"], *[[row[0], row[3]] for row in numeric_records]])
        for name, payload in (
            (series + "-download-page.html", form),
            (series + "-realtime-part1.zip", _zip(series + ".csv", table)),
            (series + "-sample-vintages.zip", _zip(series + "-sample.csv", sample)),
        ):
            path = root / name
            path.write_bytes(payload)
            manifest.append({"path": str(path), "source_url": "synthetic://alfred-format/" + name,
                             "sha256": sha256(payload), "bytes": len(payload), "synthetic": True,
                             "retrieved_at_utc": CAPTURE_BOUND, "timestamp_kind": "synthetic_capture_bound",
                             "notice": DEMO_NOTICE, "form_fields": None})
    (root / "manifest.json").write_bytes(json_bytes(manifest))
    return vintages


def _inventory_sources(root, vintages):
    root.mkdir(parents=True)
    released_dates = {_first_wednesday(year, 8 if year == 2011 else 1) for year in range(2011, 2023)}
    for released in [day for day in vintages if day.year >= 2017]:
        released_dates.update(released - timedelta(weeks=offset) for offset in range(5))
    # Mandatory independently validated supplements are added by the canonical
    # issue builder, exactly as in a real archive capture.
    released_dates -= {date.fromisoformat(row["release_date"]) for row in SUPPLEMENTARY_ISSUES}
    rows = []
    for released in sorted(released_dates):
        observed = released - timedelta(days=5)
        slug = released.strftime("%Y_%m_%d")
        rows.append(f'<tr><td>{MONTH_NAMES[released.month]}</td><td><a href="{ARCHIVE_INDEX}'
                    f'{released.year}/{slug}/wpsr_{slug}.php">{released.day}</a></td>'
                    f'<td>{observed.month}/{observed.day}</td></tr>')
    index = ("<!-- " + DEMO_NOTICE + " --><table>" + "".join(rows) + "</table>").encode()

    def capture(name, payload, url):
        (root / name).write_bytes(payload)
        return {"file": name, "requested_url": url, "resolved_url": url, "status": "downloaded",
                "sha256": sha256(payload), "bytes": len(payload), "synthetic": True,
                "capture_method": "generated_fixture_not_downloaded", "notice": DEMO_NOTICE}

    indexed = parse_index(index)
    issues = _canonical_issues(indexed)
    manifest = {
        "schema_version": 1, "complete": True, "all_downloads_succeeded": True,
        "synthetic": True, "notice": DEMO_NOTICE, "release_years": [2011, 2022],
        "index": capture("archive-index.html", index, ARCHIVE_INDEX),
        "indexed_issue_count": len(indexed), "issues": issues,
        "additional_unindexed_issue_ids": [row["issue_id"] for row in issues
                                            if row["issue_id"] not in {item["issue_id"] for item in indexed}],
        "sources": [], "supplementary_issue_pages": [],
    }
    for issue in issues:
        observed = date.fromisoformat(issue["observation_date"])
        previous = observed - timedelta(days=7)
        compare1, compare2 = observed - timedelta(days=364), observed - timedelta(days=728)
        stock_rows = [["STUB_1", observed.strftime("%m/%d/%y"), previous.strftime("%m/%d/%y"),
                       "Difference", compare1.strftime("%m/%d/%y"), "Percent Change",
                       compare2.strftime("%m/%d/%y"), "Percent Change"]]
        for name, base, multiplier in (("Commercial (Excluding SPR)", 400_000, 17), ("Cushing", 20_000, 11)):
            current = base + (observed.toordinal() * multiplier) % 10_000
            prior = base + (previous.toordinal() * multiplier) % 10_000
            stock_rows.append([name, f"{current / 1000:.3f}", f"{prior / 1000:.3f}",
                               f"{(current - prior) / 1000:.3f}", "0", "0", "0", "0"])
        stock_rows.append([DEMO_NOTICE])
        manifest["sources"].append({"issue_id": issue["issue_id"], **capture(
            issue["issue_id"] + "-table4.csv", _csv(stock_rows), issue["table_url"])})
    for issue in SUPPLEMENTARY_ISSUES:
        released, observed = map(date.fromisoformat, (issue["release_date"], issue["observation_date"]))
        page = (f"<p>{DEMO_NOTICE}</p><p>Release Date: {MONTH_NAMES[released.month]} {released.day:02d}, {released.year}"
                f" Data for week ending {MONTH_NAMES[observed.month]} {observed.day:02d}, {observed.year}</p>").encode()
        manifest["supplementary_issue_pages"].append({"issue_id": issue["issue_id"], **capture(
            issue["issue_id"] + "-issue.html", page, issue["issue_url"])})
    (root / "manifest.json").write_bytes(json_bytes(manifest))


def generate_sources(output_root: Path) -> dict:
    """Create reproducible typed source fixtures without network access."""
    root = fresh_directory(output_root)
    root.mkdir(parents=True)
    vintages = _price_sources(root / "vintages")
    _inventory_sources(root / "inventories", vintages)
    result = {"synthetic": True, "notice": DEMO_NOTICE,
              "vintage_dir": str(root / "vintages"), "inventory_snapshot": str(root / "inventories"),
              "initial_training_issuances": 160, "development_issuances": 7,
              "last_forecast_date": next_business_day(vintages[-1]).isoformat(),
              "maximum_numeric_observation_date": vintages[-1].isoformat(),
              "synthetic_future_gate_sentinels_per_series": 2,
              "generation": "deterministic arithmetic; no prices, inventory values or files copied from research captures"}
    (root / "synthetic_sources.json").write_bytes(json_bytes(result))
    return result


def run(run_root: Path, *, tree: bool = False) -> dict:
    root = fresh_directory(run_root)
    root.mkdir(parents=True)
    sources = generate_sources(root / "sources")
    result = replay_sources(Path(sources["vintage_dir"]), Path(sources["inventory_snapshot"]), root / "replay",
                            brent=True, benchmarks=True, tree=tree, synthetic=True)
    (root / "demo.json").write_bytes(json_bytes({"synthetic": True, "notice": DEMO_NOTICE,
                                                 "workflow_manifest": str(root / "replay/workflow.json")}))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=workspace_path, required=True)
    parser.add_argument("--tree", action="store_true", help="also exercise the fixed small-tree candidate")
    args = parser.parse_args()
    result = run(**vars(args))
    print(json.dumps({"status": result["status"], "notice": DEMO_NOTICE,
                      "workflow_manifest": str(args.run_root / "replay/workflow.json"),
                      "holdout_evaluated": False}, indent=2))


if __name__ == "__main__":
    main()

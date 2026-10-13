"""Synthetic inventory feature replay and closed-audit authentication checks."""

from copy import deepcopy
from datetime import date, timedelta
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from oil_audit.calendar import add_business_days, next_business_day
from oil_audit.dataset import json_bytes, jsonl_bytes, sha256
from oil_audit.features import WTI_FEATURE_COLUMNS
from oil_audit.inventory import run
from oil_eval.inventory_features import (INVENTORY_FEATURE_COLUMNS, INVENTORY_MODEL_COLUMNS,
                                         attach_inventory_features, load_inventory_reference)
from tests import test_inventory_audit as audit_fixtures


def feature(day):
    issued = date.fromisoformat(day)
    return {"row_id": "wti-direction-" + day.replace("-", ""), "forecast_date": day,
            "target_date": add_business_days(issued, 5).isoformat(),
            "information_cutoff_date": (issued - timedelta(days=1)).isoformat(),
            "features": dict.fromkeys(WTI_FEATURE_COLUMNS, 1.0), "eligibility": {"wti": True}}


def release(released, observed, commercial=400.0, cushing=20.0, previous=None,
            changes=(1.0, 0.5), issue_id=None):
    released_day = date.fromisoformat(released)
    slug = issue_id or released.replace("-", "_")
    issue = {"issue_id": slug, "release_date": released, "observation_date": observed,
             "issue_url": "https://www.eia.gov/petroleum/supply/weekly/archive/" + released[:4] + "/" + slug + "/wpsr_" + slug + ".php",
             "table_url": "https://www.eia.gov/petroleum/supply/weekly/archive/" + released[:4] + "/" + slug + "/csv/table4.csv"}
    previous = previous or (commercial - changes[0], cushing - changes[1])
    record = {**issue,
              "previous_observation_date": (date.fromisoformat(observed) - timedelta(days=7)).isoformat(),
              "earliest_eligible_issuance_date": next_business_day(released_day).isoformat(),
              "source_file": slug + "-table4.csv", "source_sha256": sha256(slug.encode()), "series": {}}
    for index, (name, current) in enumerate((("commercial_crude", commercial), ("cushing", cushing))):
        regime = ("cushing_reported" if name == "cushing" else
                  "includes_lease_stocks" if released < "2016-10-13" else "excludes_lease_stocks")
        record["series"][name] = {"current_million_barrels": current,
                                   "previous_million_barrels": previous[index],
                                   "published_change_million_barrels": changes[index],
                                   "definition_regime": regime}
    return issue, record


def build(days, releases):
    return attach_inventory_features([feature(day) for day in days],
                                     [item[0] for item in releases], [item[1] for item in releases])


class InventoryFeatureTests(unittest.TestCase):
    def test_fixed_columns_and_published_weekly_change_not_rounded_subtraction(self):
        self.assertEqual(len(INVENTORY_FEATURE_COLUMNS), 5)
        self.assertEqual(INVENTORY_MODEL_COLUMNS, WTI_FEATURE_COLUMNS + INVENTORY_FEATURE_COLUMNS)
        issue = release("2018-02-07", "2018-02-02", previous=(399.999, 20.001), changes=(0.002, -0.002))
        rows, summary = build(["2018-02-08"], [issue])
        values = rows[0]["features"]
        self.assertEqual(values[INVENTORY_FEATURE_COLUMNS[0]], 0.002)
        self.assertEqual(values[INVENTORY_FEATURE_COLUMNS[2]], -0.002)
        self.assertEqual(values[INVENTORY_FEATURE_COLUMNS[4]], 1)
        self.assertEqual(summary["row_count"], 1)

    def test_no_earlier_release_stays_null_without_backfilling_prior_column(self):
        issue = release("2011-07-27", "2011-07-22")
        rows, summary = build(["2011-07-20", "2011-07-27", "2011-07-28"], [issue])
        for row in rows[:2]:
            self.assertTrue(all(row["features"][name] is None for name in INVENTORY_FEATURE_COLUMNS))
            self.assertEqual(row["inventory_provenance"]["status"], "no_earlier_release")
        self.assertEqual(summary["latest_issue_status_counts"]["no_earlier_release"], 2)
        self.assertIsNotNone(rows[2]["features"][INVENTORY_FEATURE_COLUMNS[0]])

    def test_unusable_latest_declared_issue_keeps_age_without_older_stock_substitution(self):
        old = release("2019-07-03", "2019-06-28")
        stale = release("2019-07-10", "2019-07-05")
        rows, summary = attach_inventory_features([feature("2019-07-11")], [old[0], stale[0]], [old[1]])
        values, provenance = rows[0]["features"], rows[0]["inventory_provenance"]
        self.assertTrue(all(values[name] is None for name in INVENTORY_FEATURE_COLUMNS[:4]))
        self.assertEqual(values[INVENTORY_FEATURE_COLUMNS[4]], 1)
        self.assertEqual(provenance["latest_declared_issue"]["issue_id"], stale[0]["issue_id"])
        self.assertIsNone(provenance["latest_declared_issue"]["source_sha256"])
        self.assertEqual(provenance["features"][INVENTORY_FEATURE_COLUMNS[0]]["missing_reason"],
                         "latest_declared_issue_unusable")
        self.assertEqual(summary["feature_available_counts"][INVENTORY_FEATURE_COLUMNS[4]], 1)

    def test_delayed_dual_release_selects_latest_observation_only_after_release(self):
        releases = [release("2022-06-15", "2022-06-10", commercial=410),
                    release("2022-06-29", "2022-06-17", commercial=999, issue_id="2022_06_17_data"),
                    release("2022-06-29", "2022-06-24", commercial=420, changes=(3, 2))]
        rows, _ = build(["2022-06-23", "2022-06-29", "2022-06-30"], releases)
        self.assertEqual([row["inventory_provenance"]["latest_declared_issue"]["observation_date"] for row in rows],
                         ["2022-06-10", "2022-06-10", "2022-06-24"])
        self.assertEqual(rows[-1]["features"][INVENTORY_FEATURE_COLUMNS[0]], 3)

    def test_four_week_reference_uses_exact_date_and_latest_prior_column_revision(self):
        releases = [release("2018-01-10", "2018-01-05", commercial=390, cushing=15),
                    release("2018-01-17", "2018-01-12", previous=(391, 16)),
                    release("2018-02-07", "2018-02-02", commercial=410, cushing=25)]
        rows, _ = build(["2018-02-08"], releases)
        self.assertEqual(rows[0]["features"][INVENTORY_FEATURE_COLUMNS[1]], 19)
        self.assertEqual(rows[0]["features"][INVENTORY_FEATURE_COLUMNS[3]], 9)
        provenance = rows[0]["inventory_provenance"]["features"][INVENTORY_FEATURE_COLUMNS[1]]
        self.assertEqual(provenance["reference_observation_date"], "2018-01-05")
        self.assertEqual(provenance["reference"]["column"], "previous")
        self.assertEqual(provenance["reference"]["source_sha256"], releases[1][1]["source_sha256"])
        self.assertEqual(provenance["reference"]["value_observation_date"], "2018-01-05")

    def test_nearby_reference_cannot_replace_exact_four_week_date(self):
        releases = [release("2018-01-03", "2017-12-29"), release("2018-02-07", "2018-02-02")]
        rows, _ = build(["2018-02-08"], releases)
        for name in (INVENTORY_FEATURE_COLUMNS[1], INVENTORY_FEATURE_COLUMNS[3]):
            self.assertIsNone(rows[0]["features"][name])
            self.assertEqual(rows[0]["inventory_provenance"]["features"][name]["missing_reason"],
                             "no_available_reference_observation")

    def test_regime_break_blocks_commercial_reference_but_not_cushing(self):
        releases = [release("2016-09-21", "2016-09-16", commercial=431, cushing=18),
                    release("2016-10-19", "2016-10-14", commercial=401, cushing=21)]
        rows, _ = build(["2016-10-20"], releases)
        self.assertIsNone(rows[0]["features"][INVENTORY_FEATURE_COLUMNS[1]])
        self.assertEqual(rows[0]["features"][INVENTORY_FEATURE_COLUMNS[3]], 3)
        self.assertEqual(rows[0]["inventory_provenance"]["features"][INVENTORY_FEATURE_COLUMNS[1]]["missing_reason"],
                         "definition_regime_mismatch")

    def test_new_regime_prior_restatement_matures_into_valid_four_week_reference(self):
        releases = [release("2016-10-05", "2016-09-30", commercial=430),
                    release("2016-10-13", "2016-10-07", previous=(399, 17)),
                    release("2016-11-02", "2016-10-28", commercial=402, cushing=20)]
        rows, _ = build(["2016-11-03"], releases)
        self.assertEqual(rows[0]["features"][INVENTORY_FEATURE_COLUMNS[1]], 3)
        reference = rows[0]["inventory_provenance"]["features"][INVENTORY_FEATURE_COLUMNS[1]]["reference"]
        self.assertEqual(reference["issue_id"], "2016_10_13")
        self.assertEqual(reference["definition_regime"], "excludes_lease_stocks")
        self.assertEqual(reference["value_observation_date"], "2016-09-30")

    def test_future_reports_and_revisions_cannot_change_earlier_frozen_rows(self):
        releases = [release("2018-01-10", "2018-01-05", commercial=390),
                    release("2018-02-07", "2018-02-02", commercial=410),
                    release("2018-02-14", "2018-02-09", commercial=420)]
        before, _ = build(["2018-02-08"], releases)
        changed = deepcopy(releases)
        changed[-1][1]["series"]["commercial_crude"].update(current_million_barrels=999, previous_million_barrels=888)
        after, _ = build(["2018-02-08", "2018-02-15"], changed)
        self.assertEqual(before[0], after[0])
        self.assertEqual(before[0]["features"][INVENTORY_FEATURE_COLUMNS[1]], 20)

    def test_late_reference_revision_cannot_enter_on_its_own_release_date(self):
        releases = [release("2018-01-10", "2018-01-05", commercial=390),
                    release("2018-02-07", "2018-02-02", commercial=410),
                    release("2018-02-08", "2018-01-05", commercial=395)]
        rows, _ = build(["2018-02-08", "2018-02-09"], releases)
        self.assertEqual([row["features"][INVENTORY_FEATURE_COLUMNS[1]] for row in rows], [20, 15])
        self.assertEqual(rows[0]["inventory_provenance"]["features"][INVENTORY_FEATURE_COLUMNS[1]]["reference"]["issue_id"],
                         "2018_01_10")

    def test_input_rows_and_sources_are_unchanged_and_refreezing_is_rejected(self):
        rows, releases = [feature("2018-02-08")], [release("2018-02-07", "2018-02-02")]
        original_rows, original_releases = deepcopy(rows), deepcopy(releases)
        attached, _ = attach_inventory_features(rows, [releases[0][0]], [releases[0][1]])
        attached[0]["eligibility"]["wti"] = False
        self.assertEqual(rows, original_rows)
        self.assertEqual(releases, original_releases)
        with self.assertRaisesRegex(ValueError, "already been frozen"):
            attach_inventory_features(attached, [releases[0][0]], [releases[0][1]])

    def test_direct_holdout_forecasts_targets_and_bad_cutoffs_are_rejected(self):
        for row in (feature("2023-01-03"), feature("2022-12-30"),
                    {**feature("2018-02-08"), "information_cutoff_date": "2018-02-08"}):
            with self.subTest(row=row), self.assertRaises(ValueError):
                attach_inventory_features([row], [], [])

    def test_bad_record_scope_numbers_maturity_hashes_and_duplicate_issues_fail(self):
        original = release("2018-02-07", "2018-02-02")
        mutations = [lambda i, r: r.update(source_sha256="bad"),
                     lambda i, r: r.update(earliest_eligible_issuance_date="2018-02-07"),
                     lambda i, r: r["series"]["commercial_crude"].update(current_million_barrels=float("nan")),
                     lambda i, r: r["series"]["commercial_crude"].update(definition_regime="includes_lease_stocks"),
                     lambda i, r: i.update(release_date="2023-02-07")]
        for mutate in mutations:
            issue, record = deepcopy(original)
            mutate(issue, record)
            with self.subTest(mutation=mutate), self.assertRaises(ValueError):
                attach_inventory_features([feature("2018-02-08")], [issue], [record])
        with self.assertRaises(ValueError):
            attach_inventory_features([feature("2018-02-08")], [original[0], original[0]], [original[1]])


class InventoryReferenceTests(unittest.TestCase):
    def setUp(self):
        self.fixture = audit_fixtures.SnapshotTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root, self.snapshot = self.fixture.root, self.fixture.snapshot
        self.report_path = self.root / "inventory-report.json"
        reference = {"run_id": "synthetic", "evaluation_directory": str(self.root / "reference")}
        with patch("oil_audit.inventory._load_saved_run", return_value=(reference, {}, [feature("2017-01-05")])):
            self.report = run(self.snapshot, self.root / "audits", self.report_path)
        self.directory = Path(self.report["audit_directory"])
        self.manifest = json.loads((self.directory / "manifest.json").read_bytes())

    def save_manifest(self):
        identity = {key: value for key, value in self.manifest.items() if key not in ("audit_id", "artifacts")}
        self.manifest["audit_id"] = sha256(json_bytes(identity))[:20]
        destination = self.directory.parent / self.manifest["audit_id"]
        if destination != self.directory:
            self.directory.rename(destination)
            self.directory = destination
        (self.directory / "manifest.json").write_bytes(json_bytes(self.manifest))
        summary = json.loads((self.directory / "summary.json").read_bytes())
        self.report = {"audit_id": self.manifest["audit_id"], "audit_directory": str(self.directory),
                       "source_snapshot_directory": str(self.snapshot), "reference_run": self.manifest["reference_run"],
                       "auxiliary_evidence": self.manifest["auxiliary_evidence"], **summary}
        self.report_path.write_bytes(json_bytes(self.report))

    def replace_artifact(self, name, payload):
        (self.directory / name).write_bytes(payload)
        self.manifest["artifacts"][name] = {"sha256": sha256(payload), "bytes": len(payload)}
        self.save_manifest()

    def test_authenticates_every_primary_source_and_artifact_without_models_or_price_dataset(self):
        with patch("oil_eval.data.load_development_data", side_effect=AssertionError("price data read")), \
             patch("oil_eval.models.Pipeline.fit", side_effect=AssertionError("model fit")):
            identity, issues, records = load_inventory_reference(self.report_path)
        self.assertEqual(identity["audit_id"], self.report["audit_id"])
        self.assertEqual(set(identity["artifacts"]),
                         {"records.jsonl", "overlap_changes.jsonl", "development_coverage.jsonl", "summary.json"})
        self.assertEqual(len(issues), len(records))
        self.assertEqual(identity["source_snapshot"]["manifest_sha256"], sha256((self.snapshot / "manifest.json").read_bytes()))

    def test_historical_code_hashes_are_retained_without_current_code_equality(self):
        self.manifest["audit_code_sha256"]["oil_audit/inventory.py"] = "f" * 64
        self.save_manifest()
        identity, _, _ = load_inventory_reference(self.report_path)
        self.assertEqual(identity["audit_code_sha256"]["oil_audit/inventory.py"], "f" * 64)

    def test_each_corrupt_audit_artifact_fails_before_primary_source_replay(self):
        for name in self.manifest["artifacts"]:
            path = self.directory / name
            original = path.read_bytes()
            path.write_bytes(original + b"corrupt")
            with self.subTest(name=name), patch("oil_audit.inventory.load_snapshot", side_effect=AssertionError("source parse")):
                with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                    load_inventory_reference(self.report_path)
            path.write_bytes(original)

    def test_corrupt_later_primary_source_fails_even_for_earlier_forecast_coverage(self):
        later = next(row for row in self.fixture.manifest["sources"] if row["issue_id"].startswith("2022"))
        path = self.snapshot / later["file"]
        path.write_bytes(path.read_bytes() + b"corrupt")
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            load_inventory_reference(self.report_path)

    def test_rehashed_records_must_still_match_primary_source_extraction(self):
        rows = [json.loads(line) for line in (self.directory / "records.jsonl").read_bytes().splitlines()]
        rows[0]["series"]["commercial_crude"]["current_million_barrels"] += 100
        self.replace_artifact("records.jsonl", jsonl_bytes(rows))
        with self.assertRaisesRegex(ValueError, "records differ"):
            load_inventory_reference(self.report_path)

    def test_rehashed_overlap_artifact_is_semantically_verified(self):
        self.replace_artifact("overlap_changes.jsonl", b'{"invented": true}\n')
        with self.assertRaisesRegex(ValueError, "overlap artifact differs"):
            load_inventory_reference(self.report_path)

    def test_poisoned_holdout_coverage_is_rejected_before_decoding(self):
        self.replace_artifact("development_coverage.jsonl", b'{"forecast_date":"2023-01-03", NOT JSON}\n')
        with self.assertRaisesRegex(ValueError, "development boundary"):
            load_inventory_reference(self.report_path)

    def test_rehashed_summary_and_unanchored_report_mutations_fail(self):
        self.report["parsed_issue_count"] += 1
        self.report_path.write_bytes(json_bytes(self.report))
        with self.assertRaisesRegex(ValueError, "report does not match"):
            load_inventory_reference(self.report_path)
        summary = json.loads((self.directory / "summary.json").read_bytes())
        summary["parsed_issue_count"] += 1
        self.replace_artifact("summary.json", json_bytes(summary))
        with self.assertRaisesRegex(ValueError, "summary differs"):
            load_inventory_reference(self.report_path)

    def test_unsupported_paths_artifact_names_and_holdout_scope_fail(self):
        for path in (Path("../inventory-report.json"), self.report_path.resolve()):
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, "inside the workspace"):
                load_inventory_reference(path)
        self.manifest["artifacts"]["../outside.json"] = {"sha256": "0" * 64, "bytes": 0}
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, "exactly four"):
            load_inventory_reference(self.report_path)
        del self.manifest["artifacts"]["../outside.json"]
        self.manifest["config"]["release_years"] = [2011, 2023]
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, "closed historical"):
            load_inventory_reference(self.report_path)

    def test_report_directory_symlink_cannot_escape_workspace(self):
        link = self.root / "escape"
        link.symlink_to(Path.cwd().parent, target_is_directory=True)
        self.report["audit_directory"] = str(link / self.directory.name)
        self.report_path.write_bytes(json_bytes(self.report))
        with self.assertRaisesRegex(ValueError, "inside the workspace"):
            load_inventory_reference(self.report_path)


if __name__ == "__main__":
    unittest.main()

"""Synthetic source authentication and exact-date delayed feature construction."""

from copy import deepcopy
from datetime import date, timedelta
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from oil_audit.availability import PointInTimeSeries
from oil_audit.dataset import json_bytes, sha256
from oil_audit.features import WTI_FEATURE_COLUMNS, build_features
from oil_eval.delay_data import (build_delayed_feature_rows, delay_information_dates,
                                 load_delay_wti_source, reconstruct_baseline_wti_rows)
from oil_eval.inventory_features import INVENTORY_FEATURE_COLUMNS, attach_inventory_features
from tests.test_availability import row as version
from tests.test_bounded_vintages import write_archive
from tests.test_inventory_features import feature, release


def original_rows(wti, dates, releases=()):
    rows = []
    for day in dates:
        row = feature(day)
        values, provenance, eligibility = build_features(wti, wti, date.fromisoformat(day))
        row.update(features={name: values[name] for name in WTI_FEATURE_COLUMNS},
                   provenance={"wti": provenance["wti"]}, eligibility=eligibility)
        rows.append(row)
    return attach_inventory_features(rows, [item[0] for item in releases], [item[1] for item in releases])[0]


def february_series():
    return PointInTimeSeries("DCOILWTICO", [
        version("2018-01-31", "2018-02-01", "", "40"),
        version("2018-02-05", "2018-02-07", "", "50"),
        version("2018-02-06", "2018-02-08", "", "60"),
        version("2018-02-07", "2018-02-09", "", "70"),
    ])


class DelayFeatureTests(unittest.TestCase):
    def test_extra_business_day_clock_handles_weekends_and_observed_holidays(self):
        cases = {"2018-02-09": ("2018-02-08", "2018-02-07"),
                 "2018-02-12": ("2018-02-09", "2018-02-08"),
                 "2019-07-05": ("2019-07-03", "2019-07-02"),
                 "2021-06-21": ("2021-06-17", "2021-06-16")}
        for day, expected in cases.items():
            self.assertEqual(tuple(value.isoformat() for value in delay_information_dates(date.fromisoformat(day))), expected)
        for day in ("2023-01-03", "2018-02-10"):
            with self.assertRaises(ValueError):
                delay_information_dates(date.fromisoformat(day))

    def test_reconstructs_actual_dates_with_source_delay_and_ages_at_original_issuance(self):
        wti = february_series()
        releases = [release("2018-01-31", "2018-01-26", changes=(1, 0.5)),
                    release("2018-02-07", "2018-02-02", changes=(3, 2))]
        dates = ["2018-02-08", "2018-02-09", "2018-02-12"]
        original = original_rows(wti, dates, releases)
        before = deepcopy((original, releases))
        _, check = reconstruct_baseline_wti_rows(original, wti)
        self.assertTrue(check["all_wti_features_and_eligibility_identical"])
        self.assertEqual(check["original_wti_projection_sha256"], check["reconstructed_wti_projection_sha256"])
        delayed, summary = build_delayed_feature_rows(original, wti,
                                                       [item[0] for item in releases], [item[1] for item in releases])
        self.assertEqual([row["features"]["wti_latest_price"] for row in delayed], [40, 50, 60])
        self.assertEqual([row["features"]["wti_age_calendar_days"] for row in delayed], [8, 4, 6])
        self.assertEqual([row["features"][INVENTORY_FEATURE_COLUMNS[0]] for row in delayed], [1, 3, 3])
        self.assertEqual([row["features"][INVENTORY_FEATURE_COLUMNS[-1]] for row in delayed], [8, 2, 5])
        for row, baseline in zip(delayed, original):
            self.assertEqual({key: row[key] for key in ("row_id", "forecast_date", "target_date")},
                             {key: baseline[key] for key in ("row_id", "forecast_date", "target_date")})
            self.assertEqual(row["inventory_provenance"]["information_cutoff_date"], row["information_cutoff_date"])
        self.assertEqual(summary["row_count"], 3)
        self.assertEqual((original, releases), before)

    def test_initial_unavailable_wti_row_stays_explicit_and_baseline_mismatch_stops(self):
        wti = PointInTimeSeries("DCOILWTICO", [version("2011-04-05", "2011-04-06", "", "50")])
        original = original_rows(wti, ["2011-04-07"])
        delayed, summary = build_delayed_feature_rows(original, wti, [], [])
        self.assertEqual(len(delayed), 1)
        self.assertFalse(delayed[0]["eligibility"]["wti"])
        self.assertTrue(all(value is None for value in delayed[0]["features"].values()))
        self.assertEqual(summary["wti_status_counts"], {"no_numeric_observation_at_delayed_cutoff": 1})
        original[0]["features"]["wti_latest_price"] += 1
        with self.assertRaisesRegex(ValueError, "Baseline WTI archive reconstruction differs"):
            reconstruct_baseline_wti_rows(original, wti)

    def test_withdrawals_use_delayed_active_state_and_never_revive_older_numeric_value(self):
        wti = PointInTimeSeries("DCOILWTICO", [
            version("2018-02-05", "2018-02-07", "2018-02-08", "50"),
            version("2018-02-05", "2018-02-09", "", "."),
        ])
        original = original_rows(wti, ["2018-02-09", "2018-02-12", "2018-02-13"])
        delayed, _ = build_delayed_feature_rows(original, wti, [], [])
        self.assertEqual([row["features"]["wti_latest_price"] for row in delayed], [50, 50, None])
        self.assertFalse(delayed[-1]["eligibility"]["wti"])

    def test_inventory_unusable_issue_and_four_week_revisions_follow_same_delay(self):
        wti = february_series()
        releases = [release("2018-01-10", "2018-01-05", commercial=390),
                    release("2018-02-07", "2018-02-02", commercial=410),
                    release("2018-02-08", "2018-01-05", commercial=395)]
        original = original_rows(wti, ["2018-02-09", "2018-02-12"], releases)
        delayed, _ = build_delayed_feature_rows(original, wti,
                                                [item[0] for item in releases], [item[1] for item in releases])
        self.assertEqual([row["features"][INVENTORY_FEATURE_COLUMNS[1]] for row in delayed], [20, 15])
        stale = release("2018-02-07", "2018-02-02")
        old = release("2018-01-31", "2018-01-26")
        original = original_rows(wti, ["2018-02-08", "2018-02-09"], [old])
        delayed, _ = build_delayed_feature_rows(original, wti, [old[0], stale[0]], [old[1]])
        self.assertIsNotNone(delayed[0]["features"][INVENTORY_FEATURE_COLUMNS[0]])
        self.assertTrue(all(delayed[1]["features"][name] is None for name in INVENTORY_FEATURE_COLUMNS[:4]))
        self.assertEqual(delayed[1]["features"][INVENTORY_FEATURE_COLUMNS[-1]], 2)
        self.assertEqual(delayed[1]["inventory_provenance"]["status"], "latest_declared_issue_unusable")

    def test_future_sources_cannot_change_frozen_earlier_rows_and_holdout_metadata_fails(self):
        wti = february_series()
        releases = [release("2018-02-07", "2018-02-02"), release("2018-02-14", "2018-02-09")]
        original = original_rows(wti, ["2018-02-09"], releases)
        before = build_delayed_feature_rows(original, wti, [item[0] for item in releases], [item[1] for item in releases])[0]
        changed = deepcopy(releases)
        changed[-1][1]["series"]["commercial_crude"]["current_million_barrels"] = 9999
        after = build_delayed_feature_rows(original, wti, [item[0] for item in changed], [item[1] for item in changed])[0]
        self.assertEqual(before, after)
        with self.assertRaises(ValueError):
            build_delayed_feature_rows([feature("2023-01-03")], wti, [], [])
        issue, record = release("2023-01-04", "2022-12-30")
        record["series"] = "must not be inspected"
        with self.assertRaisesRegex(ValueError, "holdout sources"):
            build_delayed_feature_rows(original, wti, [issue], [record])


class DelaySourceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix=".test-delay-source-", dir=".")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).relative_to(Path.cwd())
        self.archive = write_archive(self.root, [b"2011-04-05,50,2011-04-06,2026-01-01",
                                                b"2022-12-27,70,2022-12-28,",
                                                b"2023-01-03,\xff,2023-01-04,"])
        self.page = self.root / "DCOILWTICO-download-page.html"
        self.page.write_text('<select id="form_selected_vintage_dates">'
                             '<option value="2011-04-06"><option value="2022-12-28">'
                             '<option value="2023-01-04"></select>'
                             '<input id="form_obs_start_date" value="1986-01-01">'
                             '<input id="form_obs_end_date" value="2026-01-01">')
        records = [{"path": str(path), "sha256": sha256(path.read_bytes()), "bytes": path.stat().st_size,
                    "retrieved_at_utc": "2024-01-01T00:00:00+00:00"} for path in (self.archive, self.page)]
        self.manifest = json_bytes(records)
        (self.root / "manifest.json").write_bytes(self.manifest)

    def test_source_authenticates_full_bytes_but_materializes_only_bounded_versions(self):
        metadata, series = load_delay_wti_source(self.root, sha256(self.manifest))
        self.assertEqual(metadata["source_manifest_sha256"], sha256(self.manifest))
        self.assertEqual(len(metadata["verified_source_artifacts"]), 2)
        self.assertEqual(metadata["retained_interval_count"], 2)
        self.assertEqual(series.last_vintage, date(2022, 12, 28))
        self.assertNotIn(date(2023, 1, 3), series.versions_by_observation)
        self.assertIsNone(series.versions_by_observation[date(2011, 4, 5)][0].vintage_end)

    def test_manifest_mismatch_or_corrupt_source_fails_before_archive_parser(self):
        with patch("oil_eval.delay_data.bounded_records", side_effect=AssertionError("must authenticate first")):
            with self.assertRaisesRegex(ValueError, "manifest differs"):
                load_delay_wti_source(self.root, "0" * 64)
            self.archive.write_bytes(self.archive.read_bytes() + b"corruption")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                load_delay_wti_source(self.root, sha256(self.manifest))


if __name__ == "__main__":
    unittest.main()

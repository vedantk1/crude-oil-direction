"""Adversarial tests for the pre-decoding development data gate."""

from copy import deepcopy
from datetime import date, datetime, time, timedelta
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from zoneinfo import ZoneInfo

from oil_audit.calendar import CALENDAR_ID, add_business_days
from oil_audit.features import FEATURE_COLUMNS, FEATURE_SETS, WTI_FEATURE_COLUMNS
from oil_eval.data import ARTIFACTS, load_development_data


def feature(day="2011-04-07", feature_set="wti"):
    forecast = date.fromisoformat(day)
    return {
        "row_id": "wti-direction-" + forecast.strftime("%Y%m%d"),
        "forecast_date": day,
        "forecast_timestamp": datetime.combine(
            forecast, time(18), tzinfo=ZoneInfo("America/New_York")).isoformat(),
        "target_date": add_business_days(forecast, 5).isoformat(),
        "information_cutoff_date": (forecast - timedelta(days=1)).isoformat(),
        "features": {**dict.fromkeys(FEATURE_SETS[feature_set], 1.0), "brent_latest_price": 200.0},
        "eligibility": {"wti": True, **({"wti_brent": True} if feature_set == "wti_brent" else {})},
        "provenance": {"wti": {"latest_observation_date": "2011-04-06"}},
    }


def history(row):
    identity = {key: row[key] for key in ("row_id", "forecast_date", "target_date")}
    missing = {
        **identity, "available_from": row["forecast_date"], "available_until": "2011-04-15",
        "label_status": "missing_both", "label": None, "origin_price": None,
        "target_price": None, "origin_vintage_start": None, "target_vintage_start": None,
    }
    available = {
        **identity, "available_from": "2011-04-15", "available_until": None,
        "label_status": "available", "label": 1, "origin_price": -36.98,
        "target_price": 10.0, "origin_vintage_start": row["forecast_date"],
        "target_vintage_start": row["target_date"],
    }
    return [missing, available]


def fixture(feature_set="wti"):
    row = feature(feature_set=feature_set)
    events = history(row)
    scoring = {key: value for key, value in events[-1].items() if key != "available_until"}
    return {"features.jsonl": [row], "label_events.jsonl": events,
            "evaluation_labels.jsonl": [scoring]}


def save(directory, content):
    root = Path(directory)
    artifacts = {}
    for name in ARTIFACTS:
        payload = b"".join((line if isinstance(line, bytes) else json.dumps(line).encode()) + b"\n"
                           for line in content[name])
        (root / name).write_bytes(payload)
        artifacts[name] = {"sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}
    manifest = {"dataset_id": "synthetic-dataset", "source_manifest_sha256": "0" * 64,
                "config": {
                    "maximum_supported_as_of_date": "2026-10-08", "schema_version": 1,
                    "calendar_id": CALENDAR_ID, "horizon_business_days": 5,
                    "forecast_time": "18:00", "forecast_timezone": "America/New_York",
                    "information_cutoff": "preceding calendar date, inclusive",
                    "feature_sets": {name: list(columns) for name, columns in FEATURE_SETS.items()},
                },
                "artifacts": artifacts}
    (root / "manifest.json").write_text(json.dumps(manifest))
    return root


class DevelopmentDataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)

    def load(self, content=None, feature_set="wti"):
        content = content if content is not None else fixture(feature_set)
        return load_development_data(save(self.temp.name, content), feature_set=feature_set)

    def test_valid_rows_expose_only_wti_features_and_bounded_identity(self):
        identity, rows, events, scoring = self.load()
        self.assertEqual(set(rows[0]["features"]), set(WTI_FEATURE_COLUMNS))
        self.assertEqual(rows[0]["provenance"], fixture()["features.jsonl"][0]["provenance"])
        self.assertEqual(identity["maximum_as_of_date"], "2022-12-31")
        self.assertNotIn("summary", identity)
        self.assertEqual(events[-1]["label"], 1)
        self.assertEqual(scoring[0]["origin_price"], -36.98)

    def test_combined_rows_expose_fourteen_ordered_features(self):
        identity, rows, events, scoring = self.load(feature_set="wti_brent")
        self.assertEqual(tuple(rows[0]["features"]), FEATURE_COLUMNS)
        self.assertEqual(len(rows[0]["features"]), 14)
        self.assertEqual(rows[0]["eligibility"]["wti_brent"], True)
        wti_identity, _, wti_events, wti_scoring = self.load(fixture("wti_brent"))
        self.assertEqual(identity, wti_identity)
        self.assertEqual(events, wti_events)
        self.assertEqual(scoring, wti_scoring)

    def test_unknown_feature_set_is_rejected_before_source_io(self):
        for selection in ("brent", "all", "", None, []):
            with self.subTest(selection=selection), self.assertRaisesRegex(ValueError, "Unknown feature set"):
                load_development_data(Path(self.temp.name) / "nonexistent", selection)

    def test_wti_default_ignores_invalid_or_missing_brent_features(self):
        for value in (float("nan"), True, "invalid", None):
            content = fixture()
            content["features.jsonl"][0]["features"]["brent_latest_price"] = value
            self.assertEqual(tuple(self.load(content)[1][0]["features"]), WTI_FEATURE_COLUMNS)
        content["features.jsonl"][0]["features"].pop("brent_latest_price")
        self.assertEqual(tuple(self.load(content)[1][0]["features"]), WTI_FEATURE_COLUMNS)

    def test_combined_source_config_requires_exact_ordered_columns(self):
        for columns in (None, list(reversed(FEATURE_COLUMNS)), list(WTI_FEATURE_COLUMNS)):
            with self.subTest(columns=columns):
                root = save(self.temp.name, fixture("wti_brent"))
                manifest = json.loads((root / "manifest.json").read_text())
                manifest["config"]["feature_sets"]["wti_brent"] = columns
                (root / "manifest.json").write_text(json.dumps(manifest))
                with self.assertRaisesRegex(ValueError, "feature_sets.wti_brent"):
                    load_development_data(root, "wti_brent")
                # The WTI-only contract is independent of the unused feature set.
                self.assertEqual(tuple(load_development_data(root)[1][0]["features"]), WTI_FEATURE_COLUMNS)

    def test_combined_optional_nulls_are_retained_and_not_eligibility_requirements(self):
        content = fixture("wti_brent")
        optional = ("brent_change_5_business_days_usd", "brent_change_20_business_days_usd",
                    "brent_change_std_20_business_days_usd", "spread_age_calendar_days")
        content["features.jsonl"][0]["features"].update(dict.fromkeys(optional))
        row = self.load(content, "wti_brent")[1][0]
        self.assertTrue(row["eligibility"]["wti_brent"])
        for column in optional:
            self.assertIsNone(row["features"][column])

    def test_combined_missing_required_prices_are_retained_when_ineligible(self):
        for column in ("wti_latest_price", "brent_latest_price", "brent_wti_spread_usd"):
            with self.subTest(column=column):
                content = fixture("wti_brent")
                row = content["features.jsonl"][0]
                row["features"][column] = None
                row["eligibility"]["wti_brent"] = False
                if column == "wti_latest_price":
                    row["eligibility"]["wti"] = False
                result = self.load(content, "wti_brent")[1][0]
                self.assertFalse(result["eligibility"]["wti_brent"])
                self.assertIsNone(result["features"][column])

    def test_combined_nonfinite_missing_columns_and_eligibility_mismatch_rejected(self):
        for mutation in (
            lambda row: row["features"].update(brent_latest_price=float("inf")),
            lambda row: row["features"].update(brent_change_5_business_days_usd=float("nan")),
            lambda row: row["features"].update(brent_latest_price=True),
            lambda row: row["features"].pop("brent_coverage_20_business_days"),
            lambda row: row["features"].update(brent_latest_price=None),
            lambda row: row["features"].update(brent_wti_spread_usd=None),
            lambda row: row["eligibility"].update(wti_brent=False),
            lambda row: row["eligibility"].update(wti_brent=1),
            lambda row: row["eligibility"].pop("wti_brent"),
        ):
            content = fixture("wti_brent")
            mutation(content["features.jsonl"][0])
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.load(content, "wti_brent")

    def test_combined_holdout_and_future_labels_are_filtered_before_decoding(self):
        content = fixture("wti_brent")
        for name in ARTIFACTS:
            content[name].append(b'{"forecast_date":"2023-01-03", cannot decode this}')
        content["label_events.jsonl"][1]["available_until"] = "2023-01-03"
        for name in ("label_events.jsonl", "evaluation_labels.jsonl"):
            content[name].append(b'{"forecast_date":"2011-04-07", "available_from":"2023-01-03", poison}')
        baseline = self.load(fixture("wti_brent"), "wti_brent")[1:]
        self.assertEqual(self.load(content, "wti_brent")[1:], baseline)

    def test_checksum_and_size_mismatches_rejected_before_parsing(self):
        root = save(self.temp.name, fixture())
        (root / "features.jsonl").write_bytes(b"not json")
        with self.assertRaisesRegex(ValueError, "checksum"):
            load_development_data(root)
        root = save(self.temp.name, fixture())
        manifest = json.loads((root / "manifest.json").read_text())
        manifest["artifacts"]["features.jsonl"]["bytes"] += 1
        (root / "manifest.json").write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "checksum"):
            load_development_data(root)

    def test_holdout_poison_is_filtered_before_json_decoding(self):
        content = fixture()
        for name in ARTIFACTS:
            content[name].append(b'{"forecast_date":"2023-01-03", cannot decode this}')
        baseline = self.load(fixture())[1:]
        self.assertEqual(self.load(content)[1:], baseline)

    def test_future_label_poison_is_filtered_and_interval_end_masked(self):
        content = fixture()
        content["label_events.jsonl"][-1]["available_until"] = "2023-01-03"
        for name in ("label_events.jsonl", "evaluation_labels.jsonl"):
            content[name].append(b'{"forecast_date":"2011-04-07", "available_from":"2023-01-03", poison}')
        baseline = self.load(fixture())[1:]
        self.assertEqual(self.load(content)[1:], baseline)

    def test_future_withdrawal_does_not_change_development_events(self):
        original = self.load()[1:]
        content = fixture()
        content["label_events.jsonl"][-1]["available_until"] = "2026-09-17"
        withdrawal = deepcopy(content["label_events.jsonl"][-1])
        withdrawal.update(available_from="2026-09-17", available_until=None,
                          label_status="missing_target", label=None, target_price=None)
        content["label_events.jsonl"].append(withdrawal)
        self.assertEqual(self.load(content)[1:], original)

    def test_boundary_crossing_target_retained_for_explicit_runner_purge(self):
        content = fixture()
        boundary = feature("2022-12-29")
        content["features.jsonl"].append(boundary)
        initial = history(boundary)[0]
        initial["available_until"] = None
        content["label_events.jsonl"].append(initial)
        _, rows, _, _ = self.load(content)
        self.assertEqual(rows[-1]["target_date"], "2023-01-06")

    def test_invalid_feature_values_dates_and_duplicate_rows_rejected(self):
        for mutation in (
            lambda row: row["features"].update(wti_latest_price=float("nan")),
            lambda row: row["features"].update(wti_latest_price=True),
            lambda row: row.update(target_date="2011-04-15"),
            lambda row: row.update(information_cutoff_date="2011-04-07"),
            lambda row: row.update(forecast_timestamp="2011-04-07T18:00:00-05:00"),
            lambda row: row.update(forecast_timestamp="2011-04-07T17:00:00-04:00"),
            lambda row: row.update(row_id="wrong-id"),
            lambda row: row["eligibility"].update(wti=False),
        ):
            with self.subTest(mutation=mutation):
                content = fixture()
                mutation(content["features.jsonl"][0])
                with self.assertRaises(ValueError):
                    self.load(content)
        content = fixture()
        content["features.jsonl"].append(deepcopy(content["features.jsonl"][0]))
        with self.assertRaisesRegex(ValueError, "unique dates"):
            self.load(content)

    def test_ambiguous_or_pre_start_gate_rejected(self):
        for raw in (
            b'{"forecast_date":"2011-04-07","forecast_date":"2023-01-03"}',
            b'{"forecast_date":"2010-04-07"}',
            b'{"forecast_date":"2011-99-99"}',
        ):
            content = fixture()
            content["features.jsonl"] = [raw]
            with self.assertRaises(ValueError):
                self.load(content)

    def test_escaped_duplicate_json_key_cannot_bypass_gate(self):
        content = fixture()
        raw = json.dumps(content["features.jsonl"][0]).encode()
        content["features.jsonl"][0] = raw[:-1] + b', "forecast\\u005fdate":"2023-01-03"}'
        with self.assertRaisesRegex(ValueError, "Duplicate JSON"):
            self.load(content)

    def test_available_event_requires_both_endpoint_dates_and_valid_prices(self):
        for mutation in (
            lambda row: row.update(available_from="2011-04-14"),
            lambda row: row.update(origin_vintage_start="2011-04-06"),
            lambda row: row.update(target_price=None),
            lambda row: row.update(label=0),
            lambda row: row.update(label=True),
        ):
            content = fixture()
            mutation(content["label_events.jsonl"][-1])
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.load(content)

    def test_history_gap_overlap_or_missing_initial_event_rejected(self):
        for replacement in ("2011-04-14", "2011-04-18", None):
            content = fixture()
            content["label_events.jsonl"][0]["available_until"] = replacement
            with self.subTest(replacement=replacement), self.assertRaises(ValueError):
                self.load(content)
        content = fixture()
        content["label_events.jsonl"] = content["label_events.jsonl"][1:]
        with self.assertRaisesRegex(ValueError, "start at"):
            self.load(content)

    def test_missing_or_inconsistent_first_available_scoring_labels_rejected(self):
        content = fixture()
        content["evaluation_labels.jsonl"] = []
        with self.assertRaisesRegex(ValueError, "Missing first"):
            self.load(content)
        content = fixture()
        content["evaluation_labels.jsonl"][0]["target_price"] = 20.0
        with self.assertRaisesRegex(ValueError, "differs from its first"):
            self.load(content)

    def test_labels_for_unretained_forecasts_are_ignored(self):
        content = fixture()
        unknown = history(feature("2011-04-08"))[0]
        content["label_events.jsonl"].append(unknown)
        self.assertEqual(self.load(content)[1:], self.load(fixture())[1:])

    def test_early_snapshot_rejected(self):
        root = save(self.temp.name, fixture())
        manifest = json.loads((root / "manifest.json").read_text())
        manifest["config"]["maximum_supported_as_of_date"] = "2022-12-30"
        (root / "manifest.json").write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "does not cover"):
            load_development_data(root)

    def test_incompatible_source_configuration_rejected(self):
        for key, value in (
            ("schema_version", 2), ("schema_version", True),
            ("calendar_id", "exchange-sessions"), ("horizon_business_days", 10),
            ("forecast_time", "09:00"), ("forecast_timezone", "UTC"),
            ("information_cutoff", "same calendar date"),
            ("feature_sets", {"wti": list(reversed(WTI_FEATURE_COLUMNS))}),
        ):
            with self.subTest(key=key, value=value):
                root = save(self.temp.name, fixture())
                manifest = json.loads((root / "manifest.json").read_text())
                manifest["config"][key] = value
                (root / "manifest.json").write_text(json.dumps(manifest))
                with self.assertRaisesRegex(ValueError, "Incompatible source"):
                    load_development_data(root)


if __name__ == "__main__":
    unittest.main()

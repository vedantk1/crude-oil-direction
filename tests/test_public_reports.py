"""Public report handling uses synthetic values and never decodes scalars."""

import hashlib
import json
import unittest
from unittest.mock import patch

from oil_audit.public_reports import POLICY_VERSION, REPLACEMENT_MEANING, sanitize_report


def encode(value):
    return json.dumps(value, sort_keys=True).encode()


class PublicReportTests(unittest.TestCase):
    def test_price_policy_keeps_dates_counts_and_omits_every_quote(self):
        series = {
            "minimum": {"date": "2020-01-01", "value": -991.25},
            "maximum": {"date": "2020-01-02", "value": "synthetic price"},
            "nonpositive_observations": [{"date": "2020-01-01", "value": -991.25}],
            "largest_absolute_adjacent_changes": [
                {"previous_date": "2020-01-01", "date": "2020-01-02",
                 "previous_price": -991.25, "price": 997.75, "change_usd": 1989,
                 "calendar_days_between": 1}],
            "longest_calendar_gap": {"previous_price": 997.75, "price": 999.25,
                                     "change_usd": 1.5, "calendar_days_between": 4},
            "numeric_observations": 7,
        }
        payload = encode({"series": {"wti": series, "brent": series},
                          "source_provenance": [{"source_id": "synthetic"}]})
        public = json.loads(sanitize_report("price_audit.json", payload))
        for key in ("wti", "brent"):
            result = public["series"][key]
            self.assertEqual(result["numeric_observations"], 7)
            self.assertEqual(result["minimum"], {"date": "2020-01-01", "value": None})
            self.assertIsNone(result["maximum"]["value"])
            self.assertIsNone(result["nonpositive_observations"][0]["value"])
            change = result["largest_absolute_adjacent_changes"][0]
            for field in ("previous_price", "price", "change_usd"):
                self.assertIsNone(change[field])
                self.assertIsNone(result["longest_calendar_gap"][field])
            self.assertEqual(change["calendar_days_between"], 1)
        counts = public["public_sanitization"]["redacted_path_counts"]
        self.assertEqual(sum(counts.values()), 18)
        self.assertEqual(counts["series.*.minimum.value"], 2)

    def test_vintage_series_containers_survive_and_price_columns_do_not(self):
        original = {}
        for series in ("DCOILWTICO", "DCOILBRENTEU"):
            row = {"period_start_date": "2020-01-01", series: "poisoned quote",
                   "realtime_start_date": "2020-01-02", "realtime_end_date": ""}
            original[series] = {
                "series_id": series,
                "date_level_reconstruction_verified_against_samples": [
                    {"latest_value": "poisoned quote", "latest_observation": "2020-01-01",
                     "nonmissing_count": 4}],
                "first_numeric_lag_after_initial_vintage": {
                    "median": 1, "longest_examples": [{"lag": 2, **row}]},
                "same_or_future_observation_date_records": [row],
                "impossible_future_observation_date_records": [row],
                "event_examples": [{"before": "poisoned before", "after": "poisoned after",
                                    "kind": "numeric_to_numeric"}],
            }
        public = json.loads(sanitize_report("vintage_audit.json", encode(original)))
        for series in original:
            result = public[series]
            self.assertIsInstance(result, dict)
            self.assertEqual(result["series_id"], series)
            self.assertIsNone(result["date_level_reconstruction_verified_against_samples"][0]
                              ["latest_value"])
            self.assertIsNone(result["first_numeric_lag_after_initial_vintage"]
                              ["longest_examples"][0][series])
            for field in ("same_or_future_observation_date_records",
                          "impossible_future_observation_date_records"):
                self.assertIsNone(result[field][0][series])
                self.assertEqual(result[field][0]["period_start_date"], "2020-01-01")
            self.assertEqual(result["event_examples"][0], {
                "before": None, "after": None, "kind": "numeric_to_numeric"})
        self.assertEqual(sum(public["public_sanitization"]["redacted_path_counts"].values()), 12)

    def test_reconciliation_policy_keeps_mismatch_finding_without_prices(self):
        payload = encode({
            "current_eia_vs_alfred": {
                "wti": {"mismatch_count": 1, "mismatches": [
                    {"date": "2020-01-01", "eia": 991, "alfred": "999"}]},
                "brent": {"mismatch_count": 0, "mismatches": []}},
            "sampled_weekly_releases": [{"issue_date": "2020-01-02", "mismatches": [
                {"observation_date": "2020-01-01", "archived_price": "999",
                 "current_price": None}]}],
        })
        public = json.loads(sanitize_report("source_reconciliation.json", payload))
        self.assertEqual(public["current_eia_vs_alfred"]["wti"]["mismatch_count"], 1)
        self.assertEqual(public["current_eia_vs_alfred"]["wti"]["mismatches"], [
            {"date": "2020-01-01", "eia": None, "alfred": None}])
        self.assertEqual(public["sampled_weekly_releases"][0]["mismatches"], [
            {"observation_date": "2020-01-01", "archived_price": None,
             "current_price": None}])
        self.assertEqual(sum(public["public_sanitization"]["redacted_path_counts"].values()), 4)

    def test_only_keys_are_decoded_even_for_poisoned_source_scalar_tokens(self):
        # Both tokens are valid JSON. Converting the exponent to a float would
        # overflow, and the quoted marker must never reach a scalar decoder.
        payload = (b'{"series":{"wti":{"minimum":{"value":1e999999},'
                   b'"maximum":{"value":"DO_NOT_DECODE_VALUE"}}}}')
        original_loads = json.loads
        keys = []

        def keys_only(token):
            self.assertIn(token, {b'"series"', b'"wti"', b'"minimum"',
                                  b'"maximum"', b'"value"'})
            keys.append(token)
            return original_loads(token)

        with patch("oil_audit.public_reports.json.loads", side_effect=keys_only):
            public = sanitize_report("price_audit.json", payload)
        self.assertEqual(len(keys), 6)
        self.assertNotIn(b"1e999999", public)
        self.assertNotIn(b"DO_NOT_DECODE_VALUE", public)

    def test_unredacted_scalar_spellings_remain_byte_exact(self):
        payload = (b'{"series":{"wti":{"minimum":{"value":-12.5}}},'
                   b'"count":1.2300e+04,"zero":-0,"flag":true,'
                   b'"marker":"\\u0061\\/escaped","empty":null}')
        public = sanitize_report("price_audit.json", payload)
        for token in (b"1.2300e+04", b"-0", b"true", b'"\\u0061\\/escaped"', b"null"):
            self.assertIn(token, public)
        self.assertNotIn(b"-12.5", public)

    def test_metadata_identifies_original_bytes_without_private_paths(self):
        payload = b'{"series":{"wti":{"minimum":{"value":"synthetic"}}}}\n'
        metadata = json.loads(sanitize_report("price_audit.json", payload))["public_sanitization"]
        self.assertEqual(metadata["policy_version"], POLICY_VERSION)
        self.assertEqual(metadata["report"], "price_audit.json")
        self.assertEqual(metadata["original_sha256"], hashlib.sha256(payload).hexdigest())
        self.assertEqual(metadata["original_bytes"], len(payload))
        self.assertEqual(metadata["replacement_meaning"], REPLACEMENT_MEANING)
        self.assertEqual(set(metadata), {"policy_version", "report", "original_sha256",
                                         "original_bytes", "redacted_path_counts", "replacement_meaning"})

    def test_empty_lists_and_original_nulls_are_supported(self):
        payload = encode({"series": {"wti": {"minimum": {"value": None},
                                              "nonpositive_observations": []}}})
        public = json.loads(sanitize_report("price_audit.json", payload))
        self.assertEqual(public["series"]["wti"]["nonpositive_observations"], [])
        self.assertEqual(public["public_sanitization"]["redacted_path_counts"],
                         {"series.*.minimum.value": 1})

    def test_escaped_object_keys_match_redaction_policy(self):
        payload = b'{"series":{"wti":{"minimum":{"val\\u0075e":"synthetic"}}}}'
        public = sanitize_report("price_audit.json", payload)
        self.assertIn(b'"val\\u0075e": null', public)
        self.assertNotIn(b'"synthetic"', public)

    def test_matching_root_series_column_only_does_not_remove_metadata(self):
        payload = encode({"DCOILWTICO": {"same_or_future_observation_date_records": [
            {"DCOILWTICO": "synthetic", "series_id": "DCOILWTICO", "lag": 1}]}})
        public = json.loads(sanitize_report("vintage_audit.json", payload))
        row = public["DCOILWTICO"]["same_or_future_observation_date_records"][0]
        self.assertEqual(row, {"DCOILWTICO": None, "series_id": "DCOILWTICO", "lag": 1})

    def test_idempotence_preserves_original_identity_and_entire_public_payload(self):
        for name in ("price_audit.json", "vintage_audit.json", "source_reconciliation.json"):
            with self.subTest(name=name):
                once = sanitize_report(name, b"{}")
                self.assertEqual(sanitize_report(name, once), once)
        once = sanitize_report("price_audit.json", encode({"series": {
            "wti": {"minimum": {"value": "synthetic"}}}}))
        self.assertEqual(sanitize_report("price_audit.json", once), once)

    def test_duplicate_decoded_keys_fail_even_inside_redacted_values(self):
        for payload in (b'{"series":{},"ser\\u0069es":{}}',
                        b'{"series":{"wti":{"minimum":{"value":{"a":1,"a":2}}}}}'):
            with self.subTest(payload=payload), self.assertRaisesRegex(ValueError, "duplicate"):
                sanitize_report("price_audit.json", payload)

    def test_malformed_json_fails_before_any_output(self):
        for payload in (b"", b"[]", b"{} {}", b'{"x":NaN}', b'{"x":Infinity}',
                        b'{"x":01}', b'{"x":1.}', b'{"x":1e}', b'{"x":+1}',
                        b'{"x":truefalse}', b'{"x":}', b'{"x":"truncated}',
                        b'{"x":"bad\\q"}', b'{"x":"bad\\u12xy"}',
                        b'{"x":"\x01"}', b'{"x":"\xff"}', b'{"x":"\xc0\x80"}',
                        b'{"x":"\xed\xa0\x80"}', b'{"x":"\xf4\x90\x80\x80"}',
                        b'{"x":[1,]}', b'{"x":1,}', b'{"x":{', b'{"x":['):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                sanitize_report("price_audit.json", payload)

    def test_valid_utf8_and_escaped_string_syntax_are_preserved_without_decode(self):
        payload = '{"marker":"£ € 𐀀","escape":"\\\"\\\\\\/\\b\\f\\n\\r\\t\\u0061"}'.encode()
        public = sanitize_report("price_audit.json", payload)
        self.assertIn('"£ € 𐀀"'.encode(), public)
        self.assertEqual(json.loads(public)["marker"], "£ € 𐀀")

    def test_unknown_names_and_non_byte_payloads_fail_closed(self):
        for name in ("unknown.json", "reports/price_audit.json", "../price_audit.json"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                sanitize_report(name, b"{}")
        with self.assertRaises(TypeError):
            sanitize_report("price_audit.json", "{}")

    def test_unrecognized_existing_metadata_is_not_treated_as_sanitized(self):
        for metadata in (None, {}, {"policy_version": "unknown"}):
            with self.subTest(metadata=metadata), self.assertRaisesRegex(ValueError, "metadata"):
                sanitize_report("price_audit.json", encode({"public_sanitization": metadata}))

    def test_tampered_sanitized_quotes_and_counts_are_rejected(self):
        initial = json.loads(sanitize_report("price_audit.json", encode({"series": {
            "wti": {"minimum": {"value": "synthetic"}}}})))
        initial["series"]["wti"]["minimum"]["value"] = "new synthetic quote"
        with self.assertRaisesRegex(ValueError, "source quotation"):
            sanitize_report("price_audit.json", encode(initial))
        initial["series"]["wti"]["minimum"]["value"] = None
        initial["public_sanitization"]["redacted_path_counts"]["series.*.minimum.value"] = 99
        with self.assertRaisesRegex(ValueError, "counts"):
            sanitize_report("price_audit.json", encode(initial))

    def test_excessive_nesting_fails_with_a_controlled_error(self):
        payload = b'{"nested":' + b"[" * 102 + b"0" + b"]" * 102 + b"}"
        with self.assertRaisesRegex(ValueError, "nesting"):
            sanitize_report("price_audit.json", payload)


if __name__ == "__main__":
    unittest.main()

"""Adversarial checks on information available before forecast issuance."""

from datetime import date
import math
import unittest

from oil_audit.availability import PointInTimeSeries
from oil_audit.calendar import add_business_days
from oil_audit.features import FEATURE_COLUMNS, build_features


def row(series, observed, available, value, end=""):
    return {"period_start_date": observed, "realtime_start_date": available,
            "realtime_end_date": end, series: value}


def series(identifier, rows):
    return PointInTimeSeries(identifier, rows)


class FeatureTests(unittest.TestCase):
    def test_future_revisions_and_withdrawals_leave_earlier_values_and_provenance_identical(self):
        original, revised = [], []
        for identifier, price in (("DCOILWTICO", "30"), ("DCOILBRENTEU", "35")):
            original.append(series(identifier, [
                row(identifier, "2020-05-11", "2020-05-12", price)]))
            revised.append(series(identifier, [
                row(identifier, "2020-05-11", "2020-05-12", price, "2020-05-19"),
                row(identifier, "2020-05-11", "2020-05-20", "90", "2020-05-24"),
                row(identifier, "2020-05-11", "2020-05-25", ""),
                row(identifier, "2020-05-15", "2020-05-19", "100")]))
        before = build_features(*original, date(2020, 5, 14))
        after = build_features(*revised, date(2020, 5, 14))
        self.assertEqual(before, after)
        self.assertNotIn("vintage_end", repr(after[1]))
        self.assertNotIn("2020-05-20", repr(after[1]))

    def test_missing_exact_lookback_is_not_replaced_by_nearby_price(self):
        inputs = []
        for identifier in ("DCOILWTICO", "DCOILBRENTEU"):
            inputs.append(series(identifier, [
                row(identifier, "2020-05-01", "2020-05-12", "25"),
                row(identifier, "2020-05-05", "2020-05-12", "27"),
                row(identifier, "2020-05-11", "2020-05-12", "30")]))
        features, provenance, eligible = build_features(*inputs, date(2020, 5, 13))
        self.assertIsNone(features["wti_change_5_business_days_usd"])
        self.assertIsNone(features["wti_change_std_20_business_days_usd"])
        self.assertEqual(features["wti_coverage_20_business_days"], 3 / 21)
        missing = next(ref for ref in provenance["wti"]["input_versions"]
                       if ref["observation_date"] == "2020-05-04")
        self.assertEqual(missing["status"], "not_in_archive")
        self.assertTrue(eligible["wti"])
        self.assertFalse(eligible["complete_wti_features"])

    def test_each_series_has_independent_availability_and_spread_uses_same_date(self):
        wti = series("DCOILWTICO", [
            row("DCOILWTICO", "2020-05-07", "2020-05-12", "17"),
            row("DCOILWTICO", "2020-05-11", "2020-05-12", "20")])
        brent = series("DCOILBRENTEU", [
            row("DCOILBRENTEU", "2020-05-07", "2020-05-12", "22"),
            row("DCOILBRENTEU", "2020-05-08", "2020-05-12", "24"),
            row("DCOILBRENTEU", "2020-05-11", "2020-05-13", "100")])
        features, provenance, eligible = build_features(wti, brent, date(2020, 5, 13))
        self.assertEqual(features["wti_latest_price"], 20)
        self.assertEqual(features["brent_latest_price"], 24)
        self.assertEqual(features["wti_age_calendar_days"], 2)
        self.assertEqual(features["brent_age_calendar_days"], 5)
        self.assertEqual(features["brent_wti_spread_usd"], 5)
        self.assertEqual(features["spread_age_calendar_days"], 6)
        self.assertEqual(provenance["spread"]["observation_date"], "2020-05-07")
        self.assertTrue(eligible["wti_brent"])
        next_day = build_features(wti, brent, date(2020, 5, 14))[0]
        self.assertEqual(next_day["brent_latest_price"], 100)
        self.assertEqual(next_day["brent_wti_spread_usd"], 80)

    def test_historical_context_outside_calendar_keeps_price_and_null_windows(self):
        inputs = [series(identifier, [
            row(identifier, "2010-12-31", "2011-04-06", "70")])
            for identifier in ("DCOILWTICO", "DCOILBRENTEU")]
        features, provenance, _ = build_features(*inputs, date(2011, 4, 7))
        self.assertEqual(features["wti_latest_price"], 70)
        self.assertEqual(features["wti_age_calendar_days"],
                         (date(2011, 4, 7) - date(2010, 12, 31)).days)
        self.assertEqual(provenance["wti"]["window_status"], "outside_calendar_range")
        for column in ("wti_change_5_business_days_usd", "wti_change_20_business_days_usd",
                       "wti_change_std_20_business_days_usd", "wti_coverage_20_business_days"):
            self.assertIsNone(features[column])

    def test_negative_prices_produce_finite_dollar_features(self):
        inputs = []
        anchor = date(2020, 4, 20)
        for identifier in ("DCOILWTICO", "DCOILBRENTEU"):
            rows = []
            for offset in range(21):
                value = -36.98 if offset == 0 and identifier == "DCOILWTICO" else 20 + offset
                rows.append(row(identifier, add_business_days(anchor, -offset).isoformat(),
                                "2020-04-21", str(value)))
            inputs.append(series(identifier, rows))
        features, _, eligible = build_features(*inputs, date(2020, 4, 22))
        self.assertEqual(set(features), set(FEATURE_COLUMNS))
        self.assertEqual(features["wti_latest_price"], -36.98)
        self.assertAlmostEqual(features["wti_change_5_business_days_usd"], -61.98)
        self.assertTrue(all(value is not None and math.isfinite(value)
                            for value in features.values()))
        self.assertTrue(eligible["complete_wti_brent_features"])

    def test_active_withdrawal_does_not_supply_lookback_or_spread(self):
        wti = series("DCOILWTICO", [
            row("DCOILWTICO", "2020-05-04", "2020-05-05", "25", "2020-05-11"),
            row("DCOILWTICO", "2020-05-04", "2020-05-12", ""),
            row("DCOILWTICO", "2020-05-11", "2020-05-12", "30")])
        brent = series("DCOILBRENTEU", [
            row("DCOILBRENTEU", "2020-05-04", "2020-05-05", "29")])
        features, provenance, eligible = build_features(wti, brent, date(2020, 5, 13))
        self.assertIsNone(features["wti_change_5_business_days_usd"])
        self.assertIsNone(features["brent_wti_spread_usd"])
        self.assertIsNone(provenance["spread"])
        self.assertFalse(eligible["wti_brent"])
        deleted = next(ref for ref in provenance["wti"]["input_versions"]
                       if ref["observation_date"] == "2020-05-04")
        self.assertEqual(deleted["status"], "missing")


if __name__ == "__main__":
    unittest.main()

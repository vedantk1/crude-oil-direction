"""Unfitted, point-in-time features with an explicit model-input allowlist."""

from datetime import date, timedelta
import statistics

from .availability import PointInTimeSeries, Version
from .calendar import add_business_days


SERIES_FEATURES = (
    "latest_price", "age_calendar_days", "change_5_business_days_usd",
    "change_20_business_days_usd", "change_std_20_business_days_usd",
    "coverage_20_business_days",
)
WTI_FEATURE_COLUMNS = tuple("wti_" + name for name in SERIES_FEATURES)
FEATURE_COLUMNS = WTI_FEATURE_COLUMNS + tuple(
    "brent_" + name for name in SERIES_FEATURES
) + ("brent_wti_spread_usd", "spread_age_calendar_days")
FEATURE_SETS = {"wti": WTI_FEATURE_COLUMNS, "wti_brent": FEATURE_COLUMNS}


def reference(observation: date, version: Version | None) -> dict:
    """Keep source references outside model inputs; never export future ends."""
    status = ("not_in_archive" if version is None else
              "quarantined" if version.quarantined else
              "missing" if version.value is None else "numeric")
    return {
        "observation_date": observation.isoformat(),
        "vintage_start": version.vintage_start.isoformat() if version else None,
        "status": status,
    }


def _one_series(series: PointInTimeSeries, cutoff: date, forecast: date):
    state = series.state(cutoff)
    usable = series.usable_state(cutoff)
    values = dict.fromkeys(SERIES_FEATURES)
    provenance = {"series_id": series.series_id, "latest_observation_date": None,
                  "window_status": "no_numeric_observation", "input_versions": []}
    if not usable:
        return values, provenance, usable
    latest = max(usable)
    values["latest_price"] = usable[latest].value
    values["age_calendar_days"] = (forecast - latest).days
    provenance["latest_observation_date"] = latest.isoformat()
    provenance["input_versions"] = [reference(latest, usable[latest])]
    try:
        window = [add_business_days(latest, -n) for n in range(21)]
    except ValueError:
        # The archive may contain pre-calendar historical context. Preserve
        # the price/age, but do not invent a calendar outside its bounds.
        provenance["window_status"] = "outside_calendar_range"
        return values, provenance, usable
    provenance["input_versions"] = [reference(day, state.get(day)) for day in window]
    count = sum(day in usable for day in window)
    provenance["window_status"] = "complete" if count == 21 else "missing_observations"
    values["coverage_20_business_days"] = count / 21
    for length in (5, 20):
        if window[length] in usable:
            values[f"change_{length}_business_days_usd"] = (
                usable[latest].value - usable[window[length]].value
            )
    if count == 21:
        changes = [usable[a].value - usable[b].value
                   for a, b in zip(window, window[1:])]
        values["change_std_20_business_days_usd"] = statistics.pstdev(changes)
    return values, provenance, usable


def build_features(wti: PointInTimeSeries, brent: PointInTimeSeries,
                   forecast_date: date) -> tuple[dict, dict, dict]:
    """Use archive states through the calendar day preceding issuance."""
    cutoff = forecast_date - timedelta(days=1)
    features, provenance, states = {}, {}, {}
    for name, series in (("wti", wti), ("brent", brent)):
        values, refs, usable = _one_series(series, cutoff, forecast_date)
        features.update({name + "_" + key: value for key, value in values.items()})
        provenance[name] = refs
        states[name] = usable
    common = states["wti"].keys() & states["brent"].keys()
    features["brent_wti_spread_usd"] = None
    features["spread_age_calendar_days"] = None
    provenance["spread"] = None
    if common:
        day = max(common)
        features["brent_wti_spread_usd"] = states["brent"][day].value - states["wti"][day].value
        features["spread_age_calendar_days"] = (forecast_date - day).days
        provenance["spread"] = {
            "observation_date": day.isoformat(),
            "wti": reference(day, states["wti"][day]),
            "brent": reference(day, states["brent"][day]),
        }
    eligibility = {
        "wti": features["wti_latest_price"] is not None,
        "wti_brent": all(features[key] is not None for key in
                         ("wti_latest_price", "brent_latest_price", "brent_wti_spread_usd")),
        "complete_wti_features": all(features[key] is not None for key in WTI_FEATURE_COLUMNS),
        "complete_wti_brent_features": all(features[key] is not None for key in FEATURE_COLUMNS),
    }
    assert set(features) == set(FEATURE_COLUMNS)
    return features, provenance, eligibility

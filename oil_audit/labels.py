"""Reconstruct target availability separately from forecast feature values."""

from datetime import date, timedelta

from .availability import PointInTimeSeries
from .calendar import next_business_day


def build_label_events(
    series: PointInTimeSeries, forecast_date: date, target_date: date
) -> list[dict]:
    """Describe each label state under next-business-day availability.

    Endpoints are exact dates; missing prices never move the horizon to another
    observation. Version starts collapsing onto one business day are resolved
    from the archive state at the end of the preceding calendar day. Event end
    dates are exclusive. Source version ends remain outside model features.
    """
    if type(forecast_date) is not date or type(target_date) is not date:
        raise TypeError("Forecast and target must be dates without time components")
    if target_date <= forecast_date:
        raise ValueError("Target date must follow forecast date")
    available_dates = {forecast_date}
    for endpoint in (forecast_date, target_date):
        for version in series.versions_by_observation.get(endpoint, []):
            transitions = [version.vintage_start]
            if version.vintage_end is not None:
                # Adjacent versions normally supply this same transition.
                # Include it independently so a finite final interval cannot
                # leave a formerly numeric label active beyond its source.
                transitions.append(version.vintage_end + timedelta(days=1))
            for transition in transitions:
                available_from = next_business_day(transition)
                if available_from >= forecast_date:
                    available_dates.add(available_from)

    events = []
    for available_from in sorted(available_dates):
        cutoff = available_from - timedelta(days=1)
        state = series.state(cutoff)
        origin = state.get(forecast_date)
        target = state.get(target_date)

        def numeric(version):
            return (version is not None and version.value is not None
                    and not version.quarantined
                    and version.observation_date <= cutoff)

        origin_ok, target_ok = numeric(origin), numeric(target)
        quarantined = any(version is not None and version.quarantined
                          for version in (origin, target))
        if quarantined:
            status = "quarantined"
        elif not origin_ok and not target_ok:
            status = "missing_both"
        elif not origin_ok:
            status = "missing_origin"
        elif not target_ok:
            status = "missing_target"
        else:
            status = "available"
        events.append({
            "forecast_date": forecast_date.isoformat(),
            "target_date": target_date.isoformat(),
            "available_from": available_from.isoformat(),
            "available_until": None,
            "label_status": status,
            "label": int(target.value > origin.value) if status == "available" else None,
            "origin_price": origin.value if origin_ok else None,
            "target_price": target.value if target_ok else None,
            "origin_vintage_start": origin.vintage_start.isoformat() if origin else None,
            "target_vintage_start": target.vintage_start.isoformat() if target else None,
        })
    for previous, current in zip(events, events[1:]):
        previous["available_until"] = current["available_from"]
    return events


def select_label(events: list[dict], as_of: date) -> dict | None:
    """Select the active label state, including explicit unavailable states."""
    if type(as_of) is not date:
        raise TypeError("Expected an as-of date without a time component")
    day = as_of.isoformat()
    for event in reversed(events):
        if event["available_from"] <= day and (
                event["available_until"] is None or day < event["available_until"]):
            return event
    return None

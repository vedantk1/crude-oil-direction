"""Fixed calendar window applied only after causal training eligibility."""

from datetime import date


ROLLING_WINDOW_YEARS = 4


def rolling_window_start(forecast_date: date) -> date:
    """Subtract four calendar years, clamping leap day to February 28."""
    try:
        return forecast_date.replace(year=forecast_date.year - ROLLING_WINDOW_YEARS)
    except ValueError:
        return forecast_date.replace(year=forecast_date.year - ROLLING_WINDOW_YEARS, day=28)


def select_rolling_training_rows(expanding_rows: list[dict], forecast_date: date) -> list[dict]:
    """Preserve the ordered causal rows within [four years earlier, issuance).

    The caller must first apply historical label availability and maturity.
    This helper only removes old examples; it does not select labels, inspect
    outcomes, discard missing features, or fit preprocessing. The caller also
    enforces the unchanged minimum training count and both label classes.
    """
    start = rolling_window_start(forecast_date)
    selected = []
    for row in expanding_rows:
        row_date = date.fromisoformat(row["forecast_date"])
        if row_date >= forecast_date:
            raise ValueError("Rolling training requires strictly earlier forecast dates")
        if row_date >= start:
            selected.append(row)
    return selected

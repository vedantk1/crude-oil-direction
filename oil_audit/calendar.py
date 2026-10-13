"""Deterministic operational dates, not an exchange trading calendar.

Use recurring US federal holidays with Monday-Friday observance. OPM's
rules put Saturday holidays on Friday and Sunday holidays on Monday:
https://www.opm.gov/policy-data-oversight/pay-leave/federal-holidays/
Juneteenth starts in 2021, as documented in OPM's Juneteenth fact sheet:
https://www.opm.gov/policy-data-oversight/pay-leave/federal-holidays/juneteenth-national-independence-day-holiday.pdf

This fixed policy supports 2011-2100 inclusive. Later years project the
same recurring rules; they do not assert future publication schedules.
"""

from datetime import date, timedelta
from functools import lru_cache


CALENDAR_ID = "us-federal-observed-2011-2100-v1"
CALENDAR_NOTES = (
    "Operational Monday-Friday calendar excluding recurring US federal "
    "observed holidays, with Juneteenth from 2021. Not a NYMEX trading "
    "calendar or evidence of source publication. Good Friday, exceptional "
    "closures and regional Inauguration Day are not excluded. Rules after "
    "the source snapshot are projections, subject to policy changes."
)
MIN_DATE = date(2011, 1, 1)
MAX_DATE = date(2100, 12, 31)
_DAY = timedelta(days=1)


def _validate_date(day: date) -> None:
    if type(day) is not date:
        raise TypeError("Expected a date without a time component")
    if not MIN_DATE <= day <= MAX_DATE:
        raise ValueError("Calendar supports dates from 2011-01-01 to 2100-12-31")


def _observed(day: date) -> date:
    if day.weekday() == 5:
        return day - _DAY
    if day.weekday() == 6:
        return day + _DAY
    return day


def _nth_weekday(year: int, month: int, weekday: int, occurrence: int) -> date:
    first = date(year, month, 1)
    return first + timedelta(days=(weekday - first.weekday()) % 7 + 7 * (occurrence - 1))


@lru_cache(maxsize=90)
def _holidays(year: int) -> frozenset[date]:
    # Include adjacent years because a Saturday January 1 is observed in
    # December of the previous year. Return only dates in the requested year.
    result = set()
    for holiday_year in (year - 1, year, year + 1):
        fixed = [(1, 1), (7, 4), (11, 11), (12, 25)]
        if holiday_year >= 2021:
            fixed.append((6, 19))
        result.update(_observed(date(holiday_year, month, day)) for month, day in fixed)
        result.update((
            _nth_weekday(holiday_year, 1, 0, 3),  # Martin Luther King Jr. Day
            _nth_weekday(holiday_year, 2, 0, 3),  # Washington's Birthday
            date(holiday_year, 5, 31)
            - timedelta(days=date(holiday_year, 5, 31).weekday()),  # Memorial Day
            _nth_weekday(holiday_year, 9, 0, 1),  # Labor Day
            _nth_weekday(holiday_year, 10, 0, 2),  # Columbus Day
            _nth_weekday(holiday_year, 11, 3, 4),  # Thanksgiving Day
        ))
    return frozenset(day for day in result if day.year == year)


def is_business_day(day: date) -> bool:
    """Whether a date is operational under this fixed calendar policy."""
    _validate_date(day)
    return day.weekday() < 5 and day not in _holidays(day.year)


def add_business_days(day: date, count: int) -> date:
    """Move by operational days, excluding the start; zero leaves it unchanged.

    Positive and negative counts are supported, including when the start is
    a weekend or holiday. Crossing the supported date range raises ValueError.
    """
    _validate_date(day)
    if type(count) is not int:
        raise TypeError("Business-day count must be an integer")
    direction = _DAY if count > 0 else -_DAY
    remaining = abs(count)
    while remaining:
        day += direction
        if is_business_day(day):
            remaining -= 1
    return day


def next_business_day(day: date) -> date:
    """Return the first operational date strictly after the input date."""
    return add_business_days(day, 1)


def previous_business_day(day: date) -> date:
    """Return the first operational date strictly before the input date."""
    return add_business_days(day, -1)

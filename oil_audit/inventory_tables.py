"""Parse archived EIA Table 4 inventory levels without inferring availability.

EIA Table 4 defines stocks and differences in million barrels in its companion
PDF; the CSV itself omits that unit. The parser accepts only its eight-column
layout and the two named crude-stock rows. It never uses today's series values.
Published differences use unrounded numbers, whereas displayed stock levels
have three decimals. Their difference can therefore disagree by 0.001 million
barrels. Keep both figures and permit an absolute residual of at most 0.00101.
"""

import csv
from datetime import date, datetime, timedelta
from decimal import Decimal, InvalidOperation
import io
import math
import re


SERIES_LABELS = {
    "Commercial (Excluding SPR)": "commercial_crude",
    "Cushing": "cushing",
}
ROUNDING_TOLERANCE = Decimal("0.00101")
_NUMBER = re.compile(r"[+-]?(?:\d+|\d{1,3}(?:,\d{3})+)(?:\.\d+)?")
_DATE = re.compile(r"\d{1,2}/\d{1,2}/\d{2}")


def _header_date(value: str) -> date:
    if not _DATE.fullmatch(value):
        raise ValueError("Unexpected Table 4 date header: " + value)
    try:
        return datetime.strptime(value, "%m/%d/%y").date()
    except ValueError as exc:
        raise ValueError("Invalid Table 4 date header: " + value) from exc


def _number(value: str, field: str) -> Decimal:
    if not _NUMBER.fullmatch(value):
        raise ValueError("Missing, flagged or invalid Table 4 numeric value: " + field)
    try:
        result = Decimal(value.replace(",", ""))
    except InvalidOperation as exc:
        raise ValueError("Invalid Table 4 numeric value: " + field) from exc
    if not result.is_finite() or not math.isfinite(float(result)):
        raise ValueError("Non-finite Table 4 numeric value: " + field)
    return result


def read_table4(payload: bytes, release_date: date,
                expected_observation_date: date | None = None) -> dict:
    """Return dates and both named stocks from one actual historical release.

    ``release_date`` must come from the issue/index's release metadata, not its
    URL directory: the delayed June 17, 2022 issue was released June 29, 2022.
    Dates in the result are date objects. Values are floats computed through
    Decimal arithmetic; rounding_residual is published minus rounded change.
    Comparison-year columns are validated as date headers but not exported.
    Duplicate 'Percent Change' headers are intentional, so use csv.reader.
    """
    if type(release_date) is not date or (
            expected_observation_date is not None and type(expected_observation_date) is not date):
        raise ValueError("Release and expected observation dates must be date objects")
    if not isinstance(payload, bytes):
        raise ValueError("Table 4 payload must be bytes")
    try:
        decoded = payload.decode("utf-8-sig")
    except UnicodeDecodeError:
        try:
            decoded = payload.decode("cp1252")
        except UnicodeDecodeError as exc:
            raise ValueError("Unsupported Table 4 text encoding") from exc
    try:
        rows = list(csv.reader(io.StringIO(decoded), strict=True))
    except csv.Error as exc:
        raise ValueError("Malformed Table 4 CSV") from exc
    if not rows:
        raise ValueError("Empty Table 4 CSV")
    header = rows[0]
    if (len(header) != 8 or header[0] != "STUB_1" or header[3] != "Difference"
            or header[5] != "Percent Change" or header[7] != "Percent Change"):
        raise ValueError("Unsupported Table 4 header layout")
    current, previous, year_ago, two_years_ago = [_header_date(header[index]) for index in (1, 2, 4, 6)]
    if current.weekday() != 4 or previous.weekday() != 4 or current - previous != timedelta(days=7):
        raise ValueError("Table 4 current and previous dates must be consecutive Fridays")
    if not two_years_ago < year_ago < previous:
        raise ValueError("Table 4 comparison dates must precede the previous week")
    if current >= release_date:
        raise ValueError("Table 4 observation must precede its actual release date")
    if expected_observation_date is not None and current != expected_observation_date:
        raise ValueError("Table 4 observation differs from issue/index metadata")
    series = {}
    for row in rows[1:]:
        if not row or row[0] not in SERIES_LABELS:
            continue
        name = SERIES_LABELS[row[0]]
        if name in series:
            raise ValueError("Duplicate required Table 4 row: " + row[0])
        if len(row) != 8:
            raise ValueError("Unsupported Table 4 row layout: " + row[0])
        level = _number(row[1], name + " current")
        prior = _number(row[2], name + " previous")
        change = _number(row[3], name + " published change")
        if level < 0 or prior < 0:
            raise ValueError("Table 4 stock levels must be nonnegative: " + name)
        rounded_change = level - prior
        residual = change - rounded_change
        if abs(residual) > ROUNDING_TOLERANCE:
            raise ValueError("Table 4 published change exceeds rounding tolerance: " + name)
        series[name] = {
            "current_million_barrels": float(level),
            "previous_million_barrels": float(prior),
            "published_change_million_barrels": float(change),
            "rounded_level_change_million_barrels": float(rounded_change),
            "rounding_residual": float(residual),
        }
    if set(series) != set(SERIES_LABELS.values()):
        raise ValueError("Missing required Table 4 commercial-crude or Cushing row")
    return {"current_date": current, "previous_date": previous, "series": series}

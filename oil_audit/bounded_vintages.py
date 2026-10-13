"""Read bounded ALFRED intervals without decoding later price values.

Callers authenticate the source manifest and ZIP bytes before using these
records. This module only enforces the inclusive observation/vintage boundary.
"""

import csv
from datetime import date, timedelta
from pathlib import Path
import re
import zipfile

from .vintages import SERIES


def _token(raw: bytes) -> bytes:
    return raw[1:-1] if raw.startswith(b'"') and raw.endswith(b'"') else raw


def _date(raw: bytes) -> date:
    value = _token(raw)
    if not re.fullmatch(rb"\d{4}-\d{2}-\d{2}", value):
        raise ValueError("Invalid bounded ALFRED date gate")
    return date.fromisoformat(value.decode("ascii"))


def _records_from_zip(series_id, path, maximum_as_of):
    fields = ("period_start_date", series_id, "realtime_start_date", "realtime_end_date")
    boundary = maximum_as_of + timedelta(days=1)
    with zipfile.ZipFile(path) as archive:
        members = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if len(members) != 1:
            raise ValueError("Expected exactly one ALFRED interval CSV per ZIP")
        with archive.open(members[0]) as stream:
            header = stream.readline().rstrip(b"\r\n").removeprefix(b"\xef\xbb\xbf")
            if tuple(_token(field).decode("ascii") for field in header.split(b",")) != fields:
                raise ValueError("Unexpected ALFRED interval CSV schema")
            for raw in stream:
                line = raw.rstrip(b"\r\n")
                try:
                    observed_raw, remainder = line.split(b",", 1)
                    _, start_raw, end_raw = remainder.rsplit(b",", 2)
                except ValueError as exc:
                    raise ValueError("Malformed ALFRED interval date fields") from exc
                observed, start = _date(observed_raw), _date(start_raw)
                if observed > maximum_as_of or start > maximum_as_of:
                    continue
                end = _date(end_raw) if _token(end_raw) else None
                if end is not None and end < start:
                    raise ValueError("ALFRED version end precedes its start")
                # CSV/Unicode/numeric value interpretation starts only after
                # both source-date gates. Future intervals never reach here.
                try:
                    values = next(csv.reader([line.decode("utf-8")], strict=True))
                except (UnicodeDecodeError, csv.Error) as exc:
                    raise ValueError("Invalid bounded ALFRED CSV record") from exc
                if len(values) != len(fields):
                    raise ValueError("Malformed bounded ALFRED CSV fields")
                row = dict(zip(fields, values))
                if row[fields[0]] != observed.isoformat() or row[fields[2]] != start.isoformat():
                    raise ValueError("ALFRED date gates differ from decoded record")
                row["realtime_end_date"] = end.isoformat() if end is not None and end < boundary else ""
                yield row


def bounded_records(series_id: str, data_dir: Path, maximum_as_of: date) -> list[dict]:
    """Return deduplicated records with observation/start no later than max.

    Future interval ends become open within this bounded view. Missing values,
    withdrawals, and predated versions remain explicit for the existing PIT
    validator; this function never substitutes an older numeric observation.
    """
    if series_id not in SERIES or type(maximum_as_of) is not date:
        raise ValueError("Bounded ALFRED reading requires a known series and a date")
    paths = sorted(Path(data_dir).glob(series_id + "-realtime-part*.zip"))
    if not paths:
        raise ValueError("No bounded ALFRED interval ZIPs for " + series_id)
    unique = {}
    for path in paths:
        for row in _records_from_zip(series_id, path, maximum_as_of):
            unique[tuple(row.items())] = row
    return sorted(unique.values(), key=lambda row: (row["period_start_date"], row["realtime_start_date"]))

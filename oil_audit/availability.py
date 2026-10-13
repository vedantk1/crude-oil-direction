"""Validated, date-level reconstruction of ALFRED version intervals.

The lookup describes the archive's recorded state, not an intraday publication
time. Consumers apply their forecast issuance delay separately.
"""

from bisect import bisect_right
from dataclasses import dataclass
from datetime import date, timedelta
import math


@dataclass(frozen=True)
class Version:
    observation_date: date
    vintage_start: date
    vintage_end: date | None
    value: float | None

    @property
    def quarantined(self) -> bool:
        """A predated version remains invalid even after its observation date."""
        return self.vintage_start < self.observation_date


def _date(value: str, field: str) -> date:
    try:
        return date.fromisoformat(value)
    except (TypeError, ValueError) as error:
        raise ValueError("Invalid ALFRED " + field) from error


class PointInTimeSeries:
    """Index complete, inclusive source intervals without filling missing data.

    ``state`` preserves missing and quarantined records for audit purposes.
    ``usable_state`` filters the *active* version; it never searches backward
    for an older numeric value after a missing-value revision.
    """

    def __init__(self, series_id: str, rawrows):
        self.series_id = series_id
        self.versions_by_observation: dict[date, list[Version]] = {}
        for row in rawrows:
            try:
                observed = _date(row["period_start_date"], "observation date")
                start = _date(row["realtime_start_date"], "version start")
                end = (_date(row["realtime_end_date"], "version end")
                       if row["realtime_end_date"] else None)
                raw_value = row[series_id]
            except KeyError as error:
                raise ValueError("Missing ALFRED field: " + str(error)) from error
            try:
                value = None if raw_value in (None, "", ".") else float(raw_value)
            except (TypeError, ValueError) as error:
                raise ValueError("Invalid ALFRED numeric value") from error
            if value is not None and not math.isfinite(value):
                raise ValueError("Non-finite ALFRED numeric value")
            if end is not None and end < start:
                raise ValueError("ALFRED version end precedes its start")
            version = Version(observed, start, end, value)
            self.versions_by_observation.setdefault(observed, []).append(version)
        if not self.versions_by_observation:
            raise ValueError("ALFRED series contains no version records")

        self._starts = {}
        all_starts = []
        for observed, versions in self.versions_by_observation.items():
            versions.sort(key=lambda version: version.vintage_start)
            for previous, current in zip(versions, versions[1:]):
                if (previous.vintage_end is None
                        or previous.vintage_end >= current.vintage_start):
                    raise ValueError("Overlapping ALFRED version intervals")
                if previous.vintage_end + timedelta(days=1) != current.vintage_start:
                    raise ValueError("Gap in ALFRED version intervals")
            self._starts[observed] = [version.vintage_start for version in versions]
            all_starts.extend(self._starts[observed])
        self.first_vintage = min(all_starts)
        self.last_vintage = max(all_starts)

    def state(self, cutoff: date) -> dict[date, Version]:
        """Return active versions at an inclusive date-level archive cutoff."""
        if type(cutoff) is not date:
            raise TypeError("Expected a cutoff date without a time component")
        result = {}
        for observed, versions in self.versions_by_observation.items():
            position = bisect_right(self._starts[observed], cutoff) - 1
            if position < 0:
                continue
            version = versions[position]
            if version.vintage_end is None or cutoff <= version.vintage_end:
                result[observed] = version
        return result

    def usable_state(self, cutoff: date) -> dict[date, Version]:
        """Return numeric, nonquarantined observations dated through cutoff.

        Same-date observations are allowed here because the forecast pipeline
        issues on a subsequent business day. This method itself adds no delay.
        """
        return {observed: version for observed, version in self.state(cutoff).items()
                if observed <= cutoff and version.value is not None
                and not version.quarantined}

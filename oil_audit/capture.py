"""Conservative replay bounds from saved download provenance."""

from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path


CAPTURE_POLICY = (
    "Every forecast and label eligibility date must be within captured history. "
    "For timestamped files use the recorded UTC completion time. For date-only "
    "records with unspecified timezone, use 00:00 UTC on the preceding calendar "
    "date as a deliberately early lower bound on capture time, not an invented "
    "retrieval timestamp. Apply the earliest bound independently to each series."
)


def capture_bounds(records: list[dict], series_ids: tuple[str, ...]) -> dict[str, datetime]:
    result = {}
    for series_id in series_ids:
        bounds = []
        for record in records:
            if not Path(record["path"]).name.startswith(series_id + "-"):
                continue
            if record.get("retrieved_at_utc"):
                bound = datetime.fromisoformat(record["retrieved_at_utc"])
                if bound.tzinfo is None:
                    raise ValueError("Capture timestamp must include a timezone")
                bound = bound.astimezone(timezone.utc)
            elif record.get("retrieval_date"):
                day = date.fromisoformat(record["retrieval_date"]) - timedelta(days=1)
                bound = datetime.combine(day, time(), tzinfo=timezone.utc)
            else:
                raise ValueError("Capture provenance missing for " + series_id)
            bounds.append(bound)
        if not bounds:
            raise ValueError("No capture provenance for " + series_id)
        result[series_id] = min(bounds)
    return result

"""Select historical training examples without fitting or imputing anything."""

from collections import defaultdict
from datetime import date

from .features import FEATURE_SETS
from .labels import select_label


def training_rows(feature_rows: list[dict], label_events: list[dict],
                  as_of: date, maximum_as_of: date,
                  feature_set: str = "wti") -> list[dict]:
    """Use only labels active under the 18:00 New York convention on as_of.

    Optional missing inputs stay null. Any later imputation or scaling must be
    fitted on training folds. The maximum prevents silent archive extrapolation.
    """
    if as_of > maximum_as_of:
        raise ValueError("Training cutoff exceeds this snapshot's supported replay")
    columns = FEATURE_SETS[feature_set]
    by_id = defaultdict(list)
    for event in label_events:
        by_id[event["row_id"]].append(event)
    for events in by_id.values():
        events.sort(key=lambda event: event["available_from"])
    result = []
    for row in feature_rows:
        if row["forecast_date"] > as_of.isoformat() or not row["eligibility"][feature_set]:
            continue
        label = select_label(by_id[row["row_id"]], as_of)
        if label is None or label["label_status"] != "available":
            continue
        result.append({
            "row_id": row["row_id"], "forecast_date": row["forecast_date"],
            "features": {column: row["features"][column] for column in columns},
            "label": label["label"],
        })
    return result

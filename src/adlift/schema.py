"""Dataset schema definitions and column specifications for AdLift."""

from typing import List, Dict

# The 22 raw columns present in data/raw/criteo_attribution_dataset.tsv.gz
RAW_COLUMNS: List[str] = [
    "timestamp",
    "uid",
    "campaign",
    "conversion",
    "conversion_timestamp",
    "conversion_id",
    "attribution",
    "click",
    "click_pos",
    "click_nb",
    "cost",
    "cpo",
    "time_since_last_click",
    "cat1",
    "cat2",
    "cat3",
    "cat4",
    "cat5",
    "cat6",
    "cat7",
    "cat8",
    "cat9",
]

# Raw data types observed directly from criteo_attribution_dataset.tsv.gz
RAW_COLUMN_TYPES: Dict[str, str] = {
    "timestamp": "int64",
    "uid": "int64",
    "campaign": "int64",
    "conversion": "int64",
    "conversion_timestamp": "int64",
    "conversion_id": "int64",
    "attribution": "int64",
    "click": "int64",
    "click_pos": "int64",
    "click_nb": "int64",
    "cost": "float64",
    "cpo": "float64",
    "time_since_last_click": "int64",
    "cat1": "int64",
    "cat2": "int64",
    "cat3": "int64",
    "cat4": "int64",
    "cat5": "int64",
    "cat6": "int64",
    "cat7": "int64",
    "cat8": "int64",
    "cat9": "int64",
}

# The target prediction label
TARGET_LABEL: str = "click"

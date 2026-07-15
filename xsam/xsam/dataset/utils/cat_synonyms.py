"""Category name synonyms for open-vocabulary panoptic training.

Only replaces prompt text; category ids / mask labels stay unchanged.
"""

from __future__ import annotations

import random
from typing import Dict, List, Sequence

# key: training taxonomy name (as in annotations_train.json)
PANO_CAT_SYNONYMS: Dict[str, List[str]] = {
    "residential_area": [
        "residential_area",
        "residential area",
        "building",
        "buildings",
        "house",
    ],
    "industrial_area": [
        "industrial_area",
        "industrial area",
        "building",
        "factory",
    ],
    "small_vehicle": [
        "small_vehicle",
        "small vehicle",
        "car",
        "automobile",
        "vehicle",
    ],
    "large_vehicle": [
        "large_vehicle",
        "large vehicle",
        "truck",
        "bus",
    ],
    "road": [
        "road",
        "impervious surface",
        "paved surface",
        "pavement",
    ],
    "grassland": [
        "grassland",
        "low vegetation",
        "grass",
        "vegetation",
    ],
    "forest": [
        "forest",
        "tree",
        "trees",
        "woodland",
    ],
    "bare_land": [
        "bare_land",
        "bare land",
        "bare soil",
        "soil",
    ],
    "river": [
        "river",
        "water",
        "water body",
    ],
    "lake_and_sea": [
        "lake_and_sea",
        "lake",
        "sea",
        "water",
        "water body",
    ],
}


def _lookup(name: str, synonyms: Dict[str, Sequence[str]]) -> Sequence[str] | None:
    if name in synonyms:
        return synonyms[name]
    key = name.strip().replace(" ", "_")
    return synonyms.get(key)


def sample_cat_synonym(
    name: str,
    synonyms: Dict[str, Sequence[str]] | None = None,
    keep_prob: float = 0.5,
) -> str:
    """Sample a display name.

    With probability ``keep_prob`` keep the exact annotation name.
    Otherwise sample uniformly from the synonym list (includes canonical).
    """
    table = synonyms if synonyms is not None else PANO_CAT_SYNONYMS
    cands = _lookup(name, table)
    if not cands:
        return name
    if random.random() < keep_prob:
        return name
    return random.choice(list(cands))


def apply_cat_synonyms(
    names: Sequence[str],
    synonyms: Dict[str, Sequence[str]] | None = None,
    keep_prob: float = 0.5,
) -> List[str]:
    return [sample_cat_synonym(n, synonyms=synonyms, keep_prob=keep_prob) for n in names]

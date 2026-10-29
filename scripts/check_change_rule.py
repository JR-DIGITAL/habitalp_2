#!/usr/bin/env python
"""Unit-style check of get_change_array for the dwarf-forest rule (R2-S5).

Run: python scripts/check_change_rule.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.trainers.utils import get_change_array

# (from class, to class, expected change class)
CASES = [
    (20, 11, 8),  # dwarf forest -> mature forest: other transition
    (20, 6, 8),  # dwarf forest -> young growth: other transition
    (11, 20, 8),  # mature forest -> dwarf forest: other transition
    (21, 6, 6),  # grassland -> young growth: early forest establishment
    (22, 11, 6),  # alpine -> mature forest: early forest establishment
    (19, 6, 6),  # clearcut -> young growth: early forest establishment
    (20, 19, 7),  # dwarf forest -> clearcut: clearcut
    (20, 20, 0),  # no change
    (0, 20, 255),  # nodata
]


def test_change_rule():
    from_array = np.array([[c[0] for c in CASES]], dtype=np.uint8)
    to_array = np.array([[c[1] for c in CASES]], dtype=np.uint8)
    out = get_change_array(from_array, to_array)[0]
    for (f, t, expected), got in zip(CASES, out):
        assert got == expected, f"{f} -> {t}: expected {expected}, got {got}"


if __name__ == "__main__":
    test_change_rule()
    print(f"OK: {len(CASES)} transitions")

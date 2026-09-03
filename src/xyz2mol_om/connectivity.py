"""T1 — presence of a bond inside a ligand (per-element-pair distance threshold).

⚠️ **Ported from `ognm-bh-workspace/code/analysis/scratch/260830_fit_t10_charge.py`**
(2026-09-03). Function bodies were moved **verbatim** — the decision rules are unchanged.
"""

# ruff: noqa: E501
from __future__ import annotations

import csv

from .config import DATA, RCOV, USE_DINT


DINT_CSV = DATA / "d_int.csv"

def load_dint():
    d, fb = {}, 2.0542
    if DINT_CSV.exists():
        for r in csv.DictReader(open(DINT_CSV)):
            if r["X"] == "*":
                fb = float(r["d_int"])
            else:
                d[tuple(sorted((r["X"], r["Y"])))] = float(r["d_int"])
    return d, fb

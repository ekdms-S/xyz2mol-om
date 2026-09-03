"""T1 — 배위자 내부 결합 유무 (원소쌍별 거리 임계).

⚠️ **`ognm-bh-workspace/code/analysis/scratch/260830_fit_t10_charge.py` 에서 이관한 코드다**
(2026-09-03). 함수 본문은 **그대로** 옮겼다 — 판정 규칙을 바꾸지 않는다.
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

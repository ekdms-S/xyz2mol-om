"""T3 4클래스 거리 우도 — 적합·조회. 사전확률은 **끝점 내부차수 셀**로 조건화한다.

⚠️ **`ognm-bh-workspace/code/analysis/scratch/260830_fit_t10_charge.py` 에서 이관한 코드다**
(2026-09-03). 함수 본문은 **그대로** 옮겼다 — 판정 규칙을 바꾸지 않는다.
"""

# ruff: noqa: E501
from __future__ import annotations

import collections
from pathlib import Path
import json

import numpy as np

from .config import DATA, LPCOND_NMIN


def deg_cell(el, a, b, deg):
    """`LPCOND` 셀 키 — 원소 정렬 순서에 맞춘 **끝점 내부차수쌍** (4 에서 자른다)."""
    da, db = min(deg.get(a, 0), 4), min(deg.get(b, 0), 4)
    return (db, da) if el[a] > el[b] else (da, db)

def fit_scores4(samples, n_min=300):
    """4클래스 거리 우도 적합 — **배포·CV 공용**. `LPCOND` 셀 사전확률까지 담는다.

    `samples[k] = (거리 리스트, 클래스 리스트, 셀 리스트)`  (셀은 `deg_cell` 값)
    반환 `{k: (med, scl, lp, {}, {}, lp_cell)}` — 6번째가 `{셀: {클래스: lnP}}`.
    ⚠️ 3·4번째(`rmed`·`rscl`)는 ROP 자리다. `USE_ROP` 를 안 쓰면 빈 dict 로 둔다.
    """
    out = {}
    for k, sam in samples.items():
        v, lab = np.array(sam[0]), np.array(sam[1])
        if len(v) < n_min:
            continue
        med, scl, lp = {}, {}, {}
        for c in range(4):
            m = lab == c
            if m.sum() >= 5:
                med[c] = float(np.median(v[m]))
                scl[c] = max(float(np.median(np.abs(v[m] - med[c]))) * 1.4826, 0.005)
                lp[c] = float(np.log(m.mean()))
        if len(med) < 2:
            continue
        lp_cell = {}
        if len(sam) >= 3 and sam[2] is not None:
            cnt = collections.Counter(zip(sam[2], sam[1]))
            tot = collections.Counter(sam[2])
            for cell, n in tot.items():
                if n < LPCOND_NMIN:
                    continue
                lp_cell[cell] = {c: float(np.log(max(cnt[(cell, c)], 0.5) / n)) for c in lp}
        out[k] = (med, scl, lp, {}, {}, lp_cell)
    return out


_SC4_CACHE = None


def load_scores4(path=None):
    """패키지에 실린 **적합 산출물**을 `predict_T3_EHT` 가 받는 형태로 읽는다.

    반환 `{(X, Y): (med, scl, lp, {}, {}, lp_cell)}` — `fit_scores4()` 출력과 같은 6-튜플.
    적합 조건은 파일의 `_meta` 에 있다 (train 26,075 · 원본 CSD `bond_type` · 기하 `ref_xtb2`).
    """
    global _SC4_CACHE
    if path is None and _SC4_CACHE is not None:
        return _SC4_CACHE
    p = Path(path) if path else DATA / "scores4.json"
    raw = json.loads(p.read_text())
    out = {}
    for k, v in raw.items():
        if k == "_meta":
            continue
        x, y = k.split("-")
        med = {int(c): float(u) for c, u in v["med"].items()}
        scl = {int(c): float(u) for c, u in v["scl"].items()}
        lp = {int(c): float(u) for c, u in v["lp"].items()}
        cell = {
            tuple(int(t) for t in ck.split(",")): {int(c): float(u) for c, u in cv.items()}
            for ck, cv in v.get("lp_cell", {}).items()
        }
        out[(x, y)] = (med, scl, lp, {}, {}, cell)
    if path is None:
        _SC4_CACHE = out
    return out


def scores4_meta(path=None):
    """적합 조건(`_meta`) 만 읽는다."""
    p = Path(path) if path else DATA / "scores4.json"
    return json.loads(p.read_text()).get("_meta", {})

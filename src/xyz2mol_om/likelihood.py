"""T3 4-class distance likelihood — fit and lookup. The prior is conditioned on the
**endpoint internal-degree cell**.

⚠️ **Ported from `ognm-bh-workspace/code/analysis/scratch/260830_fit_t10_charge.py`**
(2026-09-03). Function bodies were moved **verbatim** — the decision rules are unchanged.
"""

# ruff: noqa: E501
from __future__ import annotations

import collections
from pathlib import Path
import json

import numpy as np

from .config import DATA, LPCOND_NMIN


def deg_cell(el, a, b, deg):
    """`LPCOND` cell key — the **endpoint internal-degree pair** ordered to match the sorted
    element order (clipped at 4)."""
    da, db = min(deg.get(a, 0), 4), min(deg.get(b, 0), 4)
    return (db, da) if el[a] > el[b] else (da, db)

def fit_scores4(samples, n_min=300):
    """Fit the 4-class distance likelihood — **shared by the release and CV**. Also holds the
    `LPCOND` cell priors.

    `samples[k] = (distance list, class list, cell list)`  (cell = `deg_cell` value)
    Returns `{k: (med, scl, lp, {}, {}, lp_cell)}` — the 6th is `{cell: {class: lnP}}`.
    ⚠️ The 3rd and 4th (`rmed`, `rscl`) are the ROP slots. Left as empty dicts when
    `USE_ROP` is off.
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
    """Read the **fit artifact** shipped with the package in the form `predict_T3_EHT` expects.

    Returns `{(X, Y): (med, scl, lp, {}, {}, lp_cell)}` — the same 6-tuple as `fit_scores4()`.
    The fit conditions are in the file's `_meta` (train 26,075 · original CSD `bond_type` ·
    geometry `ref_xtb2`).
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
    """Read only the fit conditions (`_meta`)."""
    p = Path(path) if path else DATA / "scores4.json"
    return json.loads(p.read_text()).get("_meta", {})

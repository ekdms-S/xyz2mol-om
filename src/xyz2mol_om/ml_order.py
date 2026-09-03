"""T8 — M–L bond order (monotone Mayer threshold).

⚠️ **Ported from `ognm-bh-workspace/code/analysis/scratch/260830_fit_t10_charge.py`**
(2026-09-03). Function bodies were moved **verbatim** — the decision rules are unchanged.
"""

# ruff: noqa: E501
from __future__ import annotations

import csv
import os
from pathlib import Path

import numpy as np

from .config import CLS, DATA, T8FORM


B_ML_CSV = DATA / "b_ml_mayer.csv"
# for `lik1`/`thr` — all three forms live in one file (produced by 260901_export_t8_forms.py)
B_ML_FORMS_CSV = DATA / "b_ml_t8forms.csv"

def _load_b_ml_forms(form, path=None):
    """`b_ml_t8forms.csv` → ({(M,X): ("lik",med,scl,lp) | ("thr",t1,t2,k) | ("const",c)},
    fallback).

    `thr` score:  s(0)=0 · s(1)=k(w−t1) · s(2)=k(w−t1)+k(w−t2),  k = 1/scl_pool
    ⇒ the argmax is exactly the threshold rule, and the sign of the increment `s(c+1)−s(c)`
    used by the exact solution is right too.
    A class whose threshold is `inf` does not occur for that element pair (that is what the fit
    chose).
    """
    p = Path(path) if path else B_ML_FORMS_CSV
    mdl, fb = {}, 0
    if not p.exists():
        return mdl, fb
    acc = {}
    for r in csv.DictReader(open(p)):
        if r["M"] == "*":
            fb = int(r["const_cls"])
            continue
        k = (r["M"], r["X"])
        if r["class"] == "const":
            mdl[k] = ("const", int(r["const_cls"]))
            continue
        if r["class"] == "thr":
            if form == "thr":
                t1, t2 = (float(x) for x in r["thr"].split("|"))
                mdl[k] = ("thr", t1, t2, 1.0 / float(r["scl_pool"]))
            continue
        if form == "thr":
            continue
        c = CLS[r["class"]]
        med, scl, lp = acc.setdefault(k, ({}, {}, {}))
        med[c] = float(r["med_w"])
        scl[c] = float(r["scl_pool"]) if form == "lik1" else float(r["scl"])
        lp[c] = float(r["logprior"])
    for k, (med, scl, lp) in acc.items():
        if k in mdl:
            continue
        mdl[k] = ("const", max(lp, key=lp.get)) if len(med) < 2 else ("lik", med, scl, lp)
    return mdl, fb

def load_b_ml_mayer(path=None):
    """`b_ml_mayer.csv` → ({(M,X): ("lik", med, scl, lp) | ("const", c)}, global fallback c).

    A `const` row is a pair that has only one class, or whose sample is too thin (n < 60) to
    build a likelihood.
    """
    if T8FORM != "lik" and path is None:
        return _load_b_ml_forms(T8FORM)
    p = Path(path) if path else B_ML_CSV
    mdl, fb = {}, 0
    if not p.exists():
        return mdl, fb
    acc = {}
    for r in csv.DictReader(open(p)):
        if r["M"] == "*":
            fb = int(r["const_cls"])
            continue
        k = (r["M"], r["X"])
        if r["class"] == "const":
            mdl[k] = ("const", int(r["const_cls"]))
            continue
        c = CLS[r["class"]]
        med, scl, lp = acc.setdefault(k, ({}, {}, {}))
        med[c], scl[c], lp[c] = float(r["med_w"]), float(r["scl"]), float(r["logprior"])
    for k, (med, scl, lp) in acc.items():
        mdl[k] = ("const", max(lp, key=lp.get)) if len(med) < 2 else ("lik", med, scl, lp)
    return mdl, fb

def predict_T8(m_el, x_el, w, model, fallback=0):
    """M–L bond-order class (0 Single · 1 Double · 2 Triple).

    rule ⟺  argmax_c [ −|w − med[c]| / scl[c] + lp[c] ],  w = xtb GFN2 **Mayer** bond order.
    If that element pair has no likelihood, it falls back to that pair's `const`, then to the
    global fallback (Single).
    If `w` is missing (no wbo computed), the fallback is used.
    """
    e = model.get((m_el, x_el))
    if e is None or w is None:
        return e[1] if (e is not None and e[0] == "const") else fallback
    if e[0] == "const":
        return e[1]
    if e[0] == "thr":
        _, t1, t2, _k = e
        return 2 if w >= t2 else (1 if w >= t1 else 0)
    _, med, scl, lp = e
    return max(med, key=lambda c: -abs(w - med[c]) / scl[c] + lp[c])


def ml_order_scores(el, ml_pairs, wbo, bml_model=None, fb=None):
    """M–L order **score table** `{(m, x): {class: score}}` — used when the ④ exact solution
    optimizes M–L jointly.

    🔴 **Built in exactly one place** (2026-09-03). It used to be built separately by the CV
    script, the tool comparison, CRW and the library, and some of them did not build it at all
    (`ml_sc=None`), **pinning** the M–L order to the T8 argmax. That gives different answers for
    the same input.

    `wbo` {(metal, atom): Mayer w} · `bml_model`/`fb` = output of `load_b_ml_mayer()` (read
    directly if omitted)
    ⚠️ **The caller must remove haptic pairs beforehand** — haptic bonds get no order
    ([design doc] §3 5a).
    """
    if bml_model is None:
        bml_model, fb = load_b_ml_mayer()
    out = {}
    for m, x in ml_pairs:
        w = (wbo or {}).get((m, x), (wbo or {}).get((x, m)))
        ent = bml_model.get((el[m], el[x]))
        if ent is None or w is None or ent[0] == "const":
            c0 = ent[1] if (ent and ent[0] == "const") else fb
            out[(m, x)] = {c0: 0.0}
        elif ent[0] == "thr":
            _, t1, t2, kk = ent
            sm = {0: 0.0}
            if t1 != float("inf"):
                sm[1] = kk * (w - t1)
                if t2 != float("inf"):
                    sm[2] = sm[1] + kk * (w - t2)
            out[(m, x)] = sm
        else:
            _, med, scl, lp = ent
            out[(m, x)] = {c: -abs(w - med[c]) / scl[c] + lp[c] for c in med}
    return out


# ============================================================================
# T8 fallback — assign M–L orders **from distance alone**, without `wbo` (Mayer) (2026-09-03).
#
# Why: with `predict(..., wbo=None)` the Mayer path above degenerates entirely to the fallback
#      (`Single`). A per-element-pair distance model fills that gap.
#
# rule (adopted form = `thr`; `T8DISTFORM=lik` switches to the likelihood form)
#     Double or higher ⟺ d(M,X) <= t1(M,X)
#     Triple           ⟺ d(M,X) <= t2(M,X)      (t2 <= t1, both in Å)
#   A class with `t = -inf` is one the fit did not select for that element pair (= never predicted).
#   score form (the ④ exact solution needs increments):
#     s(Single)=0 · s(Double)=k*(t1-d) · s(Triple)=s(Double)+k*(t2-d),  k = 1/scl_pool
#   ⇒ the argmax is exactly the threshold rule above, and the sign of `s(c+1)-s(c)` is right too.
#   **Only the sign differs** from the Mayer version (`ml_order_scores`) — for Mayer a larger value
#   means a higher order, for distance a shorter one does.
#
# performance (reference labels CSD `bond_type` · geometry `ref_xtb2` · train sample 158,048 ·
#       refcode 5-fold **CV** · 2026-09-03 · reproduce with
#       `ognm-bh-workspace/code/analysis/scratch/260903_fit_t8_dist.py`):
#       variant                        Single   Double   Triple   accuracy   params
#       trivial (all Single)           0.9815   0.0000   0.0000     0.9636        0
#       distance monotone thr (here)   0.9915   0.6976   0.6515     0.9821      464
#       Mayer monotone thr (current)   0.9928   0.7318   0.7230     0.9846      464
#   ⚠️ **This measures T8 alone** — only orders were assigned, on the CSD reference M–L bonds.
#      Without `wbo` the T4 gate (bond presence) changes too, so this table says nothing about
#      whole-pipeline performance.
# ============================================================================

B_ML_DIST_CSV = DATA / "b_ml_dist.csv"
T8DISTFORM = os.environ.get("T8DISTFORM", "thr")


def load_b_ml_dist(path=None, form=None):
    """`b_ml_dist.csv` → ({(M,X): ("thr",t1,t2,k) | ("lik",med,scl,lp) | ("const",c)},
    global fallback c).

    `form="thr"` (default) gives a per-element-pair monotone threshold; `form="lik"` gives
    per-class (median, MAD, log prior).
    A `const` row is a pair that has only one class, or whose sample is too thin (n < 60) to
    build a model.
    ⚠️ `med_d`, `t1` and `t2` are in **Å**, and so is `scl_pool`.
    """
    form = form or T8DISTFORM
    p = Path(path) if path else B_ML_DIST_CSV
    mdl, fb = {}, 0
    if not p.exists():
        return mdl, fb
    acc = {}
    with open(p) as fh:
        for r in csv.DictReader(fh):
            if r["M"] == "*":
                fb = int(r["const_cls"])
                continue
            k = (r["M"], r["X"])
            if r["class"] == "const":
                mdl[k] = ("const", int(r["const_cls"]))
                continue
            if r["class"] == "thr":
                if form == "thr":
                    t1, t2 = (float(x) for x in r["thr"].split("|"))
                    mdl[k] = ("thr", t1, t2, 1.0 / float(r["scl_pool"]))
                continue
            if form == "thr":
                continue
            c = CLS[r["class"]]
            med, scl, lp = acc.setdefault(k, ({}, {}, {}))
            med[c], scl[c], lp[c] = float(r["med_d"]), float(r["scl"]), float(r["logprior"])
    for k, (med, scl, lp) in acc.items():
        if k in mdl:
            continue
        mdl[k] = ("const", max(lp, key=lp.get)) if len(med) < 2 else ("lik", med, scl, lp)
    return mdl, fb


def ml_order_scores_dist(el, ml_pairs, xyz, model=None, fb=None):
    """M–L order **score table** `{(m, x): {class: score}}` — the distance-fallback version of
    `ml_order_scores`.

    The return format is **identical** to `ml_order_scores` (it drops into the same slot).
    `xyz` = coordinate array (Å) · `model`/`fb` = output of `load_b_ml_dist()` (read directly if
    omitted)
    ⚠️ **The caller must remove haptic pairs beforehand** — haptic bonds get no order
    ([design doc] §3 5a).
    """
    if model is None:
        model, fb = load_b_ml_dist()
    out = {}
    for m, x in ml_pairs:
        d = float(np.linalg.norm(np.asarray(xyz[m], dtype=float) - np.asarray(xyz[x], dtype=float)))
        ent = model.get((el[m], el[x]))
        if ent is None or ent[0] == "const":
            c0 = ent[1] if (ent and ent[0] == "const") else fb
            out[(m, x)] = {c0: 0.0}
        elif ent[0] == "thr":
            _, t1, t2, kk = ent
            sm = {0: 0.0}
            if t1 != float("-inf"):  # `-inf` = this pair never yields Double
                sm[1] = kk * (t1 - d)
                if t2 != float("-inf"):
                    sm[2] = sm[1] + kk * (t2 - d)
            out[(m, x)] = sm
        else:
            _, med, scl, lp = ent
            out[(m, x)] = {c: -abs(d - med[c]) / scl[c] + lp[c] for c in med}
    return out


def predict_T8_dist(m_el, x_el, d, model, fallback=0):
    """M–L bond-order class (0 Single · 1 Double · 2 Triple) — the distance fallback on its own.

    rule ⟺  `Triple` if d <= t2 else `Double` if d <= t1 else `Single`  (`thr` form)
             argmax_c [ -|d - med[c]| / scl[c] + lp[c] ]                 (`lik` form)
    If that element pair has no model, it falls back to that pair's `const`, then to the global
    fallback (Single).
    """
    e = model.get((m_el, x_el))
    if e is None or d is None:
        return e[1] if (e is not None and e[0] == "const") else fallback
    if e[0] == "const":
        return e[1]
    if e[0] == "thr":
        _, t1, t2, _k = e
        return 2 if d <= t2 else (1 if d <= t1 else 0)
    _, med, scl, lp = e
    return max(med, key=lambda c: -abs(d - med[c]) / scl[c] + lp[c])

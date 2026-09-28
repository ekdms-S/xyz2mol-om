"""T8 — M–L bond-order scores from the Mayer bond order (the joint solve's Mayer consistency).
"""

# ruff: noqa: E501
from __future__ import annotations

import csv
from pathlib import Path
from ..config import CLS, DATA, T8FORM


B_ML_CSV = DATA / "b_ml_mayer.csv"
# for `lik1`/`thr` — all three forms live in one file
B_ML_FORMS_CSV = DATA / "b_ml_t8forms.csv"

def _load_b_ml_forms(form, path=None):
    """`b_ml_t8forms.csv` → ({(M,X): ("lik",med,scl,lp) | ("thr",t1,t2,k) | ("const",c)},
    fallback).

    `thr` score:  s(0)=0 · s(1)=k(w−t1) · s(2)=k(w−t1)+k(w−t2),  k = 1/scl_pool
    ⇒ the argmax is exactly the threshold rule, and the sign of the increment `s(c+1)−s(c)`
    used by the exact solution is right too.
    A class whose threshold is `inf` does not occur for that element pair (a fit result, not a
    rule).
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


def ml_order_scores(el, ml_pairs, wbo, bml_model=None, fb=None):
    """M–L order **score table** `{(m, x): {class: score}}` — used when the ④ exact solution
    optimizes M–L jointly.

    🔴 **Built in exactly one place.** A caller that builds its own table, or passes
    `ml_sc=None` (which **pins** the M–L order to the T8 argmax), gets different answers for the
    same input.

    `wbo` {(metal, atom): Mayer w} · `bml_model`/`fb` = output of `load_b_ml_mayer()` (read
    directly if omitted)
    ⚠️ **The caller must remove haptic pairs beforehand** — haptic bonds get no order
    (`docs/PIPELINE.md`).
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

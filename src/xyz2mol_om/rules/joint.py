"""Shared pieces of the joint bond-order solve (`JOINT`, `rules.joint2`): the MILP builder
(`_Model`), the ③ score reading (`order_scores`), the oxidation-state prior (`os_prior_cost`) and
the unpaired-electron pricing (`_radical_delta`).

⚠️ `scipy` is imported inside `_Model.solve` and nowhere else — with `JOINT` off it is not a
dependency.
"""

from __future__ import annotations

import math

from ..config import DATA, FULL, VAL

PERIOD2 = {"B", "C", "N", "O", "F"}
# f-block centres: (valence electrons, shell size) — `OS = valence - f`
_FSHELL = {"La": (3, 14), "Ce": (4, 14)}
# Pauling electronegativity, for the radical-site tie-break only
_EN = {"H": 2.20, "B": 2.04, "C": 2.55, "N": 3.04, "O": 3.44, "F": 3.98, "Si": 1.90, "P": 2.19,
       "S": 2.58, "Cl": 3.16, "As": 2.18, "Se": 2.55, "Br": 2.96, "Te": 2.10, "I": 2.66}
_PRIOR = None


def os_prior_cost(el, lo, hi):
    """`{os: cost}` over `[lo, hi]` — `-ln(p / p_max)` of the element's state frequency in the
    train-split CSD names, add-one smoothed. An element with no counts costs nothing."""
    global _PRIOR
    if _PRIOR is None:
        import json

        _PRIOR = json.loads((DATA / "os_prior.json").read_text())
    cnt = {int(k): v for k, v in _PRIOR.get(el, {}).items()}
    if not cnt:
        return dict.fromkeys(range(lo, hi + 1), 0.0)
    p = {v: cnt.get(v, 0) + 1 for v in range(lo, hi + 1)}
    top = max(p.values())
    return {v: math.log(top / n) for v, n in p.items()}


def order_scores(sc_e):
    """`{1: s1, 2: s2, 3: s3}` from the ③ 4-class scores. `None` = that order is not allowed.

    A `Conj` score supports order 1 and order 2 alike — a bond of aromatic length is equally
    at home as either end of a Kekule alternation, and the valence constraints decide which.
    """
    conj = sc_e.get(3)

    def best(*vals):
        vals = [v for v in vals if v is not None]
        return max(vals) if vals else None

    s1 = best(sc_e.get(0), conj)
    return {1: 0.0 if s1 is None else s1, 2: best(sc_e.get(1), conj), 3: sc_e.get(2)}


class _Model:
    """A MILP under construction: minimise `cost · x` subject to `lo <= A x <= hi`."""

    def __init__(self):
        self.cost, self.bnd, self.integer = [], [], []
        self.rows, self.lo, self.hi = [], [], []

    def var(self, cost, ub=1.0, integer=True, lb=0.0):
        self.cost.append(cost)
        self.bnd.append((lb, ub))
        self.integer.append(1 if integer else 0)
        return len(self.cost) - 1

    def row(self, coefs, lo, hi):
        self.rows.append(coefs)
        self.lo.append(lo)
        self.hi.append(hi)

    def solve(self):
        """`(x, objective)` or `None` when there is no solution. A solve stopped by the time limit
        (`JOINT_TIME`) returns the best solution it found."""
        import numpy as np
        import scipy.optimize
        from scipy.optimize import Bounds, LinearConstraint
        from scipy.sparse import coo_array

        from ..config import JOINT_TIME

        n = len(self.cost)
        if n == 0:
            return np.zeros(0), 0.0
        if self.rows:
            ri, ci, vv = [], [], []
            for i, r in enumerate(self.rows):
                for c, v in r.items():
                    ri.append(i)
                    ci.append(c)
                    vv.append(v)
            A = coo_array((vv, (ri, ci)), shape=(len(self.rows), n)).tocsr()   # duplicates summed
            lo, hi = self.lo, self.hi
        else:
            A, lo, hi = np.zeros((1, n)), [-np.inf], [np.inf]
        try:
            res = scipy.optimize.milp(c=np.array(self.cost), constraints=LinearConstraint(A, lo, hi),
                                      integrality=np.array(self.integer),
                                      bounds=Bounds([b[0] for b in self.bnd], [b[1] for b in self.bnd]),
                                      options={"time_limit": JOINT_TIME})
        except Exception:
            return None
        if res.x is None or not (res.success or res.status == 1):
            return None
        return np.round(res.x).astype(float), float(res.fun)


def _radical_delta(e, b):
    """How an unpaired electron on an atom of element `e` at bond-order sum `b` changes the charge
    `q_atom` reads: +1 where the reading counted a lone pair in its place (CH3· read as CH3-), -1
    for an electron-deficient acceptor that has put all its electrons into bonds and still has
    room (H3N->BH2·, boron read as neutral), `None` where there is no place for it."""
    v = VAL.get(e, 4)
    if v - b >= 1:
        return 1.0
    if v - b == 0 and 2 * b < FULL.get(e, 8):
        return -1.0
    return None



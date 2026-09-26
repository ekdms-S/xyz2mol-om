"""The joint bond-order solve (`JOINT`) — every ligand-internal bond's integer order in one MILP,
instead of ①② `Conj` → ④ `Triple` → ④ `Double` → ⑤ in sequence.

    variables   per bond      y2, y3 ∈ {0, 1}, y2 + y3 <= 1     order = 1 + y2 + 2·y3
                per atom      p, n >= 0                          FC = p - n   (C·N·O·F, non-coordinating)
    constraints per atom      Σ(y2 + 2·y3) <= CAP - degree
                              v + degree + Σ(y2 + 2·y3) - 8 = p - n,   -1 <= FC <= 1
    objective   maximise      Σ (s2 - s1)·y2 + (s3 - s1)·y3  -  JOINTQ · Σ (p + n)

`s1 · s2 · s3` are the ③ scores of orders 1 · 2 · 3; a `Conj` score supports both 1 and 2
(`order_scores`). Only C·N·O·F carry the FC terms: for them `FC = v + b - 8` holds over the whole
range `b <= CAP`. Hypervalent atoms and boron price their charge differently (`charge.q_atom`) and
are left to a later phase. A coordinating atom is exempt: under the ionic cut an oxo, imido or
alkylidene carries -2 or -3 legitimately. M–L bonds spend no valence here.

⚠️ `scipy` is imported inside `solve_joint` and nowhere else — with `JOINT` off it is not a
dependency.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass, field

import networkx as nx

from ..config import CAP, JOINT_MAX, JOINTQ, VAL

LINEAR_FC = {"C", "N", "O", "F"}


@dataclass
class JointResult:
    """`status` is `optimal` · `relaxed_fc` (solved with the FC range dropped) · `too_large` ·
    `unavailable` (no scipy) · `failed`. `orders` maps every internal bond to 1 · 2 · 3, and is
    empty unless the solve succeeded."""

    status: str
    orders: dict = field(default_factory=dict)
    objective: float | None = None


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


def solve_joint(G, el, sc, coord, lam_q=None, fc_bounds=True):
    """Solve every internal bond order of `G` at once. See the module docstring."""
    try:
        import numpy as np
        from scipy.optimize import Bounds, LinearConstraint, milp
    except ImportError:
        return JointResult("unavailable")
    lam_q = JOINTQ if lam_q is None else lam_q
    edges = sorted((min(a, b), max(a, b)) for a, b in G.edges)
    osc = {e: order_scores(sc.get(e, {})) for e in edges}
    col = {}
    for e in edges:
        if el[e[0]] == "H" or el[e[1]] == "H":
            continue
        if osc[e][2] is not None:
            col[("y2", e)] = len(col)
        if osc[e][3] is not None:
            col[("y3", e)] = len(col)
    lin = [x for x in sorted(G.nodes) if el[x] in LINEAR_FC and x not in coord]
    for x in lin:
        col[("p", x)] = len(col)
        col[("n", x)] = len(col)
    nv = len(col)
    if nv > JOINT_MAX:
        return JointResult("too_large")
    if nv == 0:
        return JointResult("optimal", {e: 1 for e in edges}, 0.0)

    c = np.zeros(nv)  # milp minimises
    for (kind, key), i in col.items():
        if kind == "y2":
            c[i] = -(osc[key][2] - osc[key][1])
        elif kind == "y3":
            c[i] = -(osc[key][3] - osc[key][1])
        else:
            c[i] = lam_q
    rows, lo, hi = [np.zeros(nv)], [-np.inf], [np.inf]  # a no-op row keeps the matrix non-empty

    def add(row, low, high):
        rows.append(row)
        lo.append(low)
        hi.append(high)

    for e in edges:
        if ("y2", e) in col and ("y3", e) in col:
            r = np.zeros(nv)
            r[col[("y2", e)]] = r[col[("y3", e)]] = 1
            add(r, -np.inf, 1)
    inc = collections.defaultdict(list)
    for e in edges:
        inc[e[0]].append(e)
        inc[e[1]].append(e)
    lin_set = set(lin)
    for x in sorted(G.nodes):
        deg = G.degree(x)
        extra = np.zeros(nv)
        for e in inc[x]:
            if ("y2", e) in col:
                extra[col[("y2", e)]] += 1
            if ("y3", e) in col:
                extra[col[("y3", e)]] += 2
        cap = CAP.get(el[x])
        if cap is not None:
            add(extra.copy(), -np.inf, cap - deg)
        if x in lin_set:
            v = VAL[el[x]]
            r = extra.copy()
            r[col[("p", x)]] = -1
            r[col[("n", x)]] = 1
            add(r, 8 - v - deg, 8 - v - deg)
            if fc_bounds:
                add(extra.copy(), 7 - v - deg, 9 - v - deg)
    integ = np.zeros(nv)
    ub = np.full(nv, 8.0)
    for (kind, _key), i in col.items():
        if kind in ("y2", "y3"):
            integ[i] = 1
            ub[i] = 1.0
    try:
        res = milp(c=c, constraints=LinearConstraint(np.array(rows), lo, hi),
                   integrality=integ, bounds=Bounds(np.zeros(nv), ub))
    except Exception:
        return JointResult("failed")
    if not res.success or res.x is None:
        if fc_bounds:
            again = solve_joint(G, el, sc, coord, lam_q=lam_q, fc_bounds=False)
            if again.status == "optimal":
                again.status = "relaxed_fc"
            return again
        return JointResult("failed")
    x = np.round(res.x).astype(int)
    orders = {}
    for e in edges:
        y2 = x[col[("y2", e)]] if ("y2", e) in col else 0
        y3 = x[col[("y3", e)]] if ("y3", e) in col else 0
        orders[e] = 1 + int(y2) + 2 * int(y3)
    return JointResult("optimal", orders, float(-res.fun))


def conj_annotation(G, sc, orders):
    """Stop-gap `Conj` label for the 4-class output: a bond whose best ③ class is `Conj`, solved
    to 1 or 2, and touching another such bond (the R5 rule — a lone `Conj` is not delocalised).
    Phase 3 replaces this with the second-best-solution reading."""
    cand = {e for e, o in orders.items()
            if o in (1, 2) and sc.get(e) and max(sc[e], key=sc[e].get) == 3}
    H = nx.Graph()
    H.add_edges_from(cand)
    return {e for e in cand if H.degree(e[0]) > 1 or H.degree(e[1]) > 1}

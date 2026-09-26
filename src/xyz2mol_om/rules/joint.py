"""The joint bond-order solve (`JOINT`) — every ligand-internal bond's integer order in one MILP,
instead of ①② `Conj` → ④ `Triple` → ④ `Double` → ⑤ in sequence.

    per bond    y2, y3 ∈ {0, 1}, y2 + y3 <= 1                   order = 1 + y2 + 2·y3
    per atom    one-hot level s[x, k] ∈ {0, 1}, Σ_k s = 1         k = how far x is raised above
                Σ_k k·s[x, k] = Σ (y2 + 2·y3) over x's bonds       its all-single bond sum
                q(x) = Σ_k q_atom(x, degree + k)·s[x, k]
    objective   maximise  Σ (s2 - s1)·y2 + (s3 - s1)·y3  -  JOINTQ · Σ |q(x)|·s[x, k]

`s1 · s2 · s3` are the ③ scores of orders 1 · 2 · 3; a `Conj` score supports both 1 and 2
(`order_scores`). Each level's charge is `charge.q_atom` itself — the function the output path
prices charges with — so the carbene, sulfoxide, hypervalent and boron conventions are the same in
the solve and in the result.

Levels stop at `CAP`. A non-coordinating atom only gets levels with `|q| <= 1`, a sigma donor levels with
`-(M–L order) <= q <= 1` (`_charge_ok`), and a period-2 atom only levels within the octet
(`degree + k <= 4`); `fc_bounds=False` lifts all three. Every charge is paid for (`_charge_cost`),
except the negative units a multiple M–L bond explains under the ionic cut (oxo, imido,
alkylidene). M–L bonds spend no valence here.

A ring carbon of the sequential `Conj` set with three neighbours gets one more level, a
carbenium (sextet, q = +1), so a Hückel cation can be written. `q_atom` reads that carbon as -1,
so the result carries the difference as a **residual** on that atom (the same convention as ⑥'s
Hückel residual: charge the skeleton does not carry).

Each connected fragment is solved on its own first. With a total charge, the solved fragments and
one oxidation-state variable per metal are then solved together under
`Σ q(fragments) + Σ OS = Q`, with the OS candidates from `config.os_range`.

⚠️ `scipy` is imported inside the solve and nowhere else — with `JOINT` off it is not a
dependency.
"""

from __future__ import annotations

import collections
import math
from dataclasses import dataclass, field

import networkx as nx

from ..charge.formal import q_atom
from ..config import (CAP, DATA, JOINT_MAX, JOINTCAT, JOINTDON, JOINTOSW, JOINTQ, JOINTSYM,
                      os_range)

PERIOD2 = {"B", "C", "N", "O", "F"}
_SOLVED = ("optimal", "relaxed_fc")


@dataclass
class JointResult:
    """`status` — `optimal` · `relaxed_fc` (solved with the FC range dropped) · `too_large` ·
    `unavailable` (no scipy) · `failed`, or `partial` when some fragments were solved and some
    were not. `orders` maps every bond of the **solved** fragments to 1 · 2 · 3. `components` is
    `{min atom index of the fragment: status}`.

    `q_status` — `no_q` (no total charge given) · `q_ok` · `os_widened` (solved only with the OS
    range widened to [-4, 8]) · `q_relaxed` (no solution with the total charge; the per-fragment
    answer stands). `os` — `{metal index: oxidation state}` when the total charge was applied.
    `residual` — `{carbenium atom: +2}`, the charge its `q_atom` reading misses."""

    status: str
    orders: dict = field(default_factory=dict)
    objective: float | None = None
    components: dict = field(default_factory=dict)
    os: dict = field(default_factory=dict)
    residual: dict = field(default_factory=dict)
    q_status: str = "no_q"


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
        import numpy as np
        from scipy.optimize import Bounds, LinearConstraint, milp

        n = len(self.cost)
        if n == 0:
            return np.zeros(0), 0.0
        A = np.zeros((max(len(self.rows), 1), n))
        lo, hi = [-np.inf], [np.inf]
        if self.rows:
            lo, hi = self.lo, self.hi
            for i, r in enumerate(self.rows):
                for c, v in r.items():
                    A[i, c] += v
        try:
            res = milp(c=np.array(self.cost), constraints=LinearConstraint(A, lo, hi),
                       integrality=np.array(self.integer),
                       bounds=Bounds([b[0] for b in self.bnd], [b[1] for b in self.bnd]))
        except Exception:
            return None
        if not res.success or res.x is None:
            return None
        return np.round(res.x).astype(float), float(res.fun)


def _charge_ok(q, ml_order):
    """Is charge `q` admissible under the FC range? A free atom: `|q| <= 1`. A sigma donor with
    M–L order `o`: `-o <= q <= 1` — under the ionic cut each M–L bond order can carry one unit of
    negative charge (oxo -2, nitrido -3), but a donor is never more positive than a free atom."""
    if ml_order is None:
        return abs(q) <= 1 + 1e-9
    return -ml_order - 1e-9 <= q <= 1 + 1e-9


def _charge_cost(q, ml_order, lam):
    """The charged-atom penalty of charge `q`. A free atom pays `lam·|q|`. A sigma donor pays for
    its first unit of negative charge like anyone else — a pyridine N(-) is a charge the metal would
    have to absorb — but the extra units its multiple M–L bond explains are free (an alkylidene at
    -2 pays what a -1 would)."""
    if ml_order is None or q > 0:
        return lam * abs(q)
    if q == 0:
        return 0.0
    extra = max(0.0, -q - ml_order)          # beyond what the M–L order explains
    first = min(1.0, max(0.0, -q - (ml_order - 1)))   # the donor's first unit
    return lam * (JOINTDON * first + extra)


def _default_qfun(G, el, coord):
    def qfun(x, b):
        return q_atom(el[x], float(b), G.degree(x), tuple(sorted(el[w] for w in G[x])),
                      n_ml=(1 if x in coord else 0))
    return qfun


def _fragment(M, G, nodes, el, sc, coord, qfun, lam_q, fc_bounds, ring_c, reserve, huckel):
    """Add one fragment's variables and rows to `M`.

    Returns `(ycol, lvl, q_terms, q_const)` — bond columns `{(edge, 2 | 3): col}`, level columns
    `{atom: [(col, k, q, carbenium)]}`, the fragment charge as `{col: coef}` plus a constant — or
    `None` if some atom has no admissible level at all."""
    edges = sorted((min(a, b), max(a, b)) for a, b in G.subgraph(nodes).edges)
    ycol = {}
    for e in edges:
        if el[e[0]] == "H" or el[e[1]] == "H":
            continue
        s = order_scores(sc.get(e, {}))
        for o in (2, 3):
            if s[o] is not None:
                ycol[(e, o)] = M.var(cost=-(s[o] - s[1]))
        if (e, 2) in ycol and (e, 3) in ycol:
            M.row({ycol[(e, 2)]: 1, ycol[(e, 3)]: 1}, -float("inf"), 1)
    inc = collections.defaultdict(list)
    for e in edges:
        inc[e[0]].append(e)
        inc[e[1]].append(e)
    hsign = {x: sg for atoms, sg, _n in huckel for x in atoms}
    lvl, q_terms, q_const = {}, {}, 0.0
    for x in sorted(nodes):
        deg = G.degree(x)
        ex = [(ycol[(e, o)], o - 1) for e in inc[x] for o in (2, 3) if (e, o) in ycol]
        mlo = coord.get(x)
        if not ex:
            q_const += qfun(x, deg)
            continue
        # a donor keeps room for its M–L bonds (`reserve`): the Mayer M–L order is a fixed input
        #   the internal bonds must leave space for, not something they compete with
        r = int(round(reserve.get(x, 0)))
        kmax = min(sum(w for _c, w in ex), max(CAP.get(el[x], 4) - deg - r, 0))
        if fc_bounds and el[x] in PERIOD2:
            kmax = min(kmax, max(4 - deg - r, 0))
        levels = []
        for k in range(kmax + 1):
            q = qfun(x, deg + k)
            if fc_bounds and not _charge_ok(q, mlo):
                continue
            levels.append((k, q, False))
        sign = hsign.get(x)
        if x in ring_c and el[x] == "C" and deg == 3 and mlo is None:
            levels.append((0, 1.0, True))
        if not levels:
            return None
        cols = []
        for k, q, cat in levels:
            cost = _charge_cost(q, mlo, lam_q)
            # the sign a Hückel ring does not ask for costs `JOINTCAT` more; a carbenium off such a
            #   ring always does. What the ring does ask for is refunded per ring, below.
            if cat and sign != "cat":
                cost += lam_q * JOINTCAT
            elif not cat and q < 0 and sign == "cat":
                cost += lam_q * JOINTCAT
            cols.append((M.var(cost=cost), k, q, cat))
        M.row({c: 1 for c, _k, _q, _cat in cols}, 1, 1)
        r = collections.Counter()
        for c, k, _q, _cat in cols:
            r[c] += k
        for c, w in ex:
            r[c] -= w
        M.row(dict(r), 0, 0)
        for c, _k, q, _cat in cols:
            q_terms[c] = q_terms.get(c, 0.0) + q
        lvl[x] = cols
    # ★ Hückel refund: a carbocycle whose aromatic count asks for `n` charges of one sign (C5 one
    #   anion, C7 one cation, C8 two anions) gets up to `n` of them back — the count, not every
    #   charge on the ring, or a Cp comes out Cp(3-).
    nodes_set = set(nodes)
    for atoms, sg, n in huckel:
        if not set(atoms) <= nodes_set:
            continue
        hits = [c for x in atoms for c, _k, q, cat in lvl.get(x, ())
                if (cat if sg == "cat" else (q < 0 and not cat))]
        if not hits:
            continue
        z = M.var(cost=-lam_q, lb=0.0, ub=float(n), integer=False)
        r = {c: -1 for c in hits}
        r[z] = 1
        M.row(r, -float("inf"), 0)
    return ycol, lvl, q_terms, q_const


def _read(x, G, nodes, qfun, ycol, lvl):
    """Orders and carbenium residuals of one fragment from a solution vector."""
    orders, residual = {}, {}
    for a, b in G.subgraph(nodes).edges:
        e = (min(a, b), max(a, b))
        y2 = int(x[ycol[(e, 2)]]) if (e, 2) in ycol else 0
        y3 = int(x[ycol[(e, 3)]]) if (e, 3) in ycol else 0
        orders[e] = 1 + y2 + 2 * y3
    for a, cols in lvl.items():
        for c, _k, q, cat in cols:
            if cat and x[c] > 0.5:
                residual[a] = q - qfun(a, G.degree(a))
    return orders, residual


def solve_joint(G, el, sc, coord, lam_q=None, fc_bounds=True, *, qfun=None, ring_c=(),
                q_total=None, metals=None, skip=(), seq_q=None, reserve=None, huckel=()):
    """Solve every internal bond order of `G`. See the module docstring.

    `qfun(x, b)` — the charge of atom `x` at bond-order sum `b` (default: `q_atom` from `G`).
    `coord` — sigma donors as `{atom: total M–L bond order}` (a plain set means order 1).
    `ring_c` — carbons that may take the carbenium level.
    `metals` — `{index: element}`. `reserve` — `{atom: valence kept for its M–L bonds}`.
    `huckel` — `[(ring atoms, "cat" | "an", count)]`: carbocycles whose aromatic count asks for
    `count` charges of one sign (`ring_c` should hold their carbons).
    `skip` — fragments (by min atom index) left to the sequential path, e.g. clusters.
    `seq_q` — `{fragment min index: charge}` of the sequential answer, used as a constant for
    every fragment the joint solve does not take.
    """
    try:
        import scipy.optimize  # noqa: F401
    except ImportError:
        return JointResult("unavailable")
    lam_q = JOINTQ if lam_q is None else lam_q
    ring_c = set(ring_c)
    huckel = list(huckel)
    coord = dict(coord) if isinstance(coord, dict) else dict.fromkeys(coord, 1)
    qfun = qfun or _default_qfun(G, el, coord)
    skip, seq_q, metals, reserve = set(skip), seq_q or {}, metals or {}, reserve or {}

    comps = [sorted(c) for c in nx.connected_components(G)]
    orders, residual, status, bounds, obj = {}, {}, {}, {}, 0.0
    for nodes in comps:
        key = nodes[0]
        if key in skip:
            status[key] = "skipped"
            continue
        for fb in ((True, False) if fc_bounds else (False,)):
            M = _Model()
            built = _fragment(M, G, nodes, el, sc, coord, qfun, lam_q, fb, ring_c, reserve, huckel)
            if built is not None and len(M.cost) > JOINT_MAX:
                status[key] = "too_large"
                break
            sol = M.solve() if built is not None else None
            if sol is not None:
                o, r = _read(sol[0], G, nodes, qfun, built[0], built[1])
                orders.update(o)
                residual.update(r)
                obj += sol[1]
                status[key] = "optimal" if fb == fc_bounds else "relaxed_fc"
                bounds[key] = fb
                break
            status[key] = "failed"

    res = JointResult(_summary(status), orders, -obj if orders else None, status,
                      residual=residual)
    if q_total is None:
        return res

    # ── the total charge: every solved fragment and one OS per metal, in one model ─────────────
    const = sum(seq_q.get(k, 0.0) for k, st in status.items() if st not in _SOLVED)
    for widen in (False, True):
        M = _Model()
        blocks, q_row, q_rhs = [], collections.Counter(), float(q_total) - const
        for nodes in comps:
            key = nodes[0]
            if status[key] not in _SOLVED:
                continue
            built = _fragment(M, G, nodes, el, sc, coord, qfun, lam_q, bounds[key], ring_c, reserve, huckel)
            blocks.append((nodes, built))
            for c, v in built[2].items():
                q_row[c] += v
            q_rhs -= built[3]
        os_col = {}
        for m, e in sorted(metals.items()):
            lo, hi = os_range(e)
            if widen:
                lo, hi = min(lo, -4), max(hi, 8)
            os_col[m] = M.var(cost=0.0, lb=lo, ub=hi)
            q_row[os_col[m]] += 1
            if JOINTOSW:
                # the prior needs a cost per value, so the state is also written one-hot
                one = {v: M.var(cost=JOINTOSW * c) for v, c in os_prior_cost(e, lo, hi).items()}
                M.row(dict.fromkeys(one.values(), 1), 1, 1)
                link = {t: -v for v, t in one.items()}
                link[os_col[m]] = 1
                M.row(link, 0, 0)
        ms = sorted(metals)
        for i, a in enumerate(ms):
            for b in ms[i + 1:]:
                if metals[a] == metals[b]:
                    d = M.var(cost=JOINTSYM, lb=0.0, ub=20.0, integer=False)
                    M.row({d: 1, os_col[a]: -1, os_col[b]: 1}, 0, float("inf"))
                    M.row({d: 1, os_col[a]: 1, os_col[b]: -1}, 0, float("inf"))
        M.row(dict(q_row), q_rhs, q_rhs)
        sol = M.solve()
        if sol is None:
            continue
        orders, residual = dict(orders), {}
        for nodes, built in blocks:
            o, r = _read(sol[0], G, nodes, qfun, built[0], built[1])
            orders.update(o)
            residual.update(r)
        res.orders, res.residual = orders, residual
        res.objective = -sol[1]
        res.os = {m: int(round(sol[0][c])) for m, c in os_col.items()}
        res.q_status = "os_widened" if widen else "q_ok"
        return res
    res.q_status = "q_relaxed"
    return res


def _summary(status):
    st = set(status.values()) - {"skipped"}
    if not st or st <= {"optimal"}:
        return "optimal"
    if st <= set(_SOLVED):
        return "relaxed_fc"
    if st & set(_SOLVED):
        return "partial"
    return "too_large" if "too_large" in st else "failed"

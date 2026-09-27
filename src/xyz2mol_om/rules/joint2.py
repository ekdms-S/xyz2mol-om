"""JOINT v2 — the ligand-internal bond information of the whole input in one MILP, with no
metal-free T3 pre-solve (`dev/docs/plans/2026-09-27-joint-v2.md`).

    1. topology      T1 · T4 · clusters · haptic groups · Hückel carbocycles      (`topology`)
    2. one MILP      bond orders · bond-order-sum state per atom · haptic h per group · OS ·
                     cluster charges (Wade) · unpaired electrons — no M–L information
    3. candidates    K-best by signature (or a cut loop), validated (V1: a sigma donor can bond)
    4. M–L orders    min(Mayer order, donor lone pairs); Mayer consistency ranks close candidates
    5. reading       Conj (S/D flips that keep the score) · 3c2e tags · carbenium charges

The MILP pieces (`_Model`, `order_scores`, `q_atom` levels, OS prior) are shared with `rules.joint`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx
import numpy as np

from ..config import CAP, HUCKEL, TAU_P
from ..geometry import plane_rms


@dataclass
class Topology:
    """What the MILP is built on — nothing here depends on a bond-order assignment."""

    G: nx.Graph                 # ligand-internal bonds (T1)
    ml_pred: list               # T4 M–L bonds after the agostic · saturated · bound-halide drops
    cen: set                    # metal indices
    dbond: dict                 # per-pair M–L distance cutoffs (T4)
    clusters: set = field(default_factory=set)    # fragments (min atom index) with a cage atom
    groups: list = field(default_factory=list)    # [(metal, atoms, internal bonds)] haptic candidates
    rings: list = field(default_factory=list)     # [(atoms, "cat" | "an", count)] Hückel carbocycles


def _pi_capable(el, G, x):
    """Can `x` take part in a π system — not H, and fewer neighbours than its valence cap."""
    return el[x] != "H" and G.degree(x) < CAP.get(el[x], 4)


def topology(el, xyz, wbo=None, dint=None, G=None, ml_raw=None, dbond=None, cen=None):
    """Step 1 of JOINT v2 — topology only (see the module docstring)."""
    from ..api import build_topology
    from .pipeline import drop_agostic, drop_agostic_carbon, drop_bound_halide, drop_saturated

    xyz = np.asarray(xyz, dtype=float)
    if G is None:
        G, ml_raw, dbond, _c1g, cen = build_topology(el, xyz, wbo, dint)
    ml_pred = drop_bound_halide(
        el, G, drop_saturated(el, G, drop_agostic_carbon(
            el, xyz, G, drop_agostic(el, G, ml_raw))), wbo)
    topo = Topology(G=G, ml_pred=ml_pred, cen=set(cen), dbond=dbond)
    cage = set()
    for comp in nx.connected_components(G):
        if any(G.degree(x) > CAP.get(el[x], 4) for x in comp):
            topo.clusters.add(min(comp))
            cage |= set(comp)
    # haptic groups: the atoms one metal touches that can be unsaturated, split into connected runs
    by_m = {}
    for m, x in ml_pred:
        if x not in cage and _pi_capable(el, G, x):
            by_m.setdefault(m, set()).add(x)
    for m in sorted(by_m):
        sub = G.subgraph(by_m[m])
        for comp in sorted(nx.connected_components(sub), key=min):
            if len(comp) >= 2:
                bonds = sorted((min(a, b), max(a, b)) for a, b in sub.subgraph(comp).edges)
                topo.groups.append((m, tuple(sorted(comp)), bonds))
    # Hückel carbocycles: all-carbon, every atom able to be unsaturated, planar
    for ring in nx.cycle_basis(G):
        n = len(ring)
        if any(el[x] != "C" or not _pi_capable(el, G, x) or x in cage for x in ring):
            continue
        if plane_rms(xyz[np.array(ring)]) > TAU_P:
            continue
        if n % 2 and (n - 1) in HUCKEL:
            topo.rings.append((tuple(ring), "cat", 1))
        elif n % 2 and (n + 1) in HUCKEL:
            topo.rings.append((tuple(ring), "an", 1))
        elif not n % 2 and (n + 2) in HUCKEL and n not in HUCKEL:
            topo.rings.append((tuple(ring), "an", 2))
    return topo


# ═══ the MILP ═════════════════════════════════════════════════════════════════════════════════

QLIG = range(-8, 5)          # ligand charges a fragment may take


def wade_charges(n_b, n_c, extra_h):
    """Wade–Mingos charge of an n-vertex (car)borane cage for each cage type.

    skeletal electrons = 2 per B vertex + 3 per C vertex + 1 per extra H (bridging or endo) +
    the anionic charge; closo needs 2n+2, nido 2n+4, arachno 2n+6. `{"closo": q, …}`.
    """
    n = n_b + n_c
    e = 2 * n_b + 3 * n_c + extra_h
    return {"closo": e - (2 * n + 2), "nido": e - (2 * n + 4), "arachno": e - (2 * n + 6)}


def _cluster_charges(el, G, comp, contacted):
    """Wade charge candidates of a cluster fragment and the preferred type: a cage that touches a
    metal is nido (the metal closes the polyhedron), a free cage closo."""
    cage = {x for x in comp if el[x] == "B" or (el[x] == "C" and any(el[y] == "B" for y in G[x]))}
    extra_h = 0
    for x in comp:
        if el[x] == "H" and sum(1 for y in G[x] if y in cage) >= 2:
            extra_h += 1
    for x in cage:
        hs = sum(1 for y in G[x] if el[y] == "H" and G.degree(y) == 1)
        extra_h += max(0, hs - 1)
    cand = wade_charges(sum(el[x] == "B" for x in cage), sum(el[x] == "C" for x in cage), extra_h)
    pref = "nido" if cage & contacted else "closo"
    return cand, pref


def _raw_scores(sc_e):
    """`{1: s1, 2: s2, 3: s3}` from the Single · Double · Triple classes alone. A bond whose table
    has only `Conj` (no Single/Double entries) falls back to `order_scores`."""
    from .joint import order_scores

    if 0 not in sc_e and 1 not in sc_e:
        return order_scores(sc_e)
    return {1: sc_e.get(0, 0.0), 2: sc_e.get(1), 3: sc_e.get(2)}


def _free_ok(q):
    return abs(q) <= 1 + 1e-9


def _free_cost(q, lam):
    """Charged-atom penalty of a non-coordinating atom. A coordinating atom (sigma donor or haptic
    group atom) is exempt from both the range and the penalty (orbit-integration-ideas §2.5): under
    the ionic cut its charge is the notation, not a defect."""
    return lam * abs(q)


def _pi_chains(topo, el, sc):
    """(c) — a haptic group keeps only atoms that can take part in a pi bond inside the group
    (some bond to another group atom has a Double or Triple score), split into connected chains.
    A dropped atom stays an ordinary sigma contact (an ansa Me2Si bridge T4 picked up)."""
    from dataclasses import replace

    from .joint import order_scores

    def pi_ok(e):
        s = order_scores(sc.get(e, {}))
        return s[2] is not None or s[3] is not None

    groups = []
    for m, atoms, bonds in topo.groups:
        pb = [e for e in bonds if pi_ok(e)]
        keep = {x for e in pb for x in e}
        sub = nx.Graph(pb)
        for comp in sorted(nx.connected_components(sub), key=min):
            if len(comp) >= 2 and comp <= keep:
                groups.append((m, tuple(sorted(comp)),
                               sorted(e for e in bonds if e[0] in comp and e[1] in comp)))
    return replace(topo, groups=groups)


@dataclass
class Candidate:
    orders: dict
    state: dict                 # {atom: (k, q, carbenium, unpaired)}
    h: dict                     # {group index: 0 | 1}
    os: dict
    cluster_q: dict
    qlig: dict                  # {fragment min index: charge}
    objective: float
    signature: tuple
    valid: bool = True
    failed: list = field(default_factory=list)
    ml_orders: dict = field(default_factory=dict)
    mayer: float | None = None
    metal_unpaired: dict = field(default_factory=dict)


@dataclass
class JointV2:
    status: str = "optimal"            # optimal · relaxed_fc · q_relaxed · too_large · unavailable · failed
    q_status: str = "no_q"
    best: Candidate | None = None
    candidates: list = field(default_factory=list)
    ranking: str = "milp"              # milp · mayer
    n_rejected: int = 0
    alt_os: dict = field(default_factory=dict)
    alt_gap: float | None = None
    conj: set = field(default_factory=set)
    carbenium: set = field(default_factory=set)
    radicals: dict = field(default_factory=dict)
    metal_unpaired: dict = field(default_factory=dict)

    # the chosen candidate's pieces, for callers
    @property
    def orders(self):
        return self.best.orders if self.best else {}

    @property
    def os(self):
        return self.best.os if self.best else {}

    @property
    def ml_orders(self):
        return self.best.ml_orders if self.best else {}

    @property
    def cluster_q(self):
        return self.best.cluster_q if self.best else {}

    hap: set = field(default_factory=set)


class _Build:
    """One MILP over the whole input, with handles to decode a solution."""

    def __init__(self, topo, el, sc, qfun, lam, relax, q_total, n_unpaired, metals, only=None):
        """`relax` — fragments (min atom index) solved without the FC range; `only` — build just
        these fragments (the per-fragment feasibility probe)."""
        from ..config import (FULL, JOINTADJ, JOINTRAWSC, JOINTCAT, JOINTOSW, JOINTRAD, JOINTSYM, VAL,
                              _GROUP, os_range)
        from .joint import _EN, _FSHELL, PERIOD2, _Model, _radical_delta, order_scores, os_prior_cost

        self.topo, self.el = topo, el
        G = topo.G
        M = self.M = _Model()
        contacted = {x for _m, x in topo.ml_pred}
        metal_of = {}
        for m_, x_ in topo.ml_pred:
            metal_of.setdefault(x_, set()).add(m_)
        grp_of = {}
        for gi, (_m, atoms, _b) in enumerate(topo.groups):
            for x in atoms:
                grp_of[x] = gi
        hsign = {x: sg for atoms, sg, _n in topo.rings for x in atoms}
        ring_c = {x for atoms, _s, _n in topo.rings for x in atoms if G.degree(x) == 3}
        self.frags = [sorted(c) for c in nx.connected_components(G)]
        self.hcol = {gi: M.var(cost=0.0) for gi in range(len(topo.groups))}
        self.ycol, self.lvl, self.qcol, self.ccol, self.const_q = {}, {}, {}, {}, {}
        q_row = {}
        q_rhs = float(q_total) if q_total is not None else 0.0
        rad_cols = []

        for comp in self.frags:
            key = comp[0]
            if only is not None and key not in only:
                continue
            fc_bounds = key not in relax
            if key in topo.clusters:
                cand, pref = _cluster_charges(el, G, set(comp), contacted)
                one = {}
                for kind, qv in cand.items():
                    one[kind] = M.var(cost=0.0 if kind == pref else 2.0 * lam)
                    q_row[one[kind]] = q_row.get(one[kind], 0.0) + qv
                M.row(dict.fromkeys(one.values(), 1), 1, 1)
                self.ccol[key] = (one, cand)
                continue
            edges = sorted((min(a, b), max(a, b)) for a, b in G.subgraph(comp).edges)
            for e in edges:
                if el[e[0]] == "H" or el[e[1]] == "H":
                    continue
                s = (_raw_scores if JOINTRAWSC else order_scores)(sc.get(e, {}))
                for o in (2, 3):
                    if s[o] is not None:
                        self.ycol[(e, o)] = M.var(cost=-(s[o] - s[1]))
                if (e, 2) in self.ycol and (e, 3) in self.ycol:
                    M.row({self.ycol[(e, 2)]: 1, self.ycol[(e, 3)]: 1}, -float("inf"), 1)
            inc = {}
            for e in edges:
                inc.setdefault(e[0], []).append(e)
                inc.setdefault(e[1], []).append(e)
            fq, fconst = {}, 0.0
            for x in comp:
                deg = G.degree(x)
                ex = [(self.ycol[(e, o)], o - 1) for e in inc.get(x, ()) for o in (2, 3)
                      if (e, o) in self.ycol]
                role = "coord" if (x in grp_of or x in contacted) else "free"
                rad_ok = bool(n_unpaired) and el[x] != "H"
                if not ex and not rad_ok:
                    self.const_q[x] = qfun(x, deg)
                    fconst += self.const_q[x]
                    continue
                kmax = min(sum(w for _c, w in ex), max(CAP.get(el[x], 4) - deg, 0))
                if fc_bounds and el[x] in PERIOD2:
                    kmax = min(kmax, max(4 - deg, 0))
                levels = []
                for k in range(kmax + 1):
                    q0 = qfun(x, deg + k)
                    opts = [(q0, False)]
                    if rad_ok:
                        dq = _radical_delta(el[x], deg + k)
                        if dq is not None:
                            opts.append((q0 + dq, True))
                    for q, rad in opts:
                        levels.append((k, q, False, rad))
                if x in ring_c and el[x] == "C" and (x in grp_of or x not in contacted):
                    levels.append((0, 1.0, True, False))
                cols = []
                for k, q, cat, rad in levels:
                    ok = role == "coord" or not fc_bounds or (not ex and not rad) or _free_ok(q)
                    if not ok:
                        continue
                    base = _free_cost(q, lam) if role == "free" else 0.0
                    if rad:
                        base += lam * (JOINTRAD + 0.01 * _EN.get(el[x], 2.5))
                    sign = hsign.get(x)
                    if cat and sign != "cat":
                        base += lam * JOINTCAT
                    elif not cat and not rad and q < 0 and k == 0 and sign == "cat":
                        base += lam * JOINTCAT
                    c = M.var(cost=base)
                    cols.append((c, k, q, cat, rad))
                if not cols:
                    self.infeasible = True
                    return
                M.row({c: 1 for c, *_ in cols}, 1, 1)
                r = {}
                for c, k, *_ in cols:
                    r[c] = r.get(c, 0) + k
                for c, w in ex:
                    r[c] = r.get(c, 0) - w
                M.row(r, 0, 0)
                for c, _k, q, _cat, rad in cols:
                    fq[c] = fq.get(c, 0.0) + q
                    if rad:
                        rad_cols.append(c)
                self.lvl[x] = cols
            # the fragment charge, one-hot (it is part of the signature)
            z = {v: M.var(cost=0.0) for v in QLIG}
            M.row(dict.fromkeys(z.values(), 1), 1, 1)
            r = dict(fq)
            for v, c in z.items():
                r[c] = r.get(c, 0.0) - v
            M.row(r, -fconst, -fconst)
            self.qcol[key] = z
            for v, c in z.items():
                q_row[c] = q_row.get(c, 0.0) + v
            # ★ Hückel refund per carbocycle: the charges its aromatic count asks for
            cs = set(comp)
            for atoms, sg, n in topo.rings:
                if not set(atoms) <= cs:
                    continue
                hits = [c for x in atoms for c, _k, q, cat, rad in self.lvl.get(x, ())
                        if not rad and (cat if sg == "cat" else (q < 0 and not cat))]
                if hits:
                    zr = M.var(cost=-lam, lb=0.0, ub=float(n), integer=False)
                    row = {c: -1 for c in hits}
                    row[zr] = 1
                    M.row(row, -float("inf"), 0)
            # ★ adjacent same-sign charges: a penalty per pair
            for a, b in edges:
                if el[a] == "H" or el[b] == "H" or metal_of.get(a, set()) & metal_of.get(b, set()):
                    continue   # C(-)–C(-) of a metallacyclopropane, O(-)–O(-) of a peroxide
                for sgn in (-1, 1):
                    ra, ca = self._sign(a, sgn)
                    rb, cb = self._sign(b, sgn)
                    if not ra and not rb:
                        continue
                    p = M.var(cost=JOINTADJ * lam, lb=0.0, ub=1.0, integer=False)
                    row = {p: 1}
                    for c in ra + rb:
                        row[c] = row.get(c, 0) - 1
                    M.row(row, -1.0 + ca + cb, float("inf"))
        # haptic groups: h = 1 -> floor(k/2) multiple bonds (even k: or one fewer + two anions)
        for gi, (_m, atoms, bonds) in enumerate(topo.groups):
            h = self.hcol[gi]
            k = len(atoms)
            need = k // 2
            mult = {}
            for e in bonds:
                for o in (2, 3):
                    if (e, o) in self.ycol:
                        mult[self.ycol[(e, o)]] = 1
            big = len(bonds) + 2
            a = M.var(cost=0.0) if k % 2 == 0 and k >= 4 else None
            r1 = dict(mult)
            r1[h] = r1.get(h, 0) - big
            r2 = dict(mult)
            r2[h] = r2.get(h, 0) + big
            if a is not None:
                r1[a] = 1
                r2[a] = 1
                M.row({a: 1, h: -1}, -float("inf"), 0)
                neg = {c: -1 for x in atoms for c, _k, q, *_ in self.lvl.get(x, ()) if q < 0}
                neg[a] = 2
                M.row(neg, -float("inf"), 0)
            M.row(r1, need - big, float("inf"))    # M + a >= need - big(1-h)
            M.row(r2, -float("inf"), need + big)   # M + a <= need + big(1-h)
            # §2.6: three or more in a row are haptic by topology; eta2 alone has two readings,
            #   and the bond decides it — multiple bond <=> haptic, single <=> two sigma bonds
            if k >= 3:
                M.row({h: 1}, 1, 1)
            else:
                r = dict(mult)
                r[h] = r.get(h, 0) - 1
                M.row(r, 0, 0)
            # h = 1 -> the charged (or unpaired) atoms of the chain are fixed by k alone: odd k one
            #   (allyl, Cp, pyrrolyl, C7H7+), even k none (alkene, arene, eta2-nitrile N#C) or two
            #   anions with one multiple bond fewer (COT2-). A chain with boron is left out: an sp2
            #   boron is neutral without a multiple bond (boratabenzene breaks the count).
            if any(el[x] == "B" for x in atoms):
                continue
            chg, const = {}, 0
            for x in atoms:
                if x in self.lvl:
                    for c, _k, q, _cat, rad in self.lvl[x]:
                        if abs(q) > 1e-9 or rad:
                            chg[c] = chg.get(c, 0) + 1
                elif abs(self.const_q.get(x, 0.0)) > 1e-9:
                    const += 1
            want = {h: -(k % 2)}
            if a is not None:
                want[a] = -2
            r = dict(chg)
            for c, v in want.items():
                r[c] = r.get(c, 0) + v
            bigc = k + 2
            r1 = dict(r)
            r1[h] = r1.get(h, 0) + bigc
            M.row(r1, -float("inf"), bigc - const)     # chg - want <= big(1-h)
            r2 = dict(r)
            r2[h] = r2.get(h, 0) - bigc
            M.row(r2, -bigc - const, float("inf"))     # chg - want >= -big(1-h)
        # metals: OS one-hot with prior, symmetry, open-shell electrons
        self.os_one, u_col = {}, {}
        for m, e in sorted(metals.items()):
            lo, hi = os_range(e)
            one = {v: M.var(cost=JOINTOSW * c) for v, c in os_prior_cost(e, lo, hi).items()}
            M.row(dict.fromkeys(one.values(), 1), 1, 1)
            for v, c in one.items():
                q_row[c] = q_row.get(c, 0.0) + v
            self.os_one[m] = one
            shell = (_GROUP[e], 10) if e in _GROUP else _FSHELL.get(e)
            if n_unpaired and shell:
                g, S = shell
                osv = {c: v for v, c in one.items()}
                u = M.var(cost=0.0, lb=0.0, ub=S / 2)
                hh = M.var(cost=0.0, lb=-S, ub=S)
                r = {c: v for c, v in osv.items()}
                r.update({u: 1, hh: 2})
                M.row(r, g, g)
                r = {c: v for c, v in osv.items()}
                r[u] = 1
                M.row(r, -float("inf"), g)
                r = {c: -v for c, v in osv.items()}
                r[u] = 1
                M.row(r, -float("inf"), S - g)
                u_col[m] = u
        ms = sorted(metals)
        for i, a in enumerate(ms):
            for b in ms[i + 1:]:
                if metals[a] == metals[b]:
                    d = M.var(cost=JOINTSYM, lb=0.0, ub=20.0, integer=False)
                    r = {d: 1}
                    for v, c in self.os_one[a].items():
                        r[c] = r.get(c, 0) - v
                    for v, c in self.os_one[b].items():
                        r[c] = r.get(c, 0) + v
                    M.row(r, 0, float("inf"))
                    r = {d: 1}
                    for v, c in self.os_one[a].items():
                        r[c] = r.get(c, 0) + v
                    for v, c in self.os_one[b].items():
                        r[c] = r.get(c, 0) - v
                    M.row(r, 0, float("inf"))
        if q_total is not None:
            M.row(q_row, q_rhs, q_rhs)
        if n_unpaired:
            cnt = dict.fromkeys(rad_cols, 1)
            cnt.update(dict.fromkeys(u_col.values(), 1))
            M.row(cnt, n_unpaired, n_unpaired)
        self.u_col = u_col
        self.infeasible = False
        _ = (VAL, FULL)

    def _sign(self, x, sgn):
        """(level columns of x whose charge has sign `sgn`, constant 1/0 if x is constant)."""
        if x in self.lvl:
            return [c for c, _k, q, *_ in self.lvl[x] if q * sgn > 0], 0
        q = self.const_q.get(x, 0.0)
        return [], 1 if q * sgn > 0 else 0

    def decode(self, x, obj):
        G = self.topo.G
        orders = {}
        for a, b in G.edges:
            e = (min(a, b), max(a, b))
            y2 = round(x[self.ycol[(e, 2)]]) if (e, 2) in self.ycol else 0
            y3 = round(x[self.ycol[(e, 3)]]) if (e, 3) in self.ycol else 0
            orders[e] = 1 + int(y2) + 2 * int(y3)
        h = {gi: int(round(x[c])) for gi, c in self.hcol.items()}
        hap_atoms = {a for gi, (_m, atoms, _b) in enumerate(self.topo.groups) if h[gi]
                     for a in atoms}
        donors = {a for _m, a in self.topo.ml_pred} - hap_atoms
        state, sig = {}, []
        for a, cols in self.lvl.items():
            for c, k, q, cat, rad in cols:
                if x[c] > 0.5:
                    state[a] = (k, q, cat, rad)
                    if a in donors:
                        sig.append(c)
        os_ = {}
        for m, one in self.os_one.items():
            for v, c in one.items():
                if x[c] > 0.5:
                    os_[m] = v
                    sig.append(c)
        qlig = {}
        for key, z in self.qcol.items():
            for v, c in z.items():
                if x[c] > 0.5:
                    qlig[key] = v
                    sig.append(c)
        cq = {}
        for key, (one, cand) in self.ccol.items():
            for kind, c in one.items():
                if x[c] > 0.5:
                    cq[key] = cand[kind]
                    qlig[key] = cand[kind]
                    sig.append(c)
        signature = (tuple(sorted(sig)), tuple(sorted(h.items())))
        mu = {m: int(round(x[c])) for m, c in self.u_col.items() if round(x[c])}
        return Candidate(orders=orders, state=state, h=h, os=os_, cluster_q=cq, qlig=qlig,
                         objective=obj, signature=signature, metal_unpaired=mu)

    def forbid(self, cand):
        """No-good cut: not this signature again."""
        ones, hs = cand.signature
        r = {c: 1 for c in ones}
        for gi, v in hs:
            c = self.hcol[gi]
            r[c] = r.get(c, 0) + (1 if v else -1)
        n1 = len(ones) + sum(1 for _gi, v in hs if v)
        self.M.row(r, -float("inf"), n1 - 1)


# ═══ candidates → validation → M–L orders → ranking ════════════════════════════════════════════

MAX_SOLVES = 50              # hard stop on the candidate loop when nothing passes V1


def _default_qfun(topo, el):
    from ..charge.formal import q_atom

    G = topo.G
    nml = {}
    for _m, x in topo.ml_pred:
        nml[x] = nml.get(x, 0) + 1

    def qfun(x, b):
        return q_atom(el[x], float(b), G.degree(x), tuple(sorted(el[w] for w in G[x])),
                      n_ml=nml.get(x, 0))
    return qfun


def _lone_pairs(el, x, deg, st):
    """Lone pairs of atom x in state `st = (k, q, carbenium, unpaired)`, read on the octet."""
    from ..config import VAL

    if st is None:
        return None
    k, q, cat, rad = st
    if cat:
        return 0
    n = VAL.get(el[x], 4) - q - (deg + k) - (1 if rad else 0)
    return max(int(round(n)) // 2, 0)


def _state(cand, x, deg, qfun):
    return cand.state.get(x, (0, qfun(x, deg), False, False))


def _hap_set(topo, h):
    ml = set(topo.ml_pred)
    return {(m, x) for gi, (m, atoms, _b) in enumerate(topo.groups) if h.get(gi)
            for x in atoms if (m, x) in ml}


def _btag(el, topo, cand):
    from .pipeline import bridge_tags

    cls = {e: o - 1 for e, o in cand.orders.items()}
    return bridge_tags(el, topo.G, topo.ml_pred, cls, _hap_set(topo, cand.h))


def _validate(el, topo, cand, qfun):
    """V1 — every sigma donor (h = 0 group atoms included, 3c2e bridges not) can still bond: at
    least one lone pair, and a period-2 atom at most 3 in bond-order sum. Returns failing atoms."""
    from .joint import PERIOD2

    G = topo.G
    hap = {x for _m, x in _hap_set(topo, cand.h)}
    three_c = {x for x, t in _btag(el, topo, cand).items() if t == "3c2e"}
    bad = []
    for x in sorted({x for _m, x in topo.ml_pred} - hap - three_c):
        deg = G.degree(x)
        st = _state(cand, x, deg, qfun)
        if _lone_pairs(el, x, deg, st) < 1 or (el[x] in PERIOD2 and deg + st[0] > 3):
            bad.append(x)
    return bad


def _ml_orders(el, topo, cand, qfun, ml_scores):
    """sigma M–L orders **from the candidate's donor charge** (ionic cut: a donor at q holds
    `max(n_metals, -q)` M–L order in total — neutral / X-type 1 per metal, oxo · alkylidene 2,
    nitrido 3; mu-O(2-) two singles), the extra order going to the bond Mayer rates highest for it.
    Mayer consistency = the sum of Mayer's own score (all classes, `ml_scores`) at those orders;
    a charge Mayer disagrees with scores low. `None` when no bond has a Mayer value."""
    G = topo.G
    hap = _hap_set(topo, cand.h)
    by_x = {}
    for m, x in topo.ml_pred:
        if (m, x) not in hap:
            by_x.setdefault(x, []).append(m)
    ml_scores = ml_scores or {}
    out, cons = {}, None
    for x, ms in by_x.items():
        q = _state(cand, x, G.degree(x), qfun)[1] if x in G else 0
        total = max(len(ms), int(round(-q)))
        o = dict.fromkeys(ms, 1)
        for _ in range(total - len(ms)):
            def gain(m):
                s = ml_scores.get((m, x))
                if not s:
                    return (-1e9, -m)
                return (s.get(o[m], -1e9) - s.get(o[m] - 1, -1e9), -m)
            o[max(ms, key=gain)] += 1
        for m in ms:
            out[(m, x)] = o[m]
            s = ml_scores.get((m, x))
            if s:
                cons = (cons or 0.0) + s.get(o[m] - 1, min(s.values()) - 10.0)
    return out, cons

def _conj(el, topo, cand, sc, qfun, lam, eps):
    """Bonds that flip S<->D along an alternating cycle, or an alternating path whose ends trade
    their states, with the objective moving by at most `eps` → Conj."""
    from .joint import order_scores

    G, orders = topo.G, cand.orders
    fixed = {x for x, (_k, _q, cat, rad) in cand.state.items() if cat or rad}

    def sd(e):
        return orders.get(e) in (1, 2) and el[e[0]] != "H" and el[e[1]] != "H" \
            and not (set(e) & fixed) and e[0] not in topo.clusters

    def score(e, o):
        s = order_scores(sc.get(e, {}))[o]
        return -1e9 if s is None else s

    coord = {x for _m, x in topo.ml_pred}

    def qcost(x, b):
        return 0.0 if x in coord else _free_cost(qfun(x, b), lam)

    conj = set()
    for x0 in G:
        stack = [(x0, [], True)]
        while stack:
            x, path, want_d = stack.pop()
            if len(path) >= 8:
                continue
            for y in G[x]:
                e = (min(x, y), max(x, y))
                if e in path or not sd(e) or (orders[e] == 2) != want_d:
                    continue
                p2 = path + [e]
                closes = y == x0 and len(p2) % 2 == 0
                opens = not want_d and y != x0
                if closes or opens:
                    d = sum(score(f, 3 - orders[f]) - score(f, orders[f]) for f in p2)
                    if opens:   # x0 loses one unit, y gains one
                        b0 = G.degree(x0) + sum(orders[f] - 1 for f in G.edges(x0)
                                                for f in [(min(f), max(f))])
                        b1 = G.degree(y) + sum(orders[f] - 1 for f in G.edges(y)
                                               for f in [(min(f), max(f))])
                        d -= (qcost(x0, b0 - 1) + qcost(y, b1 + 1)) - (qcost(x0, b0) + qcost(y, b1))
                        if abs(qfun(x0, b0 - 1)) > 1 + 1e-9 and x0 not in coord:
                            d = -1e9
                        if abs(qfun(y, b1 + 1)) > 1 + 1e-9 and y not in coord:
                            d = -1e9
                    if d >= -eps:
                        conj.update(p2)
                if y != x0 and not any(y in f for f in path):
                    stack.append((y, p2, not want_d))
    return conj


def solve(topo, el, sc, *, qfun=None, q_total=None, n_unpaired=0, ml_scores=None, mode=None,
          K=None, lam=None):
    """The v2 joint solve. See the module docstring and `dev/docs/plans/2026-09-27-joint-v2.md`."""
    from ..config import JOINT_MAX, JOINTCONJEPS, JOINTK, JOINTMODE, JOINTQ, JOINTTIE

    try:
        import scipy.optimize  # noqa: F401
    except ImportError:
        return JointV2(status="unavailable")
    mode = mode or JOINTMODE
    K = JOINTK if K is None else K
    lam = JOINTQ if lam is None else lam
    qfun = qfun or _default_qfun(topo, el)
    topo = _pi_chains(topo, el, sc)
    metals = {m: el[m] for m in sorted(topo.cen | {m for m, _x in topo.ml_pred})}

    def attempt(relax, qt):
        b = _Build(topo, el, sc, qfun, lam, relax, qt, n_unpaired, metals)
        if b.infeasible:
            return b, None
        return b, (b.M.solve() if len(b.M.cost) <= JOINT_MAX else "too_large")

    # FC range everywhere → without it on the fragments that cannot meet it alone → without Q
    relax = set()
    build, first = attempt(relax, q_total)
    if first == "too_large":
        return JointV2(status="too_large")
    if first is None:
        for comp in nx.connected_components(topo.G):
            key = min(comp)
            if key in topo.clusters:
                continue
            probe = _Build(topo, el, sc, qfun, lam, set(), None, 0, {}, only={key})
            if probe.infeasible or probe.M.solve() is None:
                relax.add(key)
        if relax:
            build, first = attempt(relax, q_total)
    q_status = "q_ok" if q_total is not None else "no_q"
    if first is None and q_total is not None:
        build, first = attempt(relax, None)
        q_status = "q_relaxed"
    if first is None or first == "too_large":
        return JointV2(status="failed" if first is None else "too_large", q_status=q_status)
    status = "relaxed_fc" if relax else "optimal"

    res = JointV2(status=status, q_status=q_status)
    cands, sol = [], first
    n_solves = 0
    while sol is not None and n_solves < MAX_SOLVES:
        n_solves += 1
        c = build.decode(sol[0], sol[1])
        c.failed = _validate(el, topo, c, qfun)
        c.valid = not c.failed
        cands.append(c)
        if mode == "cut":
            if c.valid:
                break
            for x in c.failed:   # forbid the failing donor state only
                for col, k, q, cat, rad in build.lvl.get(x, ()):
                    if sol[0][col] > 0.5:
                        gi = next((g for g, (_m, a, _b) in enumerate(topo.groups) if x in a), None)
                        if gi is None:
                            build.M.row({col: 1}, -float("inf"), 0)
                        else:
                            build.M.row({col: 1, build.hcol[gi]: -1}, -float("inf"), 0)
        else:
            n_valid = sum(cc.valid for cc in cands)
            if len(cands) >= K and n_valid:
                break
            build.forbid(c)
        sol = build.M.solve()

    valid = [c for c in cands if c.valid]
    res.candidates = cands
    res.n_rejected = len(cands) - len(valid)
    pool = valid or cands[:1]
    if not valid:
        res.status = "no_valid"
    for c in pool:
        c.ml_orders, c.mayer = _ml_orders(el, topo, c, qfun, ml_scores)
    best = min(pool, key=lambda c: c.objective)
    tie = [c for c in pool if c.objective <= best.objective + JOINTTIE * lam + 1e-9]
    if mode != "cut" and any(c.mayer is not None for c in tie) and len(tie) > 1:
        best = max(tie, key=lambda c: (c.mayer if c.mayer is not None else -1e18, -c.objective))
        res.ranking = "mayer"
    res.best = best
    res.hap = _hap_set(topo, best.h)
    others = [c for c in pool if c is not best and c.os != best.os]
    if others:
        alt = min(others, key=lambda c: c.objective)
        res.alt_os, res.alt_gap = dict(alt.os), alt.objective - best.objective
    res.carbenium = {x for x, (_k, _q, cat, _r) in best.state.items() if cat}
    G = topo.G
    res.radicals = {x: q - qfun(x, G.degree(x) + k)
                    for x, (k, q, _c, rad) in best.state.items() if rad}
    res.metal_unpaired = dict(best.metal_unpaired)
    res.conj = _conj(el, topo, best, sc, qfun, lam, JOINTCONJEPS)
    return res

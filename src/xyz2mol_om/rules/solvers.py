"""Search — the valence-cap assignment (④) · local search for the conjugated set (③) ·
the R6 swap.

"""

# ruff: noqa: E501
from __future__ import annotations

import collections

import networkx as nx
import numpy as np

from ..config import (CAP, CAPDUP_MAX, CAPMILP, CAPMILP_MAX, ORD4,
                     R6SWAP, TAUD, VTGT)
from ..charge.formal import _qfrag, frag_charge, q_atom


def _kek_val(G, el, cls):
    """Per-atom valence under **Kekule counting**. `k` conjugated bonds count as `k+1`
    (one of them is the π bond)."""
    nk = collections.Counter()
    bn = collections.defaultdict(float)
    for e, v in cls.items():
        if v == 3:
            nk[e[0]] += 1
            nk[e[1]] += 1
        else:
            bn[e[0]] += ORD4[v]
            bn[e[1]] += ORD4[v]
    return {x: bn[x] + (nk[x] + 1 if nk[x] else 0) for x in set(bn) | set(nk)}

def _inessential(conj):
    """Per `Conj` fragment, the atoms a maximum matching can leave **unmatched**, and how many of
    them may be left unmatched **at once** (`CAPINESS`).

    `x` is inessential in fragment `F`  ⟺  `|M(F − x)| == |M(F)|`. Such an `x` does not have to
    take a π bond, so the ④ budget may charge it `k` rather than `k+1` (see the `CAPINESS`
    comment in `config`).

    🔴 **The count matters, not just the membership.** A fragment leaves exactly
    `deficiency = |F| − 2·|M(F)|` atoms over, so granting headroom to more than that many is a
    promise it cannot keep — and the extra ones then break the cap. Measured (holdout, before
    this cap): **every one of the 42 atoms that newly violated** sat in a fragment with
    deficiency 1 that had granted headroom to 2-9 atoms.

    Returns `[(fragment atoms that are inessential, deficiency), ...]`; the caller picks which
    `deficiency` of them to actually grant.
    A fragment with a perfect matching has none — removing any vertex drops the cardinality — so
    those are skipped and the per-vertex matchings run only where the deficiency is non-zero.
    """
    Gc = nx.Graph()
    Gc.add_edges_from(conj)
    out = []
    for cm in nx.connected_components(Gc):
        F = Gc.subgraph(cm)
        n = F.number_of_nodes()
        m0 = len(nx.max_weight_matching(F, maxcardinality=True))
        if 2 * m0 == n:                      # perfect matching — nobody can be left over
            continue
        ok = set()
        for x in cm:
            H = F.subgraph([v for v in cm if v != x])
            if len(nx.max_weight_matching(H, maxcardinality=True)) == m0:
                ok.add(x)
        if ok:
            out.append((ok, n - 2 * m0))
    return out


MILP_STAT = collections.Counter()   # {"solved", "fallback_size", "fallback_fail"} — diagnostic


def _solve_cap_exact(G, el, sc, conj, bml, ml_sc, ml_max, base):
    """Solve ④ **exactly** (`CAPMILP`). Same objective and constraints as the matching solve --
    only the approximation is removed.

    `base[a]` = what the atom already spends (`Conj` costs `k+1`, the unmatched-atom exception
    applied, a non-`Conj` bond 1.0, `b_ML` included). On top of that a `Double` adds +1, a
    `Triple` +2, and each M-L increment +1.

    Returns `(out, mlout)`, or `None` if it cannot be solved (the caller falls back to the
    matching solve).
    ⚠️ `scipy` is imported here and nowhere else -- with the flag off it is not a dependency.
    """
    import numpy as _np
    from scipy.optimize import Bounds, LinearConstraint, milp

    nonc = [e for a, b in G.edges for e in [(min(a, b), max(a, b))] if e not in conj]
    ed = [e for e in nonc if e in sc and 0 in sc[e]]
    mls = [(k, sm) for k, sm in (ml_sc or {}).items() if 1 in sm and min(sm) == 0]
    nv = 2 * len(ed) + 2 * len(mls)
    if nv == 0 or nv > CAPMILP_MAX:
        MILP_STAT["fallback_size"] += 1
        return None
    NEG = -1e6
    obj = _np.zeros(nv)
    for i, e in enumerate(ed):
        obj[2 * i] = sc[e].get(1, NEG) - sc[e][0]
        obj[2 * i + 1] = sc[e].get(2, NEG) - sc[e][0]
    off = 2 * len(ed)
    for j, (_k, sm) in enumerate(mls):
        obj[off + 2 * j] = sm[1] - sm[0]
        obj[off + 2 * j + 1] = (sm[2] - sm[1]) if (ml_max >= 2 and 2 in sm) else NEG
    A, lo, hi = [], [], []
    for i in range(len(ed)):                      # Double and Triple are exclusive
        row = _np.zeros(nv); row[2 * i] = row[2 * i + 1] = 1
        A.append(row); lo.append(-_np.inf); hi.append(1)
    for j in range(len(mls)):                     # the 2nd increment needs the 1st
        row = _np.zeros(nv); row[off + 2 * j] = -1; row[off + 2 * j + 1] = 1
        A.append(row); lo.append(-_np.inf); hi.append(0)
    at, mlat = collections.defaultdict(list), collections.defaultdict(list)
    for i, e in enumerate(ed):
        at[e[0]].append(i); at[e[1]].append(i)
    for j, (k, _sm) in enumerate(mls):
        mlat[k[1]].append(j)
    for a in set(at) | set(mlat):                 # valence ceiling
        cap = CAP.get(el[a])
        if cap is None:
            continue
        row = _np.zeros(nv)
        for i in at[a]:
            row[2 * i] += 1; row[2 * i + 1] += 2
        for j in mlat[a]:
            row[off + 2 * j] += 1; row[off + 2 * j + 1] += 1
        A.append(row); lo.append(-_np.inf); hi.append(cap - base[a])
    try:
        res = milp(c=-obj, constraints=LinearConstraint(_np.array(A), lo, hi),
                   integrality=_np.ones(nv), bounds=Bounds(0, 1))
    except Exception:
        MILP_STAT["fallback_fail"] += 1
        return None
    if not res.success or res.x is None:
        MILP_STAT["fallback_fail"] += 1
        return None
    x = _np.round(res.x).astype(int)
    out = {e: 3 for e in conj}
    for i, e in enumerate(ed):
        out[e] = 2 if x[2 * i + 1] else (1 if x[2 * i] else 0)
    for e in nonc:
        out.setdefault(e, 0)
    mlout = {k: min(sm) for k, sm in (ml_sc or {}).items()} if ml_sc else {}
    for j, (k, _sm) in enumerate(mls):
        mlout[k] = int(x[off + 2 * j]) + int(x[off + 2 * j + 1])
    MILP_STAT["solved"] += 1
    return out, mlout


def _solve_cap(G, el, sc, conj, bml, ml_sc=None, ml_max=2, iness_out=None):
    """The cap-respecting assignment — high likelihood subject to the valence cap, via a matching
    reduction (Blossom, polynomial time).

    🔴 **Not the exact maximum** (measured, the docstring used to claim it was).
    `Triple` is confirmed first, greedily, before the `Double`/M–L matching runs, so one `Triple`
    is never weighed against several `Double`s. Solving the identical objective under the
    identical constraints as a MILP and comparing: **113 of 12,245** ④ calls on holdout are
    suboptimal, median loss **4.58** (max 23.07) — about one bond decision. `CAPMILP=1` is that
    exact path; it is implemented and **off**, because the ④ metrics rise while the deployment
    output gets worse (see the `CAPMILP` comment in `config`).

    If `ml_sc` is given, **the M–L orders are decided inside the same optimization** (the order
    is raised one unit at a time, with a single dummy per unit so the same unit cannot be used
    twice ⇒ `ml_max=2` allows Triple).
    Returns `(internal classes, M–L classes)`.
    """
    k_of = collections.Counter()
    for e in conj:
        k_of[e[0]] += 1
        k_of[e[1]] += 1
    nonc = [e for a, b in G.edges for e in [(min(a, b), max(a, b))] if e not in conj]
    iness = set()
    if conj:
        # 🔴 Only `deficiency` atoms per fragment can be left unmatched at once, so grant that
        #   many — the ones with the most to gain, i.e. whose blocked non-`Conj` bond has the
        #   largest `Double − Single` margin. Ties and atoms with nothing to gain are dropped.
        for ok, defic in _inessential(conj):
            need = []
            for x in ok:
                # 🔴 The headroom `r[x]` gates **both** an internal `Double` and an M–L order
                #   raise, so the demand has to look at both — scoring internal bonds only sent
                #   the whole T8 `Triple` gain (F1 .639 → .734) back to baseline.
                cands = [
                    sc[e].get(1, -1e9) - sc[e].get(0, 0.0)
                    for y in G[x]
                    for e in [(min(x, y), max(x, y))]
                    if e not in conj and e in sc
                ]
                if ml_sc:
                    cands += [
                        sm[1] - sm[0]
                        for (_m, x_), sm in ml_sc.items()
                        if x_ == x and 1 in sm and 0 in sm
                    ]
                g = max(cands, default=-1e9)
                if g > 0:
                    need.append((g, x))
            need.sort(reverse=True)
            iness.update(x for _g, x in need[:defic])
    if iness_out is not None:
        iness_out.clear()
        iness_out.update(iness)
    use = collections.defaultdict(float)
    for x in G.nodes:
        # `k+1` assumes the Kekule matching pairs `x` up. `CAPINESS` charges `k` instead where a
        #   maximum matching leaving `x` unmatched exists.
        k = k_of[x]
        use[x] = (k + (0.0 if x in iness else 1.0) if k else 0.0) + bml.get(x, 0.0)
    for e in nonc:
        use[e[0]] += 1.0
        use[e[1]] += 1.0
    if CAPMILP:
        # Solve ④ exactly. Same objective and constraints as the matching solve below, minus
        #   the greedy `Triple` pass and the replicated-edge reduction. Falls back if unsolved.
        _r = _solve_cap_exact(G, el, sc, conj, bml, ml_sc, ml_max, use)
        if _r is not None:
            return _r
    out = {e: 3 for e in conj}
    for e in nonc:  # ① Triple — only where the likelihood argmax is Triple and both ends have
        #                        headroom of at least 2
        s3 = sc.get(e)
        if not s3 or max(s3, key=s3.get) != 2:
            continue
        if all(CAP.get(el[x], 4) - use[x] >= 2 - 1e-9 for x in e):
            out[e] = 2
            use[e[0]] += 2
            use[e[1]] += 2
    # 🔴 The capacity replicas let **one** bond take **two** units: with `r[a] = r[b] = 2` the
    #   matching can hold `(a,0)-(b,0)` and `(a,1)-(b,1)` at once, double-counting `g` and
    #   spending two units for a single `Double`. That is a broken reduction, not a tunable
    #   policy, so the repair is unconditional: whatever a round selected is fixed at **one**
    #   unit, its ends are charged once, and the wasted unit is offered to the other bonds.
    #   The loop stops when a round selects no duplicate (or after `CAPDUP_MAX` rounds).
    mlout = {}
    if ml_sc:
        for key, sm in ml_sc.items():
            # 🔴 The baseline is **the lowest class that exists for that pair** (fixed
            #). The old version pinned it to 0, which emitted `Single` for pairs
            #   whose T8 constant is `Double`/`Triple`, and it read `sm[0]` unconditionally and
            #   died with a KeyError on such pairs.
            mlout[key] = min(sm)
    fixed_int, fixed_ml, spent = set(), collections.Counter(), collections.Counter()
    for _rd in range(CAPDUP_MAX):
        r = {}
        for x in G.nodes:
            v = int(np.floor(CAP.get(el[x], 4) - use[x] - spent[x] + 1e-9))
            if v > 0:
                r[x] = min(v, 2)
        H = nx.Graph()
        for e in nonc:  # ② Double — maximum weight matching within the headroom
            if e in out or e in fixed_int:
                continue
            s3 = sc.get(e)
            if not s3 or 1 not in s3 or 0 not in s3:
                continue
            # 🔴 `TAUD` shifts the **weight**, not just the gate. Widening the gate alone does
            #   nothing: `max_weight_matching` never takes a negative edge, so an edge admitted
            #   with `g <= 0` is simply ignored (measured — τ = 0.5/1/2 were byte-identical to
            #   τ = 0). The cost-sensitive form is `g + τ > 0`, i.e. `τ = log(C_FN/C_FP)`.
            g = s3[1] - s3[0] + TAUD
            if g <= 0 or r.get(e[0], 0) < 1 or r.get(e[1], 0) < 1:
                continue
            for ia in range(r[e[0]]):
                for ib in range(r[e[1]]):
                    H.add_edge((e[0], ia), (e[1], ib), weight=g, e=e)
        if ml_sc:
            for key, sm in ml_sc.items():
                m_, x_ = key
                if 1 not in sm or min(sm) != 0 or r.get(x_, 0) < 1:
                    continue
                incs = [sm[1] - sm[0]]
                if ml_max >= 2 and 2 in sm:
                    incs.append(sm[2] - sm[1])
                for u, g in enumerate(incs):
                    if g <= 0:
                        break
                    if u < fixed_ml[key]:       # already committed in an earlier round
                        continue
                    du = ("_mlu", m_, x_, u)
                    for ia in range(r[x_]):
                        H.add_edge((x_, ia), du, weight=g, e=("ML", key))
        if not H.number_of_edges():
            break
        cnt, mlc = collections.Counter(), collections.Counter()
        for u, v in nx.max_weight_matching(H, maxcardinality=False):
            tg = H[u][v]["e"]
            if isinstance(tg, tuple) and tg and tg[0] == "ML":
                mlc[tg[1]] += 1
            else:
                cnt[tg] += 1
        dup = any(c >= 2 for c in cnt.values())
        for e in cnt:                            # one bond spends **one** unit
            out[e] = 1
            fixed_int.add(e)
            spent[e[0]] += 1
            spent[e[1]] += 1
        for key, c in mlc.items():
            fixed_ml[key] += c
            spent[key[1]] += c
        if not dup:
            break
    for key, c in fixed_ml.items():
        mlout[key] = min(c, 2)
    for e in nonc:
        out.setdefault(e, 0)
    return out, mlout

def _solve_sc(G, el, sc, ringA, coord, bml, lam_hi=10.0, lam_lo=10.0, maxit=50):
    """Local search on the likelihood under **two-sided valence constraints** — coordinating atoms
    are not penalized for being under-valent.

    This is the same search `D` and `D_satA` use. Here it is used **only to fix the conjugated
    set** (the orders themselves are decided again by the cap-respecting solve that follows).
    """
    cur = {}
    for a, b in G.edges:
        e = (min(a, b), max(a, b))
        cur[e] = max(sc[e], key=sc[e].get) if e in sc else 0
    sm = collections.defaultdict(float)
    for e, v in cur.items():
        sm[e[0]] += ORD4[v]
        sm[e[1]] += ORD4[v]
    inc = collections.defaultdict(list)
    for e in sc:
        inc[e[0]].append(e)
        inc[e[1]].append(e)

    def pn(x):
        if el[x] == "B":
            return 0.0
        hi = lam_hi * max(0.0, sm[x] + bml.get(x, 0.0) - CAP.get(el[x], 4))
        lo = 0.0 if x in coord else lam_lo * max(0.0, VTGT.get(el[x], 4) - sm[x])
        return hi + lo

    for _ in range(maxit):
        hot = {x for x in sm if pn(x) > 0}
        if not hot:
            break
        best = (1e-9, None, None)
        for e in {y for x in hot for y in inc[x]}:
            i2, j2 = e
            c0 = cur[e]
            base = sc[e][c0] - pn(i2) - pn(j2)
            for c1 in sc[e]:
                if c1 == c0:
                    continue
                dv = ORD4[c1] - ORD4[c0]
                sm[i2] += dv
                sm[j2] += dv
                nw = sc[e][c1] - pn(i2) - pn(j2)
                sm[i2] -= dv
                sm[j2] -= dv
                if nw - base > best[0]:
                    best = (nw - base, e, c1)
        if best[1] is None:
            break
        e, c1 = best[1], best[2]
        dv = ORD4[c1] - ORD4[cur[e]]
        sm[e[0]] += dv
        sm[e[1]] += dv
        cur[e] = c1
    return {e: (3 if e in ringA else v) for e, v in cur.items()}

def r6_swap(G, el, xyz, cls, bml=None, maxit=6):
    """R6 — for same-element bonds on one center, **make the bond-order ranking match the
    distance ranking**.

    Modifies `cls` in place. Returns the number of swaps. For the rule and its evidence see the
    `R6SWAP` comment.
    """
    if not R6SWAP:
        return 0
    bml = bml or {}
    ORD = [1.0, 2.0, 3.0, 1.5]
    n_sw = 0
    for _ in range(maxit):
        moved = False
        for x in G.nodes:
            by = collections.defaultdict(list)
            for y in G[x]:
                e = (min(x, y), max(x, y))
                if e not in cls or cls[e] == 3:  # Conj is out of scope
                    continue
                by[el[y]].append((float(np.linalg.norm(xyz[x] - xyz[y])), y, e))
            for _ey, lst in by.items():
                if len(lst) < 2:
                    continue
                lst.sort()
                for ii in range(len(lst)):
                    for jj in range(ii + 1, len(lst)):
                        (_d1, y1, e1), (_d2, _y2, e2) = lst[ii], lst[jj]
                        if cls[e1] >= cls[e2]:
                            continue  # the shorter one is already equal or higher
                        dv = ORD[cls[e2]] - ORD[cls[e1]]
                        # after the swap Y1 gains +dv and Y2 loses dv, so only Y1's cap needs
                        # checking.
                        b1 = sum(
                            ORD[cls[(min(y1, w), max(y1, w))]]
                            for w in G[y1]
                            if (min(y1, w), max(y1, w)) in cls
                        ) + bml.get(y1, 0.0)
                        if b1 + dv > CAP.get(el[y1], 4) + 1e-9:
                            continue
                        cls[e1], cls[e2] = cls[e2], cls[e1]
                        n_sw += 1
                        moved = True
        if not moved:
            break
    return n_sw

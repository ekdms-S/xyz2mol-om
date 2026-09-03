"""Search — exact solution under the valence cap (④) · local search for the conjugated set (③) ·
the R6 swap.

⚠️ **Ported from `ognm-bh-workspace/code/analysis/scratch/260830_fit_t10_charge.py`**
(2026-09-03). Function bodies were moved **verbatim** — the decision rules are unchanged.
"""

# ruff: noqa: E501
from __future__ import annotations

import collections

import networkx as nx
import numpy as np

from .config import CAP, ORD4, R6SWAP, VTGT
from .charge import _qfrag, frag_charge, q_atom


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

def _solve_cap(G, el, sc, conj, bml, ml_sc=None, ml_max=2):
    """**Exact solution** enforcing the cap only — the maximum-likelihood assignment among those
    that satisfy the cap (Blossom, polynomial time).

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
    use = collections.defaultdict(float)
    for x in G.nodes:
        use[x] = (k_of[x] + 1 if k_of[x] else 0.0) + bml.get(x, 0.0)
    for e in nonc:
        use[e[0]] += 1.0
        use[e[1]] += 1.0
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
    r = {}
    for x in G.nodes:
        v = int(np.floor(CAP.get(el[x], 4) - use[x] + 1e-9))
        if v > 0:
            r[x] = min(v, 2)
    H = nx.Graph()
    for e in nonc:  # ② Double — exact maximum weight matching within the headroom
        if e in out:
            continue
        s3 = sc.get(e)
        if not s3 or 1 not in s3 or 0 not in s3:
            continue
        g = s3[1] - s3[0]
        if g <= 0 or r.get(e[0], 0) < 1 or r.get(e[1], 0) < 1:
            continue
        for ia in range(r[e[0]]):
            for ib in range(r[e[1]]):
                H.add_edge((e[0], ia), (e[1], ib), weight=g, e=e)
    mlout = {}
    if ml_sc:
        for key, sm in ml_sc.items():
            m_, x_ = key
            # 🔴 The baseline is **the lowest class that exists for that pair** (fixed
            #   2026-09-03). The old version pinned it to 0, which emitted `Single` for pairs
            #   whose T8 constant is `Double`/`Triple`, and it read `sm[0]` unconditionally and
            #   died with a KeyError on such pairs.
            base = min(sm)
            mlout[key] = base
            if 1 not in sm or base != 0 or r.get(x_, 0) < 1:
                continue
            incs = [sm[1] - sm[0]]
            if ml_max >= 2 and 2 in sm:
                incs.append(sm[2] - sm[1])
            for u, g in enumerate(incs):
                if g <= 0:
                    break
                du = ("_mlu", m_, x_, u)
                for ia in range(r[x_]):
                    H.add_edge((x_, ia), du, weight=g, e=("ML", key))
    if H.number_of_edges():
        cnt = collections.Counter()
        for u, v in nx.max_weight_matching(H, maxcardinality=False):
            tg = H[u][v]["e"]
            if isinstance(tg, tuple) and tg and tg[0] == "ML":
                cnt[tg[1]] += 1
            else:
                out[tg] = 1
        for key, c in cnt.items():
            mlout[key] = min(c, 2)
    for e in nonc:
        out.setdefault(e, 0)
    return out, mlout

def _solve_sc(G, el, sc, ringA, coord, bml, lam_hi=10.0, lam_lo=10.0, maxit=50):
    """Local search on the likelihood under **two-sided valence constraints** — coordinating atoms
    are not penalized for being under-valent.

    This is the same search `D` and `D_satA` use. Here it is used **only to fix the conjugated
    set** (the orders themselves are decided again by the exact solution that follows).
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
    distance ranking** (2026-09-03).

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

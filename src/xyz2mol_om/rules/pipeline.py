"""The pieces of the rules the joint solve reads — ③ bond scores (the per-element-pair distance
likelihood), the T4 contact filters, and the T7 bridge tags. `docs/PIPELINE.md`.
"""

# ruff: noqa: E501
from __future__ import annotations

import collections

import networkx as nx
import numpy as np

from ..config import (MLIKE_EXTRA, AGOC, CAP, HALOGENS, HALW, LNORM_ON, LNORM_SKIP_CONJ, LPA, ORD4,
                     SATVETO, LPCOND, LPCOND_NOCONJ, ROPW, USE_ROP, VALENCE_3C)
from .likelihood import deg_cell


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


def bond_scores(el, xyz, G, scores4, rop=None):
    """③ — the per-element-pair 4-class distance likelihood of every internal bond.

    Returns `{(i, j): {class: score}}` (0 Single · 1 Double · 2 Triple · 3 Conj). A pair with no
    fitted entry in `scores4` is left out. The joint solve reads it for the `Conj` reading after
    the solve and for the π-suppression margin.
    """
    sc = {}
    for a, b in G.edges:
        e = (min(a, b), max(a, b))
        k = tuple(sorted((el[a], el[b])))
        if k not in scores4:
            continue
        ent = scores4[k]
        med, scl, lp = ent[0], ent[1], ent[2]
        d = float(np.linalg.norm(xyz[a] - xyz[b]))
        # 🔴 `LPCOND` — swap in the prior of this bond's **endpoint degree cell** (global `lp`
        #    if the cell is absent). With `LPCOND_NOCONJ`, only `Conj` (class 3) reverts to the
        #    global prior. For the rule see the `LPCOND` comment.
        lp_e = lp
        if LPCOND and len(ent) >= 6 and ent[5]:
            _cell = deg_cell(el, a, b, {x: G.degree(x) for x in (a, b)})
            _lc = ent[5].get(_cell)
            if _lc is not None:
                lp_e = {c: (lp[c] if (LPCOND_NOCONJ and c == 3) else v) for c, v in _lc.items()}
        sc[e] = {
            c: -abs(d - med[c]) / scl[c]
            + LPA * lp_e.get(c, lp[c])
            - (float(np.log(2 * scl[c])) if LNORM_ON and not (LNORM_SKIP_CONJ and c == 3) else 0.0)
            for c in med
        }
        if USE_ROP and rop is not None and len(ent) >= 5 and e in rop:
            rmed, rscl = ent[3], ent[4]
            rv = rop[e]
            for c in list(sc[e]):
                if c in rmed:
                    sc[e][c] += ROPW * (-abs(rv - rmed[c]) / rscl[c])
    return sc


# `MLIKE_EXTRA` is defined in `config` — imported above.


def drop_agostic(el, G, ml_raw):
    """Remove `C–H···M` only — μ-H and `B–H···M` (borohydride) are genuine 3c2e and are kept
    (`docs/PIPELINE.md`).

    rule  remove ⟺ el[X] = H  AND  exactly 1 metal-like neighbor  AND  some internal neighbor is
                   not metal-like
    """
    nmet = collections.Counter(x for _m, x in ml_raw)
    out = []
    for m, x in ml_raw:
        if el[x] == "H":
            n_like = nmet[x] + sum(1 for y in G[x] if el[y] in MLIKE_EXTRA)
            if n_like == 1 and any(el[y] not in MLIKE_EXTRA for y in G[x]):
                continue
        out.append((m, x))
    return out


def drop_agostic_carbon(el, xyz, G, ml_raw, cut=None):
    """Also remove the **carbon** candidate of a `C–H···M` contact (`AGOC`) — unless that would
    detach the fragment from that metal.

    `drop_agostic` removes only the M–**H**. The M–C left behind is the problem: in an agostic
    contact the metal does not take the carbon's fourth bond, it **borrows the electron pair of
    the existing C–H bond**. This format has only `sigma` (spends one valence unit) and `haptic`
    (spends none), so the leftover M–C could only be `sigma`, and that unit saturates the carbon:
    no π, a dearomatized ring, carbanions, and a metal oxidation state two too high.

    Rule (geometry only — no fitted parameter)::

        el[X] = C  AND  X carries an H  AND  d(M,H) < d(M,C)  AND  d(M,H) < `cut` (default `AGOC`)

    ⚠️ **A (metal, fragment) pair the removal would leave with no M–L candidate is left
    untouched** — removing it would turn an overcounted σ into a false dissociation.
    """
    cut = AGOC if cut is None else cut
    if not cut or not ml_raw:
        return ml_raw
    comp = {x: i for i, c in enumerate(nx.connected_components(G)) for x in c}
    drop = set()
    for m, x in ml_raw:
        if el[x] != "C":
            continue
        hs = [h for h in G[x] if el[h] == "H"]
        if not hs:
            continue
        dmc = float(np.linalg.norm(xyz[m] - xyz[x]))
        dmh = min(float(np.linalg.norm(xyz[m] - xyz[h])) for h in hs)
        if dmh < dmc and dmh < cut:
            drop.add((m, x))
    if not drop:
        return ml_raw
    # a (metal, fragment) pair left with no contact after the removal keeps all of its candidates
    left = collections.Counter()
    for m, x in ml_raw:
        if (m, x) not in drop:
            left[(m, comp.get(x))] += 1
    drop = {(m, x) for m, x in drop if left[(m, comp.get(x))]}
    return [p for p in ml_raw if p not in drop]


def drop_saturated(el, G, ml_raw):
    """Remove an M–X candidate to an atom whose **internal neighbours already fill its valence**
    (`SATVETO`).

        remove ⟺ el[X] ∉ {H} ∪ {B, Al}  AND  no internal neighbour of X is B or Al
                 AND  deg_int(X) ≥ CAP(el[X])

    `deg_int` is the **number of internal neighbours**, not the bond-order sum. T4 runs before ③,
    so no order exists yet; and since every bond order is ≥ 1, `deg_int ≥ CAP` already implies
    `b_int ≥ CAP`. Do not switch it to `b_int`: an η²-alkene carbon has `deg 3` but
    `b_int 4 = CAP`, so a `b_int` test would veto every alkene, arene and Cp coordination.

    H is excluded — `CAP(H) = 1` and an H always has one internal neighbour, so the test would
    veto every M–H bond, μ-H and borohydride included. Agostic `C–H···M` is `drop_agostic`'s job.
    B and Al are excluded, **and so is any atom bonded to one**, because cluster bonding
    (carborane) is outside the two-centre formalism — the same exception the valence-violation
    tally makes. A dicarbollide cage carbon has `deg 5 > CAP 4`, which says nothing about the
    metal.
    """
    if not SATVETO:
        return ml_raw
    return [(m, x) for m, x in ml_raw
            if el[x] == "H" or el[x] in MLIKE_EXTRA
            or any(el[y] in MLIKE_EXTRA for y in G[x])
            or G.degree(x) < CAP.get(el[x], 99)]


def drop_bound_halide(el, G, ml_raw, wbo):
    """Remove a weak M–X candidate where **X is a halogen that already carries an internal
    covalent bond** (`HALW`).

        remove ⟺ el[X] ∈ {F, Cl, Br, I}          — never part of a π system, so never haptic
                 AND X has ≥ 1 internal neighbour  — i.e. it is not a terminal halide
                 AND w(M, X) < HALW

    A terminal halide (`M–Cl⁻`, `M–F⁻`) has **no** internal neighbour and is never touched, so
    the ordinary halide ligand is untouched. What this catches is the fluorine of a `CF₃`, a
    triflate, a `BF₄⁻` or a `PF₆⁻` sitting 2.6 Å from a late metal — a contact `d_bond.csv`
    cannot veto because it has no fitted row for the pair (or a `w_veto` of 0), and the radii
    fallback opens a 2.8 Å window with no Mayer floor at all.

    🔴 It does **not** remove a genuine oxidative-addition halide. At an `R–I`/`R–Cl` oxidative
    addition the halogen is still bonded to carbon *and* genuinely bonded to the metal, and its
    Mayer order sits well above `HALW`, while a bound `F` contact sits well below it.

    With `wbo=None` there is nothing to test, so nothing is removed.
    """
    if HALW <= 0 or not wbo:
        return ml_raw
    return [
        (m, x) for m, x in ml_raw
        if el[x] not in HALOGENS
        or G.degree(x) == 0
        or wbo.get((m, x), 1.0) >= HALW
    ]


def is_3c2e(el0, b_use, n_center):
    """The **raw predicate** of the T7 3c2e decision, used by `bridge_tags` — the rule lives here
    and nowhere else.

    `b_use`    = `b_int(X) + n_ML(X)` — what X has **used**. Internal bonds count by their
                 **order** (Kekule sum), M–L bonds by their **number** (one donated lone pair
                 each; under the ionic cut an M–L bond carries no order at all). The two halves
                 use different units on purpose. (`deg(X)` in PIPELINE.md's notation is the
                 plain internal neighbour count — a different quantity.)
    `n_center` = (number of M–L bonds) + (number of internal neighbors whose element is B or Al)
    rule  3c2e ⟺ n_center >= 2  AND  el0 ∈ VALENCE_3C  AND  b_use > VALENCE_3C[el0]

    `VALENCE_3C[el]` is **not** a neutral-atom valence — it is the closed-shell budget of the
    coordinating atom, (bonds + lone pairs). Since `VALENCE_3C[el] - b_int(X) = n_lp(X)`, the
    rule is equivalent to `n_ML > n_lp`: *X is donating to more centers than it has lone pairs
    for, so one pair has to be shared.* That is the chemical criterion the tag encodes.
    """
    v0 = VALENCE_3C.get(el0)
    return n_center >= 2 and v0 is not None and b_use > v0


def bridge_tags(el, G, ml_pred, cls, hap=()):
    """T7 (`docs/PIPELINE.md`) — the **bridge tag** per coordinating atom.
    Returns `{x: "3c2e" | "dative"}`.

    An atom that is not a bridge **has no key at all.**

    `cls` — **pass-1 internal bond classes**, `{(i,j): 0|1|2|3}`. `b_int` is read from these,
       so an atom's multiple internal bond (the C≡O of a bridging carbonyl) counts as its order.
       Required — there is no neighbour-count fallback.

       ⚠️ It must be the **pass-1** classes, not pass-2. Pass 2 needs the tag to build its
          budget, so reading pass-2 orders here would be circular. Pass 1 runs with no metal
          budget at all — the same trick the provisional haptic set uses (pass-1 π fragments →
          budget → pass 2).

    `hap` — haptic M–L bonds; they count toward `n_center` but not toward `b_use`.

    rule

        n_center(X) = (number of M–L bonds of X) + (number of internal neighbors of X whose
                                                   element is B or Al)
        b_use(X)    = (internal bond orders of X, pass-1 Kekule count) + (number of M–L bonds of X)

        bridge(X) ⟺ n_center(X) >= 2
        3c2e(X)   ⟺ bridge(X)  AND  el[X] ∈ {H, C, Si, B}  AND  b_use(X) > VALENCE_3C[el[X]]
                                                              (H 1 · C·Si 4 · B 3)
        dative(X) ⟺ bridge(X)  AND  3c2e(X) is false

    Examples (`n_center` · `b_use` · tag)

        μ-H       M–H–M         n_center 2 · b_use 2 (internal 0 + M–L 2)   →  **3c2e**
        μ-CO      M–CO–M        n_center 2 · b_use 5 (C≡O 3 + M–L 2) > 4    →  **3c2e**
        B–H···M   borohydride   n_center 2 (M 1 + neighbor B 1) · b_use 2   →  **3c2e**
        μ-Cl      M–Cl–M        n_center 2 · b_use 2 · Cl is not in the table → **dative** (3c4e)
        terminal Cl  M–Cl       n_center 1                                  →  no tag

    ⚠️ `ml_pred` is the set of T4 bonds **after agostic removal and including haptic** — haptic
       (`Pi`) M–L bonds count toward the metal count.
    """
    # 🔴 **A haptic M–L spends nothing**, so it must not enter the electron budget `b_use` —
    #   (otherwise a boryl-substituted η⁵ Cp carbon,
    #   `b_int 4 = CAP`, reads `4 + 1 = 5 > 4`). It still counts toward `n_center` — the atom
    #   *is* connected to that metal, which is the question `n_center` asks.
    _hap = {(m, x) for m, x in hap}
    nmet = collections.Counter(x for _m, x in ml_pred)
    nbud = collections.Counter(x for m, x in ml_pred if (m, x) not in _hap)
    bint = _kek_val(G, el, cls)
    tags = {}
    # 🔴 Do not narrow the candidates by `nmet` — **an atom with 0 M–L bonds can also be a
    #   bridge.** The H of `B–H–B` has 0 M–L bonds and is a bridge purely through its 2 internal
    #   B neighbors; a loop keyed on `nmet` would **miss that H entirely.** Keep the rule
    #   separate from the center-atom definition.
    for x in G.nodes():
        nm = nmet.get(x, 0)
        # 🔴 **A metal-like atom does not bridge to its own kind.** The `MLIKE_EXTRA` term is
        #   here so a *non-metal* can be seen bridging two borons with no metal in sight
        #   (`B–H–B`). For an `x` that is itself `B`/`Al` the neighbouring B is a substituent, not
        #   a third centre (a diboranyl `M–B(Mes)=B(Mes)Br` has an ordinary `B=B`).
        #   ⚠️ The term is **not** limited to `el[x] == "H"`: boryl-substituted Cp carbons and
        #      cage carbons keep their tag, and the valence tally relies on it (they are outside
        #      the two-centre formalism).
        n_like = 0 if el[x] in MLIKE_EXTRA else sum(1 for y in G[x] if el[y] in MLIKE_EXTRA)
        n_center = nm + n_like
        if n_center < 2:
            continue
        b_use = bint.get(x, 0.0) + nbud.get(x, 0)
        tags[x] = "3c2e" if is_3c2e(el[x], b_use, n_center) else "dative"
    return tags



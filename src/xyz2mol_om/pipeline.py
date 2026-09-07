"""★ Adopted pipeline — `predict_T3_EHT` ([design doc] §3 `1c`).

⚠️ **Ported from `ognm-bh-workspace/code/analysis/scratch/260830_fit_t10_charge.py`**
(2026-09-03). Function bodies were moved **verbatim** — the decision rules are unchanged.
"""

# ruff: noqa: E501
from __future__ import annotations

import collections

import networkx as nx
import numpy as np

from .config import (ADJQVETO, ADJQW, BML3C_COST, CAPINESS, CAP, EHTCOST, EHTMINFRAG, EHTSKIP, LNORM_ON, LNORM_SKIP_CONJ, LPA, ORD4,
                     LPCOND, LPCOND_NOCONJ, R2CONJ, R5SOLO, ROPW, TAU_P, USE_ROP, R7MIN, R7RING, THETA_HAPTIC,
                     VALENCE_3C,)
from .charge import _qfrag, atom_bond_sums, q_atom
from .conjugation import conj_forbidden, lp_donor, rule_a_ok
from .eht import eht_frag_charges
from .geometry import plane_rms
from .likelihood import deg_cell
from .solvers import _kek_val, _solve_cap, _solve_sc, r6_swap
from .ml_order import load_b_ml_mayer, ml_order_scores, ml_order_scores_dist


def _sgn(q):
    return (q > 1e-9) - (q < -1e-9)


def _adjq_pairs(G, el, e, db, bs, qn, deg, nbrs):
    """`ADJQW` (proposal 1) — how many **adjacent same-sign nonzero formal-charge pairs** the
    move `e: order += db` adds (2026-09-07).

    Only the two endpoints of `e` change charge (see `atom_bond_sums`), so only the bonds
    touching them can gain or lose such a pair — the count runs over exactly those bonds, with
    `e` itself counted once.
    Returns `max(0, after − before)`: this is a **penalty, not a reward** — a move that removes
    a same-sign pair is not paid a bonus, so the baseline (`ADJQW=0`) is a strict subset.
    """
    a, b = e
    q2 = dict(qn)
    for v in (a, b):
        q2[v] = q_atom(el[v], bs[v] + db, deg[v], nbrs[v])
    ed = {(min(v, w), max(v, w)) for v in (a, b) for w in G[v]}
    def cnt(q):
        n = 0
        for u, v in ed:
            su = _sgn(q.get(u, 0.0))
            if su and su == _sgn(q.get(v, 0.0)):
                n += 1
        return n
    return max(0, cnt(q2) - cnt(qn))


def predict_T3_EHT(el, xyz, G, scores4, bml=None, ml_sc=None, q_eht=None, coord=None, rop=None,
                   w_out=None):
    """★ Adopted option `D_eht` — all stages of [design doc] §3 `1c`.
    Returns `(internal classes, M–L classes)`.

    `bml`   {coordinating atom: sum of M–L bond orders} — enters the capacity budget
            (baseline Single)
    `ml_sc` {(metal, coordinating atom): {class: score}} — if given, M–L orders are optimized
            jointly
    `q_eht` {min atom idx of fragment: charge} — computed via `eht_frag_charges` if omitted
    `rop`   {(i,j): EHT overlap density} — the second dimension, when `USE_ROP=1` and the
            scores4 entry is a 5-tuple
    `coord` set of coordinating atoms — **waives the under-valence penalty** in the conjugation
            search (M–L absorbs it)
    `w_out` optional dict — filled with `{edge: score[Double] − score[Single]}`, the tie-break
            weight the ⑥ Kekule matching uses under `CONJW`. The return signature is unchanged
            because `predict_T3_EHT` is part of the public API.

    ⚠️ **Haptic M–L bonds must not go into `bml`** — a haptic bond gets no order and is shared
       across the π system, so it is not attributed to an atom ([design doc] §3 `5a`). Including
       it would waste the budget of the Cp carbons.
    """
    bml = bml or {}
    if q_eht is None:
        q_eht = eht_frag_charges(el, xyz, G)
    # ① rule A — applies only to atoms of a planar ring (size >= 5) that **still have π headroom**
    sat = {x for x in G.nodes if G.degree(x) >= CAP.get(el[x], 4)}
    ringA = set()
    for r in nx.cycle_basis(G):
        if rule_a_ok(len(r)) and plane_rms(xyz[np.array(r)]) <= TAU_P:
            ringA |= {
                (min(a, b), max(a, b))
                for a, b in zip(r, r[1:] + r[:1])
                if a not in sat and b not in sat
            }
    # ② per-element-pair 4-class distance likelihood
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
    # ③ conjugated set — decided on a likelihood where bonds touching a saturated atom are
    #   allowed to be Single only
    sc_sat = {
        e: ({0: v[0]} if (e[0] in sat or e[1] in sat) and 0 in v else dict(v))
        for e, v in sc.items()
    }
    if R2CONJ:  # ★ R2 — forbid `Conj` at pyrrole-type heteroatoms (see the comment above)
        qf = {}
        if q_eht:
            for comp0 in nx.connected_components(G):
                qf.update(dict.fromkeys(comp0, q_eht.get(min(comp0), 0)))
        _bad = conj_forbidden(G, el, qf, xyz)
        ringA -= _bad
    conj = {e for e, v in _solve_sc(G, el, sc_sat, ringA, coord or set(), bml).items() if v == 3}
    if R2CONJ:
        conj -= _bad
    if R5SOLO and conj:  # R5 — an isolated `Conj` (no neighboring `Conj` bond) is not
        #                       delocalized
        Gj = nx.Graph()
        Gj.add_edges_from(conj)
        conj = {e for e in conj if Gj.degree(e[0]) > 1 or Gj.degree(e[1]) > 1}
    w = {e: v.get(1, 0.0) - v.get(0, 0.0) for e, v in sc.items()}
    # ④ hard valence-cap constraint — exact solution (M–L up to Triple)
    iness = set()
    cls, mlout = _solve_cap(G, el, sc, conj, bml, ml_sc, ml_max=2, iness_out=iness)
    if CAPINESS and iness:
        # 🔴 `CAPINESS` gave these atoms headroom **on the promise that ⑥ leaves them unmatched**.
        #   ⑥ runs its own matching and will happily pair one up, and then the cap it was granted
        #   against is broken — measured: valence violations +0.7%p on train before this coupling.
        #   So the promise is carried into ⑥ as a large negative weight on every edge touching an
        #   atom that is now **at capacity**. `maxcardinality=True` still holds, so this cannot
        #   shrink the matching — and a maximum matching leaving the atom out is exactly what the
        #   `_inessential` test verified exists.
        k_of = collections.Counter()
        for e in cls:
            if cls[e] == 3:
                k_of[e[0]] += 1
                k_of[e[1]] += 1
        for x in iness:
            u = bml.get(x, 0.0) + sum(
                ORD4[cls[(min(x, y), max(x, y))]]
                for y in G[x]
                if cls.get((min(x, y), max(x, y))) != 3
            )
            if u + k_of[x] + 1 > CAP.get(el[x], 4) + 1e-9:
                for y in G[x]:
                    e = (min(x, y), max(x, y))
                    if cls.get(e) == 3:
                        w[e] = w.get(e, 0.0) - 1e6
    # 🔴 sync **after** the `CAPINESS` penalty — filling `w_out` before it meant ⑥ (which runs in
    #   `api.predict` on the returned dict) never saw the penalty, and the granted atom got paired
    #   up anyway. Measured: 33 atoms newly violated the cap, every one of them a granted atom in
    #   a fragment with deficiency 1 and exactly 1 grant, i.e. a promise that *was* keepable.
    if w_out is not None:
        w_out.clear()
        w_out.update(w)
    # ⑤ EHT fragment-charge target
    for comp0 in nx.connected_components(G):
        comp = set(comp0)
        tgt = q_eht.get(min(comp))
        if tgt is None:
            continue
        if len(comp) < EHTMINFRAG:
            continue  # do not trust the EHT target on small fragments (see `EHTMINFRAG` above)
        if EHTSKIP and "".join(sorted(el[x] for x in comp)) in EHTSKIP:
            continue  # for this composition the EHT target is systematically wrong (`EHTSKIP`)
        edges = [e for e in cls if e[0] in comp and cls[e] != 3 and e in sc]
        snap = {e: cls[e] for e in edges}
        cost = 0.0
        for _ in range(12):
            d = tgt - round(_qfrag(G, el, cls, comp, w))
            if d == 0 or abs(d) % 2 == 1:
                break
            use = collections.defaultdict(float, _kek_val(G, el, cls))
            for x in comp:
                use[x] += bml.get(x, 0.0)
            # `ADJQW` — charges of the fragment as it stands, recomputed once per ±1 step
            #   (the candidate scan below only shifts two atoms of them).
            bs = qn = DEGa = NBa = None
            if ADJQW > 0.0 or ADJQVETO:
                bs, DEGa, NBa = atom_bond_sums(G, el, cls, comp, w)
                qn = {v: q_atom(el[v], bs[v], DEGa[v], NBa[v]) for v in comp}
            best = None
            for e in edges:
                c0 = cls[e]
                if d > 0:
                    if c0 >= 2 or any(CAP.get(el[x], 4) - use[x] < 1 - 1e-9 for x in e):
                        continue
                    c1 = c0 + 1
                else:
                    if c0 <= 0:
                        continue
                    c1 = c0 - 1
                g = sc[e].get(c1, -1e9) - sc[e].get(c0, 0.0)
                if bs is not None:
                    adj = _adjq_pairs(G, el, e, ORD4[c1] - ORD4[c0], bs, qn, DEGa, NBa)
                    if adj and ADJQVETO:
                        continue  # rejected — ⑤ takes the next-best move, or gives up
                    g -= ADJQW * adj
                if best is None or g > best[0]:
                    best = (g, e, c1)
            if best is None:
                break
            cost += -best[0]
            if EHTCOST >= 0 and cost > EHTCOST:  # give up the target and revert
                for e, c in snap.items():
                    cls[e] = c
                break
            cls[best[1]] = best[2]
    r6_swap(
        G, el, xyz, cls, bml
    )  # ⑥ R6 — align distance and order ranking for same-element bonds on one center
    #      (a swap, so the total is unchanged)
    return cls, mlout


# ★★ unified entry point (2026-09-03) — **T3, M–L order, haptic (T5) and R7 all come out of one
#   function.**
#   Why: the scorer (`260831_propagation_prior_cv.py`) and the release (`xyz2mol-om`) each
#   assembled the same decisions themselves, and they diverged in **four places** (measured
#   2026-09-03, 0.12-0.90% of bonds):
#     ① was haptic removed from the `b_ML` budget passed to the ④ cap solution?
#     ② was haptic removed from the M–L order candidates?
#     ③ was agostic (`C–H···M`) removed?
#     ④ are T5's Y candidates the fragment neighbors or all neighbors?
#   ⇒ **Assembly is not left to the caller.** The caller supplies only the T4 candidates
#     (`ml_raw`) and Mayer (`wbo`).
#   ⚠️ The body of this function must **stay identical to** the same-named function in the release
#      `xyz2mol-om/src/xyz2mol_om/pipeline.py`. Do not change only one side.
MLIKE_EXTRA = {"B", "Al"}  # metal-like = metals ∪ {B, Al} ([design doc] §3.1 (c))


def drop_agostic(el, G, ml_raw):
    """Remove `C–H···M` only — μ-H and `B–H···M` (borohydride) are genuine 3c2e and are kept
    ([design doc] §3.0 [T4]).

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


def is_3c2e(el0, b_use, n_center):
    """The **raw predicate** of the T7 3c2e decision — shared by `bridge_tags` and the scorer
    (the rule lives in one place).

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


def bridge_tags(el, G, ml_pred, cls):
    """T7 ([design doc] §3.0 5c) — the **bridge tag** per coordinating atom.
    Returns `{x: "3c2e" | "dative"}`.

    An atom that is not a bridge **has no key at all.**

    `cls` — **pass-1 internal bond classes**, `{(i,j): 0|1|2|3}`. `b_int` is read from these,
       so an atom's multiple internal bond (the C≡O of a bridging carbonyl) counts as its order.
       Required — there is no neighbour-count fallback.

       ⚠️ It must be the **pass-1** classes, not pass-2. Pass 2 needs the tag to build its
          budget, so reading pass-2 orders here would be circular. Pass 1 runs with no metal
          budget at all — the same trick the provisional haptic set uses (pass-1 π fragments →
          budget → pass 2).
       ⚠️ The scorer `260831_propagation_prior_cv.py:is_3c2e` computes this the old way
          (neighbour count) and must be updated alongside.

    rule (the same formula as the scorer `260831_propagation_prior_cv.py:is_3c2e`)

        n_center(X) = (number of M–L bonds of X) + (number of internal neighbors of X whose
                                                   element is B or Al)
        b_use(X)    = (internal bond orders of X, pass-1 Kekule count) + (number of M–L bonds of X)

        bridge(X) ⟺ n_center(X) >= 2
        3c2e(X)   ⟺ bridge(X)  AND  el[X] ∈ {H, C, Si, B}  AND  b_use(X) > VALENCE_3C[el[X]]
                                                              (H 1 · C·Si 4 · B 3)
        dative(X) ⟺ bridge(X)  AND  3c2e(X) is false

    5 real cases (`n_center` · `b_use` · tag)

        μ-H       M–H–M         n_center 2 · b_use 2 (internal 0 + M–L 2)   →  **3c2e**
        μ-CO      M–CO–M        n_center 2 · b_use 5 (C≡O 3 + M–L 2) > 4    →  **3c2e**
        B–H···M   borohydride   n_center 2 (M 1 + neighbor B 1) · b_use 2   →  **3c2e**
        μ-Cl      M–Cl–M        n_center 2 · b_use 2 · Cl is not in the table → **dative** (3c4e)
        terminal Cl  M–Cl       n_center 1                                  →  no tag

    ⚠️ `ml_pred` is the set of T4 bonds **after agostic removal and including haptic** — the scorer
       also counts `Pi` (haptic) M–L bonds toward the metal count, so the same input is used.
    """
    nmet = collections.Counter(x for _m, x in ml_pred)
    bint = _kek_val(G, el, cls)
    tags = {}
    # 🔴 Do not narrow the candidates by `nmet` — **an atom with 0 M–L bonds can also be a
    #   bridge.** Right now B and Al are in `METALS` so they never appear as internal neighbors,
    #   but once `B` is treated as a ligand atom, the H of `B–H–B` has 0 M–L bonds and becomes a
    #   bridge purely through its 2 internal B neighbors. If this loop were keyed on `nmet` it
    #   would **miss that H entirely.** Keep the rule separate from the center-atom definition.
    #   ⚠️ Present behavior is unchanged — while B and Al are in `METALS` the internal-neighbor
    #      term is always 0.
    for x in G.nodes():
        nm = nmet.get(x, 0)
        n_center = nm + sum(1 for y in G[x] if el[y] in MLIKE_EXTRA)
        if n_center < 2:
            continue
        b_use = bint.get(x, 0.0) + nm
        tags[x] = "3c2e" if is_3c2e(el[x], b_use, n_center) else "dative"
    return tags


def bml_budget(ml_bonds, three_c, cost=None):
    """The ④·⑥ `b_ML` budget — `{coordinating atom: budget}`.

    Every M–L bond costs 1.0, **except** that an atom taking part in a 3c2e bond spends
    `cost` in total no matter how many M–L bonds it has (`config.BML3C_COST`, default 1.0).
    `cost < 0` restores the pre-2026-09-06 behaviour of one unit per bond.

    ⚠️ `ml_bonds` must already have the bonds that spend nothing removed — haptic for ④
       (`keep`), the final haptic set for ⑥.
    """
    cost = BML3C_COST if cost is None else cost
    bml = collections.defaultdict(float)
    counted = set()
    for _m, x in ml_bonds:
        if x in three_c and cost >= 0.0:
            if x not in counted:
                counted.add(x)
                bml[x] += cost
        else:
            bml[x] += 1.0
    return bml


def _angle_ok(xyz, m, x, y, theta):
    """∠(M–X–Y) < theta ?  Y is chosen by the caller."""
    v1, v2 = xyz[m] - xyz[x], xyz[y] - xyz[x]
    cs = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12))
    return float(np.degrees(np.arccos(max(-1.0, min(1.0, cs))))) < theta


def _closest_mid(xyz, x, m, cand):
    """Among X's neighbor candidates, the one **whose bond midpoint is closest to M**."""
    return min(cand, key=lambda q: float(np.linalg.norm((xyz[x] + xyz[q]) / 2 - xyz[m])))


def predict_T3_T5(el, xyz, G, scores4, ml_raw, wbo, bml_model=None, bml_fb=None,
                  q_eht=None, rop=None):
    """Takes only the T4 candidates and Mayer, and produces **the T3 4 classes, the M–L orders and
    the haptic set** end to end.

    Returns `(cls, mlout, hap, ml_pred, btag, w)`
      `cls`     {(i,j): 0 Single · 1 Double · 2 Triple · 3 Conj}   internal bonds
      `mlout`   {(m,x): class}                                     M–L orders (haptic excluded)
      `hap`     {(m,x)}                                            haptic M–L bonds
      `ml_pred` [(m,x)]                                            T4 bonds with agostic removed
      `btag`    {x: "3c2e" | "dative"}                             T7 bridge tags (pass-1 based)
      `w`       {(i,j): score[Double] − score[Single]}              ⑥ Kekule tie-break (`CONJW`)

    Why 2 passes: a haptic M–L bond **gets no order and spends no budget** ([design doc] §3 5a).
    But whether a bond is haptic can only be decided once T3 (the π fragments) is known. So the
    **first pass solves T3 with a budget of 0** to get the π candidates, haptic bonds are decided
    provisionally **from the angle alone**, and the second pass is solved with those removed from
    the budget.
    ⚠️ The provisional decision is for the budget only — **the final haptic set is decided again
    from the π fragments of the second pass.**
    """
    if bml_model is None:
        bml_model, bml_fb = load_b_ml_mayer()
    coord = {x for _m, x in ml_raw}
    ml_pred = drop_agostic(el, G, ml_raw)
    # pass 1 — budget 0 · no M–L optimization
    cls0, _ = predict_T3_EHT(el, xyz, G, scores4, {}, None, q_eht, coord, rop)
    unsat0 = {x for e, v in cls0.items() if v in (1, 2, 3) for x in e}
    hap_pre = set()
    for m, x in ml_pred:
        nb = list(G[x])
        if x in unsat0 and nb and _angle_ok(xyz, m, x, _closest_mid(xyz, x, m, nb), THETA_HAPTIC):
            hap_pre.add((m, x))
    keep = [p for p in ml_pred if p not in hap_pre]
    # 🔴 T7 bridge tags — computed **here**, between the two passes, because pass 2's budget
    #   depends on them. `cls0` (pass-1, metal-free orders) is what makes the bond-order form of
    #   the rule non-circular. The same tags are returned so the output cannot diverge from the
    #   budget decision.
    #   A unit whose single electron pair spans 3 centers is **one** bond of valence, so those
    #   atoms spend `BML3C_COST` (default 1.0) in total rather than 1.0 per M–L bond.
    #   ⚠️ They are **not** removed from `keep` (= the M–L order optimization candidates) — T8
    #      still assigns orders to them.
    btag = bridge_tags(el, G, ml_pred, cls0)
    three_c = {x for x, tg in btag.items() if tg == "3c2e"}
    bml = bml_budget(keep, three_c)  # M–L baseline = Single, 3c2e = one pair
    # 🔴 With no `wbo`, M–L orders come from the **distance fallback** (2026-09-03).
    #   Without Mayer, T8 emits `Single` for every bond (`Double` F1 **0.0000** · measured over
    #   300 structures, TP 0 / FN 40). On the same pool with refcode 5-fold CV, the monotone
    #   distance-threshold fallback gives `Double` **0.6976** · `Triple` 0.6515 (Mayer version
    #   .7318 / .7230 · trivial baseline 0.0000).
    #   ⚠️ It is **not used when `wbo` is available** — distance clearly loses (`Double` −0.034).
    if wbo:
        ml_sc = ml_order_scores(el, keep, wbo, bml_model, bml_fb)
    else:
        ml_sc = ml_order_scores_dist(el, keep, xyz)
    # pass 2 — this is the output
    w = {}
    cls, mlout = predict_T3_EHT(el, xyz, G, scores4, dict(bml), ml_sc, q_eht, coord, rop, w_out=w)
    # T5 — the final haptic set. The Y candidates are **neighbors in the same π fragment**
    #      (measured 2026-09-03: fragment neighbors F1 .9810 · all internal neighbors .9803 —
    #      precision is higher for fragment neighbors).
    pi = nx.Graph()
    pi.add_edges_from(e for e, v in cls.items() if v in (1, 2, 3))
    pifrag = {x: i for i, c in enumerate(nx.connected_components(pi)) for x in c}
    hap, hap_by_m = set(), collections.defaultdict(set)
    for m, x in ml_pred:
        if x not in pifrag:
            continue
        nb = [y for y in G[x] if pifrag.get(y) == pifrag[x]]
        if nb and _angle_ok(xyz, m, x, _closest_mid(xyz, x, m, nb), THETA_HAPTIC):
            hap.add((m, x))
            hap_by_m[m].add(x)
    # R7 — restore an R2 donor inside a haptic ring as a π candidate (adopted 2026-09-03 ·
    #      the rule is [design doc] §3.1-R7)
    if R7RING:
        mlset = set(ml_pred)
        donors = {x for x in G if lp_donor(el[x], G.degree(x))}
        for r5 in (nx.cycle_basis(G) if donors else []):
            if len(r5) != 5:
                continue
            din = [x for x in r5 if x in donors]
            if not din:
                continue
            for m in list(hap_by_m):
                if len(set(r5) & hap_by_m[m]) < R7MIN:
                    continue
                for x in din:
                    if x in hap_by_m[m] or (m, x) not in mlset:
                        continue
                    nb = [y for y in G[x] if y in pifrag]
                    if nb and _angle_ok(xyz, m, x, _closest_mid(xyz, x, m, nb), THETA_HAPTIC):
                        hap.add((m, x))
                        hap_by_m[m].add(x)
    for e in hap:  # haptic bonds get no order
        mlout.pop(e, None)
    return cls, mlout, hap, ml_pred, btag, w

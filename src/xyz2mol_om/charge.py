"""Formal charge — per atom `q_atom` · conjugated fragment `frag_charge` · fragment sum
`_qfrag` · output converter `kekulize`.

⚠️ **Ported from `ognm-bh-workspace/code/analysis/scratch/260830_fit_t10_charge.py`**
(2026-09-03). Function bodies were moved **verbatim** — the decision rules are unchanged.
"""

# ruff: noqa: E501
from __future__ import annotations


import networkx as nx

from .config import (ALT, CAP, FULL, HUCKEL, NAMEEL, ORD4, PAT, PATM, QHV, ROMAN, VAL)


def q_atom(e, b, deg=None, nb=()):
    """(a) q_i = v + b − quota (octet assumption `lp = 4 − b`).

    ★ (a′) covers only the sites where the octet breaks (2026-08-30, [design doc] §5.0.8 ④ · V2d).
    `deg` = number of ligand-*internal* neighbors · `nb` = tuple of those neighbors' elements.
    🔴 **The neighbor-element condition is essential** — keying on `(element, deg, b)` alone
    causes regressions (measured):
      · forcing `("N",2,4)` to −1 also catches the **central N of azide N₃** (neighbors N,N) and
        the **isocyanide N** (neighbors C,C), throwing the ligand charge off by −2.
      · forcing `("C",2,2)` to 0 gets **CF₂** (neighbors F,F) and `C7H6` (neighbors C,C) wrong.
    """
    # 🔴 `QHV=1` — **hypervalent generalization** (2026-09-03). `lp = 4 − b` becomes
    #   `lp = max(0, 4 − b)` ⇒ `q = v − b − 2·lp`. For `b ≤ 4` this is **identical** to
    #   `v + b − 8` (exactly the current behavior); only for `b > 4` does it become `q = v − b`.
    #   It absorbs the hand-written exceptions `S(=O)₂` (b 6) and `P=O` (b 5), and fixes nitro
    #   **`–N(=O)=O` (b 5)** from `+2` to `0` (the `ACARAZ` mechanism).
    #   measured (reference assignment · train 26,075 · 94,117 fragments): 1,427 hypervalent
    #   atoms · EHT target mismatch for their fragments **50.7% → 28.5%** (626 → 352). Not a
    #   single `b ≤ 4` site changes.
    if QHV and b > 4.0 + 1e-9:
        return VAL.get(e, 4) - b
    nO, nN = nb.count("O"), nb.count("N")
    if e == "C" and deg == 2 and b == 2.0 and nN + nO >= 1:
        # heteroatom-stabilized carbene — 6 electrons. The octet formula gives −2 (`nO` added
        #   after the owner's remark, 2026-09-02).
        #   NHC `:C(NR)₂` (N neighbors) was caught from the start, but the **Fischer carbene
        #   `:C(OR)R` (O neighbor)** was missed and kept falling to −2 — **374 cases** measured
        #   (all M-coordinated · `C–O` neighbor).
        #   ⚠️ Schrock alkylidenes (`C–H` 465 · `C–C` 397 · `H–H` 111) have `nN+nO = 0` and so
        #      **stay at −2 — which is correct.** The heteroatom condition separates the two.
        #   ⚠️ `S` was not included — none of the 95 `C–S` cases is M-coordinated, so they are
        #      not carbene donors.
        return 0
    if e == "S" and deg == 3 and b == 4.0 and nO >= 1:
        return 0  # sulfoxide S=O — 10 electrons.              octet formula +2
    if e == "S" and deg == 4 and b == 6.0 and nO >= 2:
        return 0  # sulfone / sulfonate center — 12 electrons.  octet formula +4
    if e == "P" and deg == 4 and b == 5.0 and nO + nN >= 1:
        return 0  # phosphine oxide P=O · phosphinimide P=N — 10 electrons.  octet formula +2
    if e == "N" and deg == 2 and b == 4.0 and nO == 2:
        return -1  # nitro (two N=O) — 10 electrons.            octet formula +1
    return VAL.get(e, 4) + b - FULL.get(e, 8)

def frag_charge(el, atoms, edges, orders, deg=None, nbrs=None, out=None):
    """Charge of one conjugated fragment — rule (b).

    monocyclic all-carbon **`CmHm`** → Hückel `z = m − (4n+2)`
                                       ← every ring atom must have exactly 1 external bond
    otherwise                        → maximize Kekule (maximum matching), then sum (a)

    🔴 2 bugs fixed (2026-08-30, [design doc] §5.0.8 ②):
      B1  Hückel was also applied to substituted all-carbon rings, making **phenyl C6H5 come out
          0** (truth −1). Adding the `CmHm` check drops it to Kekule and gives −1.
      B2  `min(HUCKEL, key=|m−h|)` picked **the earlier 6 on the m=8 tie between 6 and 10**,
          making **COT +2** (truth −2). On a tie, take **the larger**.
    """
    ring = len(edges) == len(atoms) and all(
        sum(1 for a, b in edges if a == v or b == v) == 2 for v in atoms
    )
    huckel = None
    if ring and all(el[a] == "C" for a in atoms):
        m = len(atoms)
        if all(orders.get(v, 0.0) == 1.0 for v in atoms):  # ← B1: CmHm check
            d = min(abs(m - h) for h in HUCKEL)
            huckel = m - max(h for h in HUCKEL if abs(m - h) == d)  # ← B2: on a tie, the larger
            if out is None:
                return huckel
            # ⑥ output converter — even on the Hückel branch, **the skeleton comes from the
            #   matching** (2026-09-02). Hückel fixes only the charge; the S/D/T skeleton comes
            #   from the matching below.
            #   ⚠️ For an even-ring dianion (η⁴-C₄R₄²⁻, η⁸-COT²⁻) the skeleton is a **neutral
            #      Kekule**, so per-atom charges cannot be inferred from it — the charge has to
            #      be reported at the fragment level.
    G = nx.Graph()
    G.add_nodes_from(atoms)
    G.add_edges_from(edges)
    match = nx.max_weight_matching(G, maxcardinality=True)
    md = {}
    for a, b in match:
        md[(min(a, b), max(a, b))] = 2.0
    if out is not None:
        for a, b in edges:
            out[(min(a, b), max(a, b))] = md.get((min(a, b), max(a, b)), 1.0)
    if huckel is not None:
        return huckel
    q = 0
    for v in atoms:
        b = sum(md.get((min(v, w), max(v, w)), 1.0) for w in G[v])
        b += orders.get(v, 0.0)  # order of bonds leaving the fragment
        q += q_atom(el[v], b, None if deg is None else deg.get(v), (nbrs or {}).get(v, ()))
    return q

def _qfrag(G, el, cls, comp):
    """Charge of one fragment — the same q rule as in the main text (conjugated fragments go
    through `frag_charge`)."""
    pc = {e for e, v in cls.items() if v == 3 and e[0] in comp}
    Gc = nx.Graph()
    Gc.add_edges_from(pc)
    ca = set(Gc.nodes)
    DEG = {v: G.degree(v) for v in comp}
    NB = {v: tuple(sorted(el[w] for w in G[v])) for v in comp}
    q = 0.0
    for cm in nx.connected_components(Gc) if pc else []:
        sub = [(a, b) for a, b in Gc.subgraph(cm).edges]
        outer = {
            v: sum(
                0.0 if (min(v, w), max(v, w)) in pc else ORD4[cls.get((min(v, w), max(v, w)), 0)]
                for w in G[v]
                if w not in cm
            )
            for v in cm
        }
        q += frag_charge(el, list(cm), sub, outer, DEG, NB)
    for v in comp:
        if v in ca:
            continue
        q += q_atom(
            el[v], sum(ORD4[cls.get((min(v, w), max(v, w)), 0)] for w in G[v]), DEG[v], NB[v]
        )
    return q

def kekulize(G, el, cls, bml=None):
    """⑥ **output converter** — turn the 4-class prediction (`Conj` included) back into integer
    S/D/T (2026-09-02).

    Returns `(orders, frag_q)`
      `orders` {(i,j): 1|2|3}                    — integer bond orders for output (all internal
                                                   bonds)
      `frag_q` {fragment representative (min idx): charge}
                                                 — holds **charge that cannot be inferred from
                                                   the skeleton**

    **Why a separate return is needed.** In an odd π system (Cp⁻, allyl⁻) the matching leaves one
    atom over and that site simply becomes −1, so the charge is readable off the skeleton. But an
    **even-ring dianion** (η⁴-C₄R₄²⁻ · η⁸-COT²⁻) has a perfect matching, so **the skeleton is a
    neutral Kekule** while the real charge is −2. That −2 is **two electrons**, not a bond
    pattern, and no S/D/T assignment can express it.
    ⇒ For such a fragment, **do not stamp it per atom; report it as the ligand charge** (the
      Hückel branch supplies that charge).

    ⚠️ It uses **the same matching** as the charge/valence path (`_qfrag`) — running a separate
       matching would let the output and the charge diverge (this is why the distance-likelihood
       weighted matching in `relabel.py` must not be reused here).
    """
    bml = bml or {}
    orders = {}
    for e, v in cls.items():
        if v != 3:
            orders[(min(e), max(e))] = int(ORD4[v])
    pc = {e for e, v in cls.items() if v == 3}
    Gc = nx.Graph()
    Gc.add_edges_from(pc)
    DEG = {v: G.degree(v) for v in G.nodes}
    NB = {v: tuple(sorted(el[w] for w in G[v])) for v in G.nodes}
    frag_q = {}
    for cm in nx.connected_components(Gc) if pc else []:
        sub = [(a, b) for a, b in Gc.subgraph(cm).edges]
        outer = {
            v: sum(
                0.0 if (min(v, w), max(v, w)) in pc else ORD4[cls.get((min(v, w), max(v, w)), 0)]
                for w in G[v]
                if w not in cm
            )
            for v in cm
        }
        out = {}
        q = frag_charge(el, list(cm), sub, outer, DEG, NB, out=out)
        for e, o in out.items():
            orders[e] = int(o)
        # if the skeleton charge and the fragment charge disagree (even-ring dianion), report it
        # as the ligand charge
        q_skel = 0
        for v in cm:
            b = sum(out.get((min(v, w), max(v, w)), 1.0) for w in Gc[v]) + outer.get(v, 0.0)
            q_skel += q_atom(el[v], b, DEG.get(v), NB.get(v, ()))
        if round(q_skel) != round(q):
            frag_q[min(cm)] = q
    return orders, frag_q

def parse_os(m, nm):
    """Read the oxidation state of metal `m` from the name. Returns None for mixed valence or
    unknown (excluded from scoring)."""
    nm = (nm or "").lower()
    stems = [NAMEEL.get(m, m.lower())] + ALT.get(m, [])
    ok = lambda x: any(x.startswith(y) or y in x for y in stems)  # noqa: E731
    if [g for g in PATM.findall(nm) if ok(g[0])]:
        return None  # mixed valence — excluded from scoring
    f = {ROMAN[r] for x, r in PAT.findall(nm) if ok(x)}
    return f.pop() if len(f) == 1 else None


# ★★ cluster fragment charge (2026-09-03) — **a fragment that cannot be written in 2-center
#   form** uses the EHT value.
#   rule  F is a cluster ⟺ F contains an atom with `b_int(x) > CAP(el[x])`
#   Why: a carborane cage follows Wade's rules (multicenter skeletal bonding) and is not
#       expressible as 2-center 2-electron. A cage `B` has 5-6 internal neighbors, so
#       `b_int > CAP(B)=4`, and the formal-charge formula (`q = v + b − 8` · hypervalent
#       `q = v − b`) piles up −2 to −3 per atom.
#       measured (`GANLUF` · 2026-09-03): formal-charge sum of the carborane ligand **−27** vs
#       EHT **−1**.
#   ⚠️ With no EHT value it falls back to the formal-charge sum (better than being silently
#      wrong).
#   ⚠️ The body of this function must **stay identical to** workspace
#      `260830_fit_t10_charge.py`.
def is_cluster_frag(G, el, cls, comp):
    """Is the fragment a cluster (multicenter skeleton)? — the rule above."""
    for x in comp:
        b = sum(ORD4[cls.get((min(x, w), max(x, w)), 0)] for w in G[x])
        if b > CAP.get(el[x], 4) + 1e-9:
            return True
    return False


def frag_charge_or_eht(G, el, cls, comp, q_eht=None):
    """Fragment charge — the **EHT fragment charge** for a cluster, otherwise the formal-charge
    sum (`_qfrag`)."""
    if is_cluster_frag(G, el, cls, comp):
        q = (q_eht or {}).get(min(comp))
        if q is not None:
            return float(q)
    return _qfrag(G, el, cls, comp)

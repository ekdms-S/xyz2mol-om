"""Formal charge — per atom `q_atom` · conjugated fragment `frag_charge` · fragment sum
`_qfrag` · output converter `kekulize`.

"""

# ruff: noqa: E501
from __future__ import annotations


import networkx as nx

from ..config import (ALT, CAP, FULL, HUCKEL, NAMEEL, ORD4, PAT, PATM, QHV, ROMAN, VAL)


def q_atom(e, b, deg=None, nb=()):
    """(a) q_i = v + b − quota (octet assumption `lp = 4 − b`).

    ★ (a′) covers only the sites where the octet breaks (, (`docs/PIPELINE.md`).
    `deg` = number of ligand-*internal* neighbors · `nb` = tuple of those neighbors' elements.
    🔴 **The neighbor-element condition is essential** — keying on `(element, deg, b)` alone
    causes regressions (measured):
      · forcing `("N",2,4)` to −1 also catches the **central N of azide N₃** (neighbors N,N) and
        the **isocyanide N** (neighbors C,C), throwing the ligand charge off by −2.
      · forcing `("C",2,2)` to 0 gets **CF₂** (neighbors F,F) and `C7H6` (neighbors C,C) wrong.
    """
    # 🔴 `QHV=1` — **hypervalent generalization**. `lp = 4 − b` becomes
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
        #   after the owner's remark).
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

def frag_charge(el, atoms, edges, orders, deg=None, nbrs=None, out=None, w=None):
    """Charge of one conjugated fragment — rule (b).

    monocyclic all-carbon **`CmHm`** → Hückel `z = m − (4n+2)`
                                       ← every ring atom must have exactly 1 external bond
    otherwise                        → maximize Kekule (maximum matching), then sum (a)

    🔴 2 bugs fixed (, (`docs/PIPELINE.md`):
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
            #   matching**. Hückel fixes only the charge; the S/D/T skeleton comes
            #   from the matching below.
            #   ⚠️ For an even-ring dianion (η⁴-C₄R₄²⁻, η⁸-COT²⁻) the skeleton is a **neutral
            #      Kekule**, so per-atom charges cannot be inferred from it — the charge has to
            #      be reported at the fragment level.
    G = nx.Graph()
    G.add_nodes_from(atoms)
    if w:
        # Same cardinality, but among those prefer the assignment the bond lengths prefer
        #   (`CONJW`: `w[e]` is `score[Double] − score[Single]` from the ③ likelihood) and
        #   never pair an atom the ④ budget promised to leave unmatched (`CAPINESS`: `−1e6`).
        G.add_edges_from((a, b, {"weight": w.get((min(a, b), max(a, b)), 0.0)}) for a, b in edges)
    else:
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

def atom_bond_sums(G, el, cls, comp, w=None):
    """Per-atom **internal** bond-order sum for one fragment, resolved through the *same* Kekule
    matching the output converter uses.

    Returns `(bsum, deg, nbrs)` so a caller can get the formal charge of atom `v` as
    `q_atom(el[v], bsum[v], deg[v], nbrs[v])` — exactly what `_qfrag` sums up, but kept per atom.

    ⚠️ M–L bonds are **not** counted (the formal charge of a ligand atom is written from its
       internal bonds only — `bml` is dative and stays out, as in `_qfrag`).
    ⚠️ The matching runs on the `Conj` subgraph alone, so the order of a **non-`Conj`** bond does
       not change it. That is what makes the `ADJQW` candidate scan cheap: a ±1 move on a
       non-`Conj` bond shifts `bsum` at its two endpoints and nowhere else.
    """
    pc = {e for e, v in cls.items() if v == 3 and e[0] in comp}
    Gc = nx.Graph()
    Gc.add_edges_from(pc)
    DEG = {v: G.degree(v) for v in comp}
    NB = {v: tuple(sorted(el[w] for w in G[v])) for v in comp}
    md = {}
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
        frag_charge(el, list(cm), sub, outer, DEG, NB, out=out, w=w)
        md.update(out)
    bs = {}
    for v in comp:
        bs[v] = sum(
            md.get((min(v, w), max(v, w)), ORD4[cls.get((min(v, w), max(v, w)), 0)]) for w in G[v]
        )
    return bs, DEG, NB

def _qfrag(G, el, cls, comp, w=None):
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
        q += frag_charge(el, list(cm), sub, outer, DEG, NB, w=w)
    for v in comp:
        if v in ca:
            continue
        q += q_atom(
            el[v], sum(ORD4[cls.get((min(v, w), max(v, w)), 0)] for w in G[v]), DEG[v], NB[v]
        )
    return q

def kekulize(G, el, cls, bml=None, w=None):
    """⑥ **output converter** — turn the 4-class prediction (`Conj` included) back into integer
    S/D/T.

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
        q = frag_charge(el, list(cm), sub, outer, DEG, NB, out=out, w=w)
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


def octet_fix_period2(el, G, orders):
    """A period-2 atom cannot exceed an octet, so `N` never carries five bonds (`NOCTET`).

    The reference (and therefore our ⑥ output) writes a nitro group as `-N(=O)=O`, which puts a
    bond-order sum of 5 on the nitrogen. Nitrogen has no d orbitals: the only Lewis structure that
    respects the octet is the charge-separated `-N+(=O)O-`. RDKit refuses the neutral form and
    rewrites it on output, which left our `bonds_kekule` and our SMILES on two different
    conventions -- and a consumer that took bonds from one and charges from the other lost an
    electron pair.

    So one `N=O` to a **terminal** O is demoted to `N-O`. Nothing else is touched: the charge then
    follows from the ordinary octet rule (`N` +1, that `O` -1), the fragment total is unchanged,
    and period-3 atoms (`S` in a sulfone, `Cl` in a perchlorate) keep the hypervalent form, which
    is legitimate for them.

    Mutates `orders` in place.
    """
    for v in G:
        if el[v] != "N":
            continue
        for _ in range(3):
            b = sum(orders.get((min(v, w), max(v, w)), 1) for w in G[v])
            if b <= 4:
                break
            cand = [w for w in G[v]
                    if el[w] == "O" and G.degree(w) == 1
                    and orders.get((min(v, w), max(v, w)), 1) == 2]
            if not cand:
                break
            w = min(cand)
            orders[(min(v, w), max(v, w))] = 1

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


# ★★ cluster fragment charge — **a fragment that cannot be written in 2-center
#   form** uses the EHT value.
#   rule  F is a cluster ⟺ F contains an atom with `b_int(x) > CAP(el[x])`
#   Why: a carborane cage follows Wade's rules (multicenter skeletal bonding) and is not
#       expressible as 2-center 2-electron. A cage `B` has 5-6 internal neighbors, so
#       `b_int > CAP(B)=4`, and the formal-charge formula (`q = v + b − 8` · hypervalent
#       `q = v − b`) piles up −2 to −3 per atom.
#       measured (`GANLUF` ·): formal-charge sum of the carborane ligand **−27** vs
#       EHT **−1**.
#   ⚠️ With no EHT value it falls back to the formal-charge sum (better than being silently
#      wrong).
#   ⚠️ The body of this function must **stay identical to** workspace
#      `260830_fit_t10_charge.py`.
def is_cluster_frag(G, el, cls, comp, orders=None):
    """Is the fragment a cluster (multicenter skeleton)? — the rule above.

    With `orders` (the Kekule integers from `kekulize`) the sum uses those instead of the
    4-class values, so a `Conj` bond counts 1 or 2 rather than 1.5. That is not optional —
    counting 1.5 made a carbon with three `Conj` bonds read 4.5 > 4 and took ordinary arenes
    for cages; see the `is_cluster_frag` note in `config`.
    """
    for x in comp:
        if orders is not None:
            b = sum(orders.get((min(x, w), max(x, w)), 1) for w in G[x])
        else:
            b = sum(ORD4[cls.get((min(x, w), max(x, w)), 0)] for w in G[x])
        if b > CAP.get(el[x], 4) + 1e-9:
            return True
    return False


def _qfrag_kek(G, el, comp, orders, frag_q=None):
    """Fragment charge counted on the **emitted Kekule integers**.

    Why this and not `_qfrag`: `_qfrag` sums `ORD4[cls]`, and a `Conj` bond counts **1.5** there,
    so an atom with three of them reads `b = 3.5` and picks up a formal charge of `-0.5` that no
    emitted bond accounts for. The reported ligand charge then disagrees with the structure the
    user receives -- measured on `MBTZRE01` (benzothiazole-2-thiolate): reported **-3**, emitted
    structure **-1**, and -1 is the correct chemistry.

    `frag_q` carries the part of the charge the skeleton genuinely cannot express (an even-ring
    dianion has a perfect matching, so its Kekule structure is neutral while the fragment is -2).
    """
    q = sum(v for k, v in (frag_q or {}).items() if k in comp)
    for v in comp:
        b = sum(orders.get((min(v, w), max(v, w)), 1.0) for w in G[v])
        q += q_atom(el[v], float(b), G.degree(v), tuple(sorted(el[w] for w in G[v])))
    return q


def frag_charge_or_eht(G, el, cls, comp, q_eht=None, orders=None, w=None, frag_q=None):
    """Fragment charge — the **EHT fragment charge** for a cluster, otherwise the formal-charge
    sum."""
    if is_cluster_frag(G, el, cls, comp, orders):
        q = (q_eht or {}).get(min(comp))
        if q is not None:
            return float(q)
    if orders is not None:
        return _qfrag_kek(G, el, comp, orders, frag_q)
    return _qfrag(G, el, cls, comp, w)  # only when the caller has no Kekule structure yet


def pi_suppressed(bonds_kekule, qat, w):
    """Bonds ⑥ wrote `Single` between **two anionic atoms** where the ③ distance likelihood
    preferred `Double`. Returns the sorted bond list, empty when there is none.

        suspect(i,j) ⟺ bonds_kekule[(i,j)] == 1 AND qat[i] < 0 AND qat[j] < 0 AND w[(i,j)] > 0

    `w = score[Double] − score[Single]` is the margin ⑥ already uses as its tie-break, so this
    adds **no threshold**. It must be the *raw* margin: ④'s `−10⁶` promise is a matching
    constraint, not a likelihood, and it lands on exactly these edges (`rules.pipeline`
    `w_raw_out`).

    Such a bond is a **charge** error, not only an order error — the π bond becomes a lone pair on
    each end, so the fragment charge comes out **2 too negative** and, on a metal-bearing molecule,
    the metal's oxidation state **2 too high**.

    ⚠️ **A flag, not a correction** — the orders and charges are returned unchanged. What it
    catches, with the numbers, is in the README under `## ⚠️ Limits`.
    """
    return sorted(e for e, o in bonds_kekule.items()
                  if o == 1 and qat.get(e[0], 0) < 0 and qat.get(e[1], 0) < 0
                  and w.get(e, 0.0) > 0.0)

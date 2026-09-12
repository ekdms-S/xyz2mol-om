"""Formal charge — per atom `q_atom` · conjugated fragment `frag_charge` · fragment sum
`_qfrag` · output converter `kekulize`.

"""

# ruff: noqa: E501
from __future__ import annotations


import collections

import networkx as nx

from ..config import (ALT, CAP, FULL, HUCKEL, KEKQ, KEKQMODE, NAMEEL, ORD4, PAT, PATM, QHV,
                      QGEM, QCHFIT, QSHIFT, ROMAN, SIGCUT, SIGCUTFIT, SIGCUTW,
                      VAL)


def q_atom(e, b, deg=None, nb=(), n_ml=0):
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
    if e == "B":
        # ★ **boron fills a sextet, not an octet.** The default `v + b - 8` comes from the
        #   octet assumption `lp = 4 - b`; boron carries no lone pair until it reaches four
        #   bonds, so its quota is 6 and `lp = max(0, 3 - b)`:
        #       b 4 -> -1 (BF4-, the N->B adduct `[N+]=[B-]`)  -- same as the octet formula
        #       b 3 ->  0 (B(OH)3, a boronic ester, B2pin2)    -- the octet formula said **-2**
        #       b 2 -> -1 (a boryl ligand `[BR2]-` after the ionic cut)
        #   🔴 The -2 does not stay local: it is a real charge on the fragment, so a
        #   metal-bearing molecule pays for it with **+2 on the oxidation state**.
        #   Measured on Gold-DIGR (1,200 reactions): **all 55** trivalent borons came out -2, in
        #   35 reactions (2.9%). `10.1039_D2CY01506D__46_TS8b` is one -- a Suzuki intermediate
        #   whose B(OH)2(OAr) read `[B-2]`, fragment -3, and put Pd at **+4** instead of +2.
        #   This is `QHV`'s `lp = max(0, quota/2 - b)` statement with the quota 6 instead of 8.
        return VAL["B"] - b - 2 * max(0.0, 3.0 - b)
    nO, nN = nb.count("O"), nb.count("N")
    if e == "C" and deg == 2 and b == 2.0 and nN + nO == 0 and n_ml == 0:
        # ★ **free carbene** — a divalent carbon with no heteroatom neighbour *and* no bond to a
        #   metal. The octet formula calls it −2, and that −2 is paired by a spurious +2 on the
        #   metal: measured on `joc.6b02957__14_14`, C40 (neighbours C·C, nearest H 1.53 Å, no
        #   Ni bond) came out `[C-2]` and put Ni at +2, while the other IRC endpoint — where the
        #   H has arrived and the carbon is a plain `C=C` — put the same Ni at 0.
        #   🔴 The `n_ml == 0` half is what keeps the Schrock case correct. An **M-coordinated**
        #   alkylidene really is −2 under the ionic cut (`C–H` 465 · `C–C` 397 · `H–H` 111 in the
        #   reference), and it is separated from this one by the metal bond, not by the
        #   neighbours. A **free** carbene is a neutral 6-electron singlet.
        #   🔴 **`deg == 1` 로 넓히는 것은 측정 후 기각했다 (2026-09-12).** 치환기 없는 말단
        #     바이닐리덴 `:C=CR₂` 도 자유 카벤과 같은 중성 6전자 종이니 같이 넣자는 안이었는데,
        #     holdout 에서 11 구조가 바뀌어 **`OS` 맞→틀 2 · 틀→맞 0** (.8971 → .8964) 이었다.
        #     Gold-DIGR 에서도 대상이 15 프레임(0.07%)뿐이다. ⇒ `deg == 2` 를 유지한다.
        return 0
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
        ew = {(min(a, b), max(a, b)): w.get((min(a, b), max(a, b)), 0.0) for a, b in edges}
        if KEKQ > 0:
            # 🔴 Quantize **within an element pair**, then break what is left canonically by atom
            #   index (see the `KEKQ` comment). Scoping to the element pair is the whole point:
            #   measured on CSD, the two Kekule alternants of an aromatic carbocycle differ by
            #   **0.008 Å** (`Conj→Single` median 1.394 vs `Conj→Double` 1.386), i.e. distance
            #   carries no signal about which C–C of a ring is the double one — but it carries a
            #   great deal about `C=O` 1.234 Å against `C–C` 1.440 Å (`EMAXAR`). Quantizing each
            #   element pair around **its own median in this fragment** flattens the first and
            #   leaves the second untouched; a single global grid large enough to flatten a ring
            #   also erases the cross-pair preference (measured: `KEKQ=16` global cost `Σq_L`
            #   −0.0034).
            # 🔴 The tie-break key must not depend on **which** edges are in the set. Ranking
            #   inside `ew` did, and that is what made the canonicalisation counter-productive
            #   exactly where the `Conj` set differs between two frames of one reaction path
            #   (the "`Conj` boundary" atoms): the set changes, every rank shifts, and the two
            #   frames get different placements from a rule meant to make them identical.
            #   Measured before this: `KEKQ` created 27 such disagreements while fixing 21.
            #   The atom indices are the same in both frames, so key on them alone.
            _sc = KEKQ / (10.0 * max(len(atoms), 1))
            eps = 1.0

            def order(e, _s=_sc):
                return _s * ((e[0] * 1048576 + e[1]) / 1099511627776.0)
            if KEKQMODE == "abs":
                # absolute grid — the bin cannot move when the geometry does
                ew = {e: round(v / KEKQ) * KEKQ - eps * order(e) for e, v in ew.items()}
            else:
                grp = collections.defaultdict(list)
                for e in ew:
                    grp[tuple(sorted((el[e[0]], el[e[1]])))].append(e)
                out2 = {}
                for _k, es in grp.items():
                    base = sorted(ew[e] for e in es)[len(es) // 2]
                    for e in es:
                        out2[e] = base + round((ew[e] - base) / KEKQ) * KEKQ - eps * order(e)
                ew = out2
        G.add_edges_from((a, b, {"weight": ew[(min(a, b), max(a, b))]}) for a, b in edges)
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


def _qfrag_kek(G, el, comp, orders, frag_q=None, coord=None):
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
        q += q_atom(el[v], float(b), G.degree(v), tuple(sorted(el[w] for w in G[v])),
                    n_ml=(1 if coord and v in coord else 0))
    return q


def frag_charge_or_eht(G, el, cls, comp, q_eht=None, orders=None, w=None, frag_q=None,
                      coord=None):
    """Fragment charge — the **EHT fragment charge** for a cluster, otherwise the formal-charge
    sum."""
    if is_cluster_frag(G, el, cls, comp, orders):
        q = (q_eht or {}).get(min(comp))
        if q is not None:
            return float(q)
    if orders is not None:
        return _qfrag_kek(G, el, comp, orders, frag_q, coord)
    return _qfrag(G, el, cls, comp, w)  # only when the caller has no Kekule structure yet


def shift_pi_to_cancel(orders, el, G, bml, coord=(), cap=None, kmax=5, fit=None):
    """⑥ 이후 후처리 — **결합차수를 경로를 따라 재분배해서 형식전하를 상쇄한다.**

    같은 결합 집합 위에 |전하| 가 더 작은 **유효한** 배치가 존재하는데 솔버가 그것을 고르지 못한
    경우만 건드린다. 모호함이 아니라 해결 가능한 실패다.

    ## 규칙

    전하를 가진 두 원자 `a`, `c` 와 그 사이 경로(결합 `k` 개)에 대해 차수를 **±1 씩 번갈아**
    바꾼다. 번갈아 바꾸면 경로 **안쪽** 원자의 차수 합은 그대로이고 양 끝만 움직인다.

        δ_i = δ_a · (−1)^i        ⇒  안쪽은 δ_i + δ_{i+1} = 0

    `q = v + b − 8` 이므로 **b 를 늘리면 q 가 오른다** — 음전하는 늘리고 양전하는 줄인다.
    `k` 가 홀수면 양 끝이 같은 방향으로, 짝수면 반대 방향으로 움직인다. 그래서

      · `k` 홀수 → **같은 부호** 전하 쌍   `C⁻–C=C–O⁻`  →  `C=C–C=O`   (오너 지목, Au 케이스)
      · `k` 짝수 → **반대 부호** 전하 쌍   `O⁺≡C–O⁻`    →  `O=C=O`     (CO₂)

    ## 🔴 `k ≥ 2` 여야 한다

    `k = 1` 은 **양쪽성 이온을 그대로 두라는 관례가 맞는** 자리다 — 일산화탄소 `[C⁻]≡[O⁺]` ·
    아민 옥사이드 `R₃N⁺–O⁻` · 인 일리드 `R₃P⁺–C⁻`. 여기서 차수를 옮기면 CO 가 `C=O` 가 되고
    금속 카보닐 전체가 무너진다. 두 원자 사이 결합이 하나뿐이면 손대지 않는다.

    ## 적용 조건 (전부 만족해야 한다)

      · 경로 결합 수 `2 ≤ k ≤ kmax`
      · 번갈아 바꾼 뒤 모든 차수가 `1..3` 안에 있다
      · 양 끝이 **원자가 상한**을 넘지 않는다 (M–L 예산 `bml` 포함)
      · **|전하 합| 이 실제로 줄어든다** — 특수 규칙(카벤·설폭사이드·나이트로)까지 반영해
        다시 계산해서 비교한다
      · 양 끝 중 **금속에 배위하는 것이 하나 이하**다 🔴

    🔴 마지막 조건이 결정적이다. 두 음이온 자리가 **둘 다** 금속에 배위하면 그것은 잘못 놓인 π 가
    아니라 **진짜 이음이온 킬레이트**다 — 다이싸이올렌 `[S⁻]C(R)=C(R)[S⁻]` · 벤젠-1,2-다이싸이
    올레이트 · 카테콜레이트 · 아미디네이트가 전부 이 꼴이고, CSD 규약은 이들을 이음이온으로
    적는다. 게이트 없이 돌리면 다이싸이올렌이 중성 다이싸이온 `S=C–C=S` 가 되고 금속이 2 내려간다.
    실측 (holdout · 게이트 전): 145 구조가 바뀌어 `Σq_L` 맞→틀 **15** · 틀→맞 4, `Σq_L`
    .8553 → .8458 · `OS` .8899 → .8802.

    ⚠️ `pi_suppressed` 와 다르다. 저쪽은 **인접한** 두 음이온 사이의 `Single` 을 보고하되
    고치지 않는다 — 캡이 막고 있어서 고칠 수가 없기 때문이다. 이쪽은 캡이 허용하는 배치가
    이미 존재하므로 **고칠 수 있고**, 고치지 않으면 금속 산화수가 2 씩 틀린다.
    """
    if not QSHIFT or not orders:
        return []
    cap = CAP if cap is None else cap
    bml = bml or {}
    g = nx.Graph()
    g.add_edges_from(orders)

    def charges():
        b = collections.Counter(); deg = collections.Counter()
        for (i, j), o in orders.items():
            b[i] += o; b[j] += o; deg[i] += 1; deg[j] += 1
        q = {}
        for a in g:
            nb = tuple(el[y] for y in g[a])
            q[a] = q_atom(el[a], float(b[a]), deg[a], nb)
        return b, q

    moved = []
    for _ in range(4):
        b, q = charges()
        tot = sum(abs(v) for a, v in q.items() if el[a] != "H")
        ch = [a for a in g if q[a] and el[a] != "H"]
        hit = None
        for n, a in enumerate(ch):
            for c in ch[n + 1:]:
                try:
                    _pth = nx.shortest_path(g, a, c)
                except nx.NetworkXNoPath:
                    continue
                _k = len(_pth) - 1
                _e1 = (min(a, c), max(a, c))
                if (a in coord) and (c in coord) and not (
                        QCHFIT and _k == 1 and fit is not None and _e1 in orders
                        and fit(a, c, orders[_e1], orders[_e1] + 1)):
                    # ★ **예외 (2026-09-12)**: 주개-주개 결합이 **짧으면** 킬레이트가 아니다.
                    #   진짜 이음이온 킬레이트(다이싸이올렌 · 카테콜레이트 · 아미디네이트)는 두
                    #   주개를 잇는 골격 결합이 **단일결합 길이**다. 반면 눌린 π 리간드는 짧다 —
                    #   `[H][N⁻][N⁻][H]` 의 N–N 이 **1.293 Å**(다이아젠 HN=NH 가 1.25 · 하이드라지도
                    #   단일이 1.45)이고, `[H][N⁻][O⁻]` 의 N–O 가 **1.336 Å** 이다. 배위 여부로
                    #   가르면 이 둘이 이음이온으로 남아 금속이 2 씩 부풀어 오른다
                    #   (오너 지적 2026-09-11 · `A_dOS2_ligand_only__01`·`__04`).
                    #   판정은 `scores4` 의 클래스별 길이 중앙값 비교라 **새 문턱이 없다.**
                    # 이음이온 킬레이트 — 잘못 놓인 π 가 아니다.
                    # ⚠️ "한쪽이 |q| ≥ 2 면 게이트를 풀자" 를 측정했다 (진짜 킬레이트는 각
                    #   자리가 −1 이므로 −3 은 솔버가 흘린 전하라는 논리). **채택 안 함** —
                    #   holdout `OS` .8938 → .8935 로 떨어지고, 그 논리를 만든 케이스
                    #   (`10.1021_acs.inorgchem.3c02611__09_Int3` 의 `[C-3]`) 는 어차피 안
                    #   고쳐진다. 유일한 경로가 이미 원자가 상한에 닿은 원자를 지나기 때문이다.
                    continue
                path, k = _pth, _k
                if k > kmax:
                    continue
                if k == 1 and q[a] * q[c] < 0:
                    # 🔴 **반대 부호**의 k = 1 은 건드리지 않는다 — 양쪽성 이온을 그대로 두라는
                    #   관례가 맞는 자리다: 일산화탄소 `[C⁻]≡[O⁺]` · 아민 옥사이드 `R₃N⁺–O⁻` ·
                    #   인 일리드 `R₃P⁺–C⁻`. 여기서 차수를 옮기면 금속 카보닐이 전부 무너진다.
                    continue
                if k == 1:
                    # ★ **같은 부호**의 k = 1 은 다르다. 결합 하나를 올리면 **양 끝이 동시에**
                    #   중성이 된다 — `[C⁻](H)(H)[O⁻]` 는 폼알데하이드 `H₂C=O` 이고,
                    #   `[C⁻]([O⁻])(H)CH₃` 는 아세트알데하이드다. 둘 다 Gold-DIGR 의 IRC 끝점에서
                    #   **떨어져 나온 자유 분자**로 이렇게 나왔다. 상한과 |전하| 감소는 아래에서
                    #   그대로 검사하므로, 여는 것은 이 부호 조건 하나뿐이다.
                    #   ⚠️ **과산화물은 예외.** `[O⁻]–[O⁻]` 를 올리면 `O=O` 가 되는데, 과산화
                    #   이음이온은 실재하는 화학종이고 금속 착물의 흔한 리간드다. 같은 원소끼리의
                    #   음이온 쌍 중 O–O 만 막는다 — `[C⁻]–[C⁻]`(에틸렌 이음이온 → 에텐) 는
                    #   올리는 것이 맞다.
                    if el[a] == "O" and el[c] == "O":
                        continue
                    d = [1 if q[a] < 0 else -1]
                    e = [(min(a, c), max(a, c))]
                    if not 1 <= orders[e[0]] + d[0] <= 3:
                        continue
                    if any(b[x] + d[0] + bml.get(x, 0.0) > cap.get(el[x], 4) + 1e-9
                           for x in (a, c) if d[0] > 0):
                        continue
                    orders[e[0]] += d[0]
                    if sum(abs(v) for x, v in charges()[1].items() if el[x] != "H") < tot:
                        hit = e
                        break
                    orders[e[0]] -= d[0]
                    continue
                if QGEM and k == 2 and q[a] * q[c] > 0:
                    # ★ **제미널 (2026-09-12)** — 공통 이웃 **하나**를 사이에 둔 같은 부호 쌍.
                    #   교대(±1)로는 원리상 못 고친다: 한쪽을 올리면 다른 쪽은 내려간다. 필요한
                    #   수는 **두 결합을 같이** 올리는 것이고, 그러면 가운데 원자가 2 를 더 쓴다.
                    #   🔴 `[O⁻]–C–[O⁻]` 가 그 꼴이다 — CO₂ 를 카벤 탄소로 쓴 것이고, 금속에
                    #   붙으면 산화수가 2 부풀어 오른다 (오너 지적 · `A_dOS2_ligand_only__02`,
                    #   Cu +3). 두 결합을 올리면 `O=C=O` 중성이 된다.
                    #   ⚠️ 길이가 **두 결합 다** 올린 차수에 맞을 때만 움직인다. 탄산염처럼 가운데
                    #      원자가 이미 꽉 찬 경우는 아래 상한 검사가 막는다.
                    _mid = path[1]
                    _e2 = [(min(x, y), max(x, y)) for x, y in zip(path, path[1:])]
                    _d0 = 1 if q[a] < 0 else -1
                    if any(not 1 <= orders[k2] + _d0 <= 3 for k2 in _e2):
                        continue
                    if fit is None or any(
                            not fit(k2[0], k2[1], orders[k2], orders[k2] + _d0) for k2 in _e2):
                        continue
                    if _d0 > 0 and (
                            b[a] + 1 + bml.get(a, 0.0) > cap.get(el[a], 4) + 1e-9
                            or b[c] + 1 + bml.get(c, 0.0) > cap.get(el[c], 4) + 1e-9
                            or b[_mid] + 2 + bml.get(_mid, 0.0) > cap.get(el[_mid], 4) + 1e-9):
                        continue
                    for k2 in _e2:
                        orders[k2] += _d0
                    if sum(abs(v) for x, v in charges()[1].items() if el[x] != "H") < tot:
                        hit = _e2
                        break
                    for k2 in _e2:
                        orders[k2] -= _d0
                    continue
                da = -1 if q[a] > 0 else 1
                dc = -1 if q[c] > 0 else 1
                if da * (-1) ** (k - 1) != dc:
                    continue        # 번갈아 바꿔서는 이 부호 조합을 못 맞춘다
                e = [(min(x, y), max(x, y)) for x, y in zip(path, path[1:])]
                d = [da * (-1) ** t for t in range(k)]
                if any(not 1 <= orders[k2] + d[t] <= 3 for t, k2 in enumerate(e)):
                    continue
                if any(b[x] + dx + bml.get(x, 0.0) > cap.get(el[x], 4) + 1e-9
                       for x, dx in ((a, da), (c, dc)) if dx > 0):
                    continue
                for t, k2 in enumerate(e):
                    orders[k2] += d[t]
                if sum(abs(v) for x, v in charges()[1].items() if el[x] != "H") < tot:
                    hit = e
                    break
                for t, k2 in enumerate(e):      # 되돌린다 — 실제로 줄지 않았다
                    orders[k2] -= d[t]
            if hit:
                break
        if not hit:
            break
        moved.append(tuple(hit))
    return moved


def abs_charge_sum(orders, el, G):
    """비수소 원자의 `|형식전하|` 합 — `SIGCUT` 재풀이를 채택할지 가르는 값."""
    b = collections.Counter()
    deg = collections.Counter()
    for (i, j), o in orders.items():
        b[i] += o; b[j] += o; deg[i] += 1; deg[j] += 1
    return sum(abs(q_atom(el[a], float(b[a]), deg[a], tuple(el[y] for y in G[a])))
               for a in b if el[a] != "H")


def sigma_ml_blocking_cancel(orders, el, G, bml, ml_pred, hap=(), cap=None, wbo=None,
                             fit=None):
    """끊어야 할 σ M–L 을 돌려준다 — 실제로 끊고 다시 푸는 것은 `api.predict` 다.

    `shift_pi_to_cancel` 의 `k = 1` 같은 부호 분기는 결합을 하나 올려 양 끝을 동시에 중성으로
    만든다. 그 분기의 상한 검사에는 **M–L 예산 `bml` 이 들어 있어서**, σ M–L 하나가 자리를
    차지하고 있으면 올릴 수가 없다. 그런데 그 σ 야말로 전하를 만든 원인이다.

    조건과 근거는 `config.SIGCUT` 에 있다. 여기서 하는 것은 **후보를 고르는 것뿐**이고,
    채택 여부는 다시 푼 뒤 `abs_charge_sum` 이 실제로 줄었는지로 정한다.
    """
    if not SIGCUT or not orders:
        return set()
    cap = CAP if cap is None else cap
    bml = bml or {}
    hapset = {(min(a, b), max(a, b)) for a, b in hap}
    b = collections.Counter()
    deg = collections.Counter()
    for (i, j), o in orders.items():
        b[i] += o; b[j] += o; deg[i] += 1; deg[j] += 1
    q = {a: q_atom(el[a], float(b[a]), deg[a], tuple(el[y] for y in G[a])) for a in b}
    coord = collections.defaultdict(set)
    sig = collections.defaultdict(set)
    for m, x in ml_pred:
        coord[x].add(m)
        # 🔴 `B`·`Al` 은 **끊지 않는다.** 둘은 조건부 중심이라 M–L 후보로 올라오지만, 카보란·
        #   보릴의 `B–C`(1.54~1.62 Å · Mayer 0.85~1.23)와 `Al–C`(2.07 Å · 0.65)는 케이지·골격의
        #   평범한 공유결합이다. 전하 결함의 수리가 그것을 끊는 것일 수는 없다 — 실측에서
        #   잘려나간 참양성의 절반 가까이가 이 둘이었다 (holdout 24 건 중 10 건).
        if el[m] in ("B", "Al"):
            continue
        if (min(m, x), max(m, x)) not in hapset:
            sig[x].add(m)
    out = set()
    for e, o in orders.items():
        a, c = e
        if el[a] == "H" or el[c] == "H" or o + 1 > 3:
            continue
        if not (q.get(a, 0) < 0 and q.get(c, 0) < 0):
            continue
        if el[a] == "O" and el[c] == "O":
            continue                      # 과산화물 — `QSHIFT` 와 같은 예외
        blocked = [x for x in (a, c)
                   if b[x] + 1 + bml.get(x, 0.0) > cap.get(el[x], 4) + 1e-9]
        if not blocked:
            continue                      # 여유가 있다 — `QSHIFT` 가 처리할 자리
        cut, ok = set(), True
        for x in blocked:
            other = c if x == a else a
            ms = sorted(sig.get(x, ()))
            if not ms or ms[0] in coord.get(other, ()):
                ok = False                # σ 가 없거나, 짝도 같은 금속에 배위한다
                break
            _w = (wbo or {}).get((ms[0], x), (wbo or {}).get((x, ms[0])))
            if _w is None:
                ok = False                # Mayer 가 없으면 끊지 않는다
                break
            if _w >= SIGCUTW and not (SIGCUTFIT and fit is not None and fit(a, c, o, o + 1)):
                # ★ **예외 (2026-09-12)**: Mayer 가 상한 위여도, 차수를 올린 쪽이 **결합 길이에
                #   더 맞으면** 끊는다. 상한은 «실재하는 M–L 을 지우지 마라» 는 뜻인데, 리간드
                #   자신의 길이는 M–L 이 얼마나 센지와 **무관한 사실**이다 — `Double` 로 적힌
                #   C–C 가 1.209 Å 이면 그것은 눌린 알카인이지 이중결합이 아니다
                #   (오너 지적 2026-09-11 · `A_dOS2_ligand_only__06` Pd–C Mayer 0.622 ·
                #   `__03` Ag–C 0.699 — 둘 다 상한 0.40 에 막혀 있었다).
                ok = False
                break
            if b[x] + bml.get(x, 0.0) > cap.get(el[x], 4) + 1e-9:
                ok = False                # 하나 끊어도 모자란다
                break
            cut.add((ms[0], x))
        if ok:
            out |= cut

    # ★ **제미널 (2026-09-12)** — 공통 이웃을 사이에 둔 음이온 쌍. `shift_pi_to_cancel` 의
    #   같은 이름 분기가 두 결합을 같이 올리려 하는데, 가운데 원자가 2 를 더 쓰므로 σ M–L 이
    #   있으면 막힌다. `[O⁻]–C(→M)–[O⁻]` 가 그 꼴이다 (CO₂ · `A_dOS2_ligand_only__02`).
    adj = collections.defaultdict(set)
    for x, y in orders:
        adj[x].add(y)
        adj[y].add(x)
    for mid in (list(adj) if QGEM else ()):
        if q.get(mid, 0) < 0:
            continue
        an = sorted(y for y in adj[mid] if q.get(y, 0) < 0 and el[y] != "H")
        for i2 in range(len(an)):
            for j2 in range(i2 + 1, len(an)):
                a, c = an[i2], an[j2]
                if coord.get(a, set()) & coord.get(c, set()):
                    continue              # 두 음이온이 같은 금속에 배위 — 진짜 이음이온
                es = [(min(mid, a), max(mid, a)), (min(mid, c), max(mid, c))]
                if any(orders[e2] + 1 > 3 for e2 in es):
                    continue
                if fit is None or any(
                        not fit(e2[0], e2[1], orders[e2], orders[e2] + 1) for e2 in es):
                    continue
                need, okg = set(), True
                for x, extra in ((a, 1), (c, 1), (mid, 2)):
                    if b[x] + extra + bml.get(x, 0.0) <= cap.get(el[x], 4) + 1e-9:
                        continue
                    ms = sorted(sig.get(x, ()))
                    if not ms:
                        okg = False
                        break
                    _w = (wbo or {}).get((ms[0], x), (wbo or {}).get((x, ms[0])))
                    if _w is None or _w >= SIGCUTW:
                        okg = False
                        break
                    if b[x] + extra + bml.get(x, 0.0) - 1.0 > cap.get(el[x], 4) + 1e-9:
                        okg = False
                        break
                    need.add((ms[0], x))
                if okg and need:
                    out |= need
    return out


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

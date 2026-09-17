"""Formal charge — per atom `q_atom` · conjugated fragment `frag_charge` · fragment sum
`_qfrag` · output converter `kekulize` · post-⑥ repairs (`docs/PIPELINE.md` §Charge).

"""

# ruff: noqa: E501
from __future__ import annotations


import collections

import networkx as nx

from ..config import (ALT, CAP, FULL, HUCKEL, KEKQ, KEKQMODE, MLIKE_EXTRA, NAMEEL, ORD4, PAT, PATM, QHV,
                      QGEM, QCHFIT, QSHIFT, ROMAN, SIGCUT, SIGCUTFIT, SIGCUTW,
                      VAL)


def q_atom(e, b, deg=None, nb=(), n_ml=0, b_3c=0.0):
    """(a) q_i = v + b − quota (octet assumption `lp = 4 − b`).

    (a′) special cases cover the sites where the octet breaks (`docs/PIPELINE.md` §Charge).
    `deg` = number of ligand-*internal* neighbors · `nb` = tuple of those neighbors' elements ·
    `n_ml` = number of M–L bonds on the atom (only the free-carbene case reads it).

    `b_3c` — the bond-order sum of this atom's internal 3c2e bonds that hold no pair of their own
    (`three_c_unpaired_edges`). Such a bond holds **one pair between three centres**, so it comes
    off `b`, and boron's sextet lone-pair term is switched off when any is present.

        `B₂H₆`   bridging H  b 2 → 0  ⇒ **−1**   ·   B  b 4 → 2, no lp  ⇒ **+1**

    ⚠️ A bridge with no internal leg to a B/Al (μ-H, μ-CO, μ-CH₃) has `b_3c = 0`.
    ⚠️ The special cases key on neighbor elements as well as `(element, deg, b)`: without them the
       nitrite case would also catch the central N of azide and the isocyanide N.
    """
    # `QHV` — hypervalent branch: `lp = max(0, 4 − b)` ⇒ `q = v − b − 2·lp`. For `b ≤ 4` this is
    #   identical to `v + b − 8`; above `b = 4` it gives `q = v − b` (sulfone S, `P=O`, `–N(=O)=O`
    #   read 0).
    # an internal 3c2e bond carries no 2c-2e pair of its own — take it off `b` first, so every
    #   branch below sees the two-centre bond count the atom really has.
    if b_3c:
        b = b - b_3c
    if QHV and b > 4.0 + 1e-9:
        return VAL.get(e, 4) - b
    if e == "B":
        # boron fills a sextet, not an octet: its quota is 6 and `lp = max(0, 3 - b)`.
        #       b 4 -> -1 (BF4-, the N->B adduct `[N+]=[B-]`)  -- same as the octet formula
        #       b 3 ->  0 (B(OH)3, a boronic ester, B2pin2)
        #       b 2 -> -1 (a boryl ligand `[BR2]-` after the ionic cut)
        #   ⚠️ with `b_3c` the lone-pair term is 0: a B in a `B–H–B` bridge has put its third
        #   electron into that bridge, so `b 2` there is `+1`, not a boryl's `-1`.
        lp = 0.0 if b_3c else 2 * max(0.0, 3.0 - b)
        return VAL["B"] - b - lp
    nO, nN = nb.count("O"), nb.count("N")
    if e == "C" and deg == 2 and b == 2.0 and nN + nO == 0 and n_ml == 0:
        # free carbene — a divalent carbon with no N/O neighbor and no M–L bond: a neutral
        #   6-electron singlet (octet formula −2).
        #   ⚠️ An M-coordinated alkylidene (Schrock) stays −2 under the ionic cut; `n_ml == 0` is
        #   what separates the two.
        return 0
    if e == "C" and deg == 2 and b == 2.0 and nN + nO >= 1:
        # heteroatom-stabilized carbene — NHC `:C(NR)₂`, Fischer `:C(OR)R` — 6 electrons. The
        #   octet formula gives −2.
        #   ⚠️ Schrock alkylidenes have `nN+nO = 0` and so stay at −2.
        return 0
    if e == "S" and deg == 3 and b == 4.0 and nO >= 1:
        return 0  # sulfoxide S=O — 10 electrons.              octet formula +2
    if e == "S" and deg == 4 and b == 6.0 and nO >= 2:
        return 0  # sulfone / sulfonate center — 12 electrons.  octet formula +4
    if e == "P" and deg == 4 and b == 5.0 and nO + nN >= 1:
        return 0  # phosphine oxide P=O · phosphinimide P=N — 10 electrons.  octet formula +2
    if e == "N" and deg == 2 and b == 4.0 and nO == 2:
        return -1  # nitrite O=N=O (two N=O) — 10 electrons.   octet formula +1
    return VAL.get(e, 4) + b - FULL.get(e, 8)

def three_c_legs(el, G, btag):
    """`{X: [(X, y), ...]}` — the **ligand-internal legs** of each 3c2e bridge, by bridging atom.

    T7 writes `n_center(X) = n_ML(X) + |{internal neighbours in MLIKE_EXTRA}|`
    (`docs/PIPELINE.md` 5″), and those two terms *are* the legs of the three-centre bond. So a
    leg is either an M–L bond — already reported in `ml_bonds[...]["bridge"]` — or an internal
    bond to a `B`/`Al` neighbour, which is what this returns.

        μ-H       M–H–M      legs: 2 M–L, 0 internal              → {}
        μ-CO      M–CO–M     legs: 2 M–L. **`C≡O` is not a leg**  → {}
        μ-CH₃     M–CH₃–M    legs: 2 M–L. `C–H` are not legs      → {}
        κ²-BH₄    B–H···M    legs: 1 M–L + **the `B–H`**          → {H: [(B,H)]}
        B–H–B     diborane   legs: 2 internal, no metal at all    → {H: [(B,H), (B,H)]}

    Whether a leg carries the pair is decided in `three_c_unpaired_edges`, not here.
    """
    out = {}
    for x, t in (btag or {}).items():
        if t != "3c2e" or x not in G or el[x] in MLIKE_EXTRA:
            continue        # a B/Al does not bridge to a B/Al — same condition as `bridge_tags`
        legs = [(min(x, y), max(x, y)) for y in G[x] if el[y] in MLIKE_EXTRA]
        if legs:
            out[x] = legs
    return out


def three_c_unpaired_edges(el, G, btag):
    """The internal legs that carry **no electron pair of their own**.

        the bridging atom holds the pair  ⟺  el[X] == "H"  OR  X has two internal legs

    A three-centre bond holds one pair and no M–L leg ever carries it, so the only question is
    whether the bridging atom can keep its legs as separate two-centre bonds.

    · **Hydrogen never can.** One orbital and one electron cannot make two σ bonds, so an H in a
      bridge has its pair spread over all three centres — the `B–H` of a κ²-`BH₄` exactly as
      much as the `B–H–B` of a borane. Both give `[H-]` against a `B(+1)`: **a bridging hydrogen
      reads the same wherever it sits.**
    · **Carbon can.** A carbon that bridges two metals and also has a boron partner keeps that
      `C–B` as an ordinary bond.
    · Two internal legs is the case where no single bond can hold the pair at all.

    `μ-CH₃`, `μ-CO` and `μ-H` have no internal leg, so none of them is touched here.

    ⚠️ A ligand whose bridging H holds the pair is **not expressible in two-centre form** — the
       H has two bonds and the boron four against a `+1` — so its `smiles_ok` is False.
    """
    return {e for x, legs in three_c_legs(el, G, btag).items()
            if el[x] == "H" or len(legs) >= 2 for e in legs}


def b_3c_of(G, orders, three_c):
    """`{atom: bond-order sum of its all-internal 3c2e bonds}` — the `b_3c` argument of
    `q_atom`, built from the same `orders` the caller charges the atom with."""
    out = collections.defaultdict(float)
    for i, j in three_c or ():
        o = float((orders or {}).get((i, j), 1.0))
        out[i] += o
        out[j] += o
    return out


def frag_charge(el, atoms, edges, orders, deg=None, nbrs=None, out=None, w=None):
    """Charge of one conjugated fragment — rule (b).

    monocyclic all-carbon **`CmHm`** → Hückel `z = m − h`, `h` ∈ `HUCKEL` nearest `m`
                                       (the larger on a tie)
                                       ← each ring atom's bonds out of the ring sum to exactly 1
    otherwise                        → maximize Kekule (maximum matching), then sum (a)

    `orders` {atom: order sum of its bonds leaving the fragment}. With `out`, the matched orders
    (1 or 2) of `edges` are written into it; `w` weights the matching (see below).
    """
    ring = len(edges) == len(atoms) and all(
        sum(1 for a, b in edges if a == v or b == v) == 2 for v in atoms
    )
    huckel = None
    if ring and all(el[a] == "C" for a in atoms):
        m = len(atoms)
        if all(orders.get(v, 0.0) == 1.0 for v in atoms):  # CmHm check
            d = min(abs(m - h) for h in HUCKEL)
            huckel = m - max(h for h in HUCKEL if abs(m - h) == d)  # on a tie, the larger
            if out is None:
                return huckel
            # ⑥ output converter — Hückel fixes only the charge; the S/D/T skeleton comes from
            #   the matching below.
            #   ⚠️ For an even-ring dianion (η⁴-C₄R₄²⁻, η⁸-COT²⁻) the skeleton is a **neutral
            #      Kekule**, so per-atom charges cannot be inferred from it — the charge has to
            #      be reported at the fragment level.
    G = nx.Graph()
    G.add_nodes_from(atoms)
    if w:
        # Same cardinality, but among those prefer the assignment the bond lengths prefer
        #   (`w[e]` is `score[Double] − score[Single]` from the ③ likelihood) and never pair an
        #   atom the ④ budget promised to leave unmatched (`−1e6`).
        ew = {(min(a, b), max(a, b)): w.get((min(a, b), max(a, b)), 0.0) for a, b in edges}
        if KEKQ > 0:
            # Quantize **within an element pair** (`KEKQMODE=pair`, around that pair's median in
            #   this fragment), then break what is left by atom index (`config.KEKQ`). A per-pair
            #   grid flattens the negligible length difference between the two Kekule
            #   alternants of a ring but keeps cross-pair preferences such as `C=O` vs `C–C`.
            # The tie-break key depends on atom indices alone, not on which edges are in the
            #   set, so two frames whose `Conj` sets differ still get the same placement.
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
       not change it: a ±1 move on a non-`Conj` bond shifts `bsum` at its two endpoints and
       nowhere else (`rules.pipeline._adjq_pairs` relies on this).
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
    """Charge of one fragment from the 4-class orders — rule (a) per atom, with conjugated
    sub-fragments going through `frag_charge`."""
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
      `frag_q` {fragment representative (min idx): q − q_skel}
                                                 — the part of a conjugated fragment's charge
                                                   the skeleton does not carry (only where the
                                                   two differ)

    **Why a separate return is needed.** The Kekule skeleton does not always carry the Hückel
    charge. An **even-ring dianion** (η⁴-C₄R₄²⁻ · η⁸-COT²⁻) has a perfect matching, so **the
    skeleton is a neutral Kekule** while the real charge is −2 — two electrons, not a bond
    pattern, and no S/D/T assignment can express it. An odd cation (C₇H₇⁺) leaves one atom
    unmatched, which reads −1 against a Hückel +1.
    ⇒ For such a fragment, **do not stamp it per atom; report it as the ligand charge**.

    ⚠️ It uses **the same matching** as the charge/valence path (`_qfrag`) — a separate matching
       would let the output and the charge diverge.
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
        # store only the part the skeleton cannot express, q - q_skel, so `_qfrag_kek` (which
        # adds frag_q to a sum that already contains q_skel) does not double-count it
        q_skel = 0
        for v in cm:
            b = sum(out.get((min(v, w), max(v, w)), 1.0) for w in Gc[v]) + outer.get(v, 0.0)
            q_skel += q_atom(el[v], b, DEG.get(v), NB.get(v, ()))
        if round(q_skel) != round(q):
            frag_q[min(cm)] = round(q) - round(q_skel)
    return orders, frag_q


def octet_fix_period2(el, G, orders):
    """A period-2 atom cannot exceed an octet, so `N` never carries five bonds (`NOCTET`).

    ⑥ follows the reference in writing a nitro group as `-N(=O)=O`, which puts a bond-order sum of
    5 on the nitrogen. Nitrogen has no d orbitals: the only Lewis structure that respects the
    octet is the charge-separated `-N+(=O)O-`, which is also the form RDKit writes, so this keeps
    `bonds_kekule` and the SMILES on one convention.

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
    when the name gives no single state."""
    nm = (nm or "").lower()
    stems = [NAMEEL.get(m, m.lower())] + ALT.get(m, [])
    ok = lambda x: any(x.startswith(y) or y in x for y in stems)  # noqa: E731
    if [g for g in PATM.findall(nm) if ok(g[0])]:
        return None  # mixed valence
    f = {ROMAN[r] for x, r in PAT.findall(nm) if ok(x)}
    return f.pop() if len(f) == 1 else None


# Cluster fragment charge — **a fragment that cannot be written in 2-center form** uses the EHT
#   value.
#   rule  F is a cluster ⟺ F contains an atom with `b_int(x) > CAP(el[x])`
#   A carborane cage follows Wade's rules (multicenter skeletal bonding): a cage `B` has 5-6
#       internal neighbors, so `b_int > CAP(B)=4`, and the formal-charge formula piles up −2 to
#       −3 per atom.
#   ⚠️ With no EHT value it falls back to the formal-charge sum (`frag_charge_or_eht`).
def is_cluster_frag(G, el, cls, comp, orders=None):
    """Is the fragment a cluster (multicenter skeleton)? — the rule above.

    With `orders` (the Kekule integers from `kekulize`) the sum uses those instead of the
    4-class values, so a `Conj` bond counts 1 or 2 rather than 1.5.
    ⚠️ Pass `orders` when available: at 1.5 per `Conj` bond a carbon with three of them reads
       4.5 > 4 and an ordinary arene looks like a cage.
    """
    for x in comp:
        if orders is not None:
            b = sum(orders.get((min(x, w), max(x, w)), 1) for w in G[x])
        else:
            b = sum(ORD4[cls.get((min(x, w), max(x, w)), 0)] for w in G[x])
        if b > CAP.get(el[x], 4) + 1e-9:
            return True
    return False


def _qfrag_kek(G, el, comp, orders, frag_q=None, coord=None, three_c=None):
    """Fragment charge counted on the **emitted Kekule integers**.

    Unlike `_qfrag`, where a `Conj` bond counts **1.5** and an atom can pick up a fractional
    charge that no emitted bond accounts for, this keeps the reported ligand charge consistent
    with the structure the user receives.

    `frag_q` carries only the part of the charge the skeleton cannot express on its own (q minus
    the skeleton's own charge), since this sum already counts the skeleton's share.
    `coord` = atoms with an M–L bond · `three_c` = the edges from `three_c_unpaired_edges`.
    """
    q = sum(v for k, v in (frag_q or {}).items() if k in comp)
    b3 = b_3c_of(G, orders, three_c)
    for v in comp:
        b = sum(orders.get((min(v, w), max(v, w)), 1.0) for w in G[v])
        q += q_atom(el[v], float(b), G.degree(v), tuple(sorted(el[w] for w in G[v])),
                    n_ml=(1 if coord and v in coord else 0), b_3c=b3.get(v, 0.0))
    return q


def frag_charge_or_eht(G, el, cls, comp, q_eht=None, orders=None, w=None, frag_q=None,
                      coord=None, three_c=None):
    """Fragment charge — the **EHT fragment charge** for a cluster, otherwise the formal-charge
    sum."""
    if is_cluster_frag(G, el, cls, comp, orders):
        q = (q_eht or {}).get(min(comp))
        if q is not None:
            return float(q)
    if orders is not None:
        return _qfrag_kek(G, el, comp, orders, frag_q, coord, three_c)
    return _qfrag(G, el, cls, comp, w)  # only when the caller has no Kekule structure yet


def shift_pi_to_cancel(orders, el, G, bml, coord=(), cap=None, kmax=5, fit=None):
    """Post-⑥ repair (`QSHIFT`, `QGEM`) — **redistribute bond orders along a path to cancel
    formal charges.**

    Acts only where a **valid** arrangement with smaller |charge| exists on the same bonds.
    Mutates `orders`; returns the moved bonds, one tuple per move (at most four moves).

    ## Rule

    For two charged atoms `a`, `c` and the shortest path between them (`k` bonds), change the
    orders **by ±1 alternately**. Alternation keeps the bond-order sum of every **inner** atom;
    only the two ends move.

        δ_i = δ_a · (−1)^i        ⇒  inner atoms: δ_i + δ_{i+1} = 0

    `q = v + b − 8`, so **raising b raises q** — an anion's bond goes up, a cation's goes down.
    For odd `k` both ends move the same way, for even `k` in opposite ways:

      · `k` odd  → **like-signed** pair       `C⁻–C=C–O⁻`  →  `C=C–C=O`
      · `k` even → **opposite-signed** pair   `O⁺≡C–O⁻`    →  `O=C=O`     (CO₂)

    `k = 1`: an opposite-signed pair keeps its zwitterion (CO `[C⁻]≡[O⁺]`, amine oxide
    `R₃N⁺–O⁻`, phosphorus ylide `R₃P⁺–C⁻`); a like-signed pair is raised, except peroxide.
    `k = 2` like-signed (`QGEM`): both bonds shift together, `[O⁻]–C–[O⁻]` → `O=C=O`.

    ## Conditions (all must hold)

      · `k ≤ kmax`
      · every order stays within `1..3`
      · a raised end stays within its **valence cap** (M–L budget `bml` included)
      · **the summed non-H |charge| actually falls** — recomputed with `q_atom`, special cases
        (carbene · sulfoxide · nitrite) included
      · **at most one end coordinates a metal** 🔴

    🔴 Two anionic sites that **both** coordinate a metal are a **genuine dianionic chelate**
    (dithiolene `[S⁻]C(R)=C(R)[S⁻]`, benzene-1,2-dithiolate, catecholate, amidinate), not a
    misplaced π, and CSD convention writes them as dianions.

    ⚠️ Not the same as `pi_suppressed`, which reports a `Single` between **adjacent** anions that
    the cap blocks. Here the cap already allows the fix.
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
                    # a dianionic chelate — not a misplaced π.
                    # ⚠️ `QCHFIT`: a donor–donor bond whose length fits the raised order better is
                    #   a suppressed π ligand (`[H][N⁻][N⁻][H]` → diazene), not a chelate.
                    continue
                path, k = _pth, _k
                if k > kmax:
                    continue
                if k == 1 and q[a] * q[c] < 0:
                    # an **opposite-signed** k = 1 pair keeps its zwitterion: CO `[C⁻]≡[O⁺]` ·
                    #   amine oxide `R₃N⁺–O⁻` · phosphorus ylide `R₃P⁺–C⁻`. Moving the order here
                    #   would break every metal carbonyl.
                    continue
                if k == 1:
                    # a **like-signed** k = 1 pair: raising the bond neutralizes both ends at once
                    #   (`[C⁻](H)(H)[O⁻]` → formaldehyde `H₂C=O`). The cap and the |charge| drop are
                    #   checked below.
                    #   ⚠️ **Peroxide is excluded.** The peroxide dianion `[O⁻]–[O⁻]` is a real
                    #   species and a common ligand; `[C⁻]–[C⁻]` and other pairs are raised.
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
                    # **geminal** — a like-signed pair on **one** common neighbor. Alternation
                    #   cannot fix it (raising one end lowers the other); both bonds shift
                    #   together, and the middle atom takes ±2 (`[O⁻]–C–[O⁻]` → `O=C=O`).
                    #   ⚠️ Only when **both** bond lengths fit the new orders. A full middle atom
                    #      (carbonate) is stopped by the cap check below.
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
                    continue        # alternation cannot reach this sign combination
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
                for t, k2 in enumerate(e):      # revert — the |charge| sum did not fall
                    orders[k2] -= d[t]
            if hit:
                break
        if not hit:
            break
        moved.append(tuple(hit))
    return moved


def abs_charge_sum(orders, el, G):
    """Sum of `|formal charge|` over non-hydrogen atoms — decides whether a `SIGCUT` re-solve is
    kept."""
    b = collections.Counter()
    deg = collections.Counter()
    for (i, j), o in orders.items():
        b[i] += o; b[j] += o; deg[i] += 1; deg[j] += 1
    return sum(abs(q_atom(el[a], float(b[a]), deg[a], tuple(el[y] for y in G[a])))
               for a in b if el[a] != "H")


def sigma_ml_blocking_cancel(orders, el, G, bml, ml_pred, hap=(), cap=None, wbo=None,
                             fit=None, eta_out=None):
    """Return the σ M–L bonds `{(m, x)}` to cut — the cut and re-solve are done by `api.predict`.

    The like-signed `k = 1` branch of `shift_pi_to_cancel` raises one bond to neutralize both ends.
    Its cap check **includes the M–L budget `bml`**, so a σ M–L occupying the last valence unit
    blocks the raise — and that σ bond is what created the charge. The geminal (`QGEM`) case is
    handled the same way.

    Conditions are listed at `config.SIGCUT`. This **only picks candidates**; the caller keeps a
    re-solve only if `abs_charge_sum` actually falls. With `eta_out`, a σ M–L too strong to cut
    and its partner atom are collected there as η² candidates (`SIGETA`).
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
        # `Al` is **never cut**: it is a centre, so it appears as an M–L candidate, but `Al–C` is
        #   an ordinary covalent skeletal bond. (`B` is not a centre and does not reach here;
        #   listing it is harmless.)
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
            continue                      # peroxide — same exception as `QSHIFT`
        blocked = [x for x in (a, c)
                   if b[x] + 1 + bml.get(x, 0.0) > cap.get(el[x], 4) + 1e-9]
        if not blocked:
            continue                      # headroom left — `QSHIFT` handles it
        cut, ok = set(), True
        for x in blocked:
            other = c if x == a else a
            ms = sorted(sig.get(x, ()))
            if not ms or ms[0] in coord.get(other, ()):
                ok = False                # no σ M–L, or the partner coordinates the same metal
                break
            _w = (wbo or {}).get((ms[0], x), (wbo or {}).get((x, ms[0])))
            if _w is None:
                ok = False                # no Mayer value — never cut
                break
            if _w >= SIGCUTW and not (SIGCUTFIT and fit is not None and fit(a, c, o, o + 1)):
                # too strong to cut, unless `SIGCUTFIT` finds the raised order fits the ligand
                #   bond length better. `SIGETA`: hand the bond and the partner atom `other` over
                #   as **η² candidates** instead — made haptic, they free the budget for the raise
                #   while the M–L bond stays.
                if eta_out is not None:
                    eta_out.add((ms[0], x))
                    eta_out.add((ms[0], other))
                ok = False
                break
            if b[x] + bml.get(x, 0.0) > cap.get(el[x], 4) + 1e-9:
                ok = False                # one cut does not free enough
                break
            cut.add((ms[0], x))
        if ok:
            out |= cut

    # **geminal** (`QGEM`) — an anion pair on one common neighbor. `shift_pi_to_cancel` raises
    #   both bonds, so the middle atom needs 2 more and a σ M–L on any of the three atoms can
    #   block it (`[O⁻]–C(→M)–[O⁻]`, bent CO₂).
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
                    continue              # both anions coordinate the same metal — a genuine dianion
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

    ⚠️ **A flag, not a correction** — the orders and charges are returned unchanged. See the
    README, `## ⚠️ Limits`.
    """
    return sorted(e for e, o in bonds_kekule.items()
                  if o == 1 and qat.get(e[0], 0) < 0 and qat.get(e[1], 0) < 0
                  and w.get(e, 0.0) > 0.0)

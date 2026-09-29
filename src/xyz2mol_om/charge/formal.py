"""Formal charge — the per-atom rule `q_atom`, the 3c2e legs that carry no pair of their own, and
the pi-suppression flag (`docs/PIPELINE.md` §Charge).
"""

# ruff: noqa: E501
from __future__ import annotations

import collections

from ..config import FULL, MLIKE_EXTRA, QHV, VAL


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

    T7 writes `n_center(X) = n_ML(X) + |{internal neighbours in MLIKE_EXTRA}|` (for H · C · Si only
    when their σ bonds fill them — `rules.pipeline.bridge_tags`) (`docs/PIPELINE.md` 5″), and those
    two terms *are* the legs of the three-centre bond. So a
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

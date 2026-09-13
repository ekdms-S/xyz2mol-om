"""🔴 Regression — boron is a **ligand atom everywhere**, and an all-internal 3c2e bridge is
priced as one pair between three centres.

`B` used to be a *conditional* centre: a centre when the structure held no transition metal, a
ligand atom when it did. The same `B–H` bond was therefore a dative arrow in `B₂H₆` and an
ordinary covalent bond inside a carborane — the output shape for boron moved with its context.
The evidence behind that conditional turned out to be a selection artifact of the CSD extraction
(ccdc does not count B as a metal, so B is the only metal-class element that survives the
heterometallic filter to be *seen* as a ligand atom); `config.centers` carries the argument.

What the tests below pin:

  · `centers()` never returns a boron, transition metal present or not
  · `B₂H₆` comes out `B(+1)` · bridging `H(−1)` · terminal `H(0)`, sum 0 — the signs matter,
    since dropping the `b_3c` correction gives the same total with `B(−1)`/`H(+1)`
  · the spurious `B–B` that `d_int(B,B) = 2.336 Å` draws across the two bridges is removed
  · a 3c2e bridge reached **through a metal** (`μ-H`, `B–H···M`, `μ-CO`) is left alone
"""

# ruff: noqa: E501
from __future__ import annotations

import math

import networkx as nx
import numpy as np
import pytest

from xyz2mol_om import predict
from xyz2mol_om.charge import q_atom, three_c_legs, three_c_unpaired_edges
from xyz2mol_om.config import METALS, centers


def _b2h6():
    """D2h diborane — B–B 1.774 Å · bridging B–H 1.330 Å · terminal B–H 1.190 Å."""
    zB = 0.887
    xb = math.sqrt(1.33**2 - zB**2)
    a = math.radians(61.0)
    el = ["B", "B", "H", "H", "H", "H", "H", "H"]
    xyz = np.array([
        [0.0, 0.0, zB],                                              # 0  B
        [0.0, 0.0, -zB],                                             # 1  B
        [xb, 0.0, 0.0],                                              # 2  H bridge
        [-xb, 0.0, 0.0],                                             # 3  H bridge
        [0.0, 1.19 * math.sin(a), zB + 1.19 * math.cos(a)],          # 4  H terminal on B0
        [0.0, -1.19 * math.sin(a), zB + 1.19 * math.cos(a)],         # 5
        [0.0, 1.19 * math.sin(a), -zB - 1.19 * math.cos(a)],         # 6  H terminal on B1
        [0.0, -1.19 * math.sin(a), -zB - 1.19 * math.cos(a)],        # 7
    ])
    return el, xyz


def test_boron_is_not_in_metals():
    assert "B" not in METALS


def test_boron_is_never_a_centre():
    assert centers(["B", "B", "H", "H"]) == set()          # no metal anywhere — used to be {0,1}
    assert centers(["Pd", "B", "H"]) == {0}                # unchanged: only the Pd


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_diborane_has_no_metal_and_one_fragment():
    el, xyz = _b2h6()
    r = predict(el, xyz, total_charge=0)
    (mol,) = r["molecules"]
    assert mol["metals"] == []                             # used to be two B(+3) centres
    (fr,) = mol["fragments"]
    assert fr["atoms"] == list(range(8))


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_diborane_drops_the_b_b_bond_the_bridges_already_pay_for():
    # `R₂B(μ-H)₂BR₂` has 2·3 + 2·1 + 4·1 = 12 valence electrons; 4 terminal B–H (8 e) plus the
    # two 3c2e bridges (4 e) already spend all 12, so a B–B would need 14. `d_int(B,B)` is
    # 2.336 Å and cannot tell 1.774 Å from a real carborane cage bond, hence the explicit rule.
    el, xyz = _b2h6()
    fr = predict(el, xyz, total_charge=0)["molecules"][0]["fragments"][0]
    assert (0, 1) not in fr["bonds_kekule"]
    assert sorted(fr["bonds_kekule"]) == [(0, 2), (0, 3), (0, 4), (0, 5),
                                          (1, 2), (1, 3), (1, 6), (1, 7)]


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_diborane_charges_are_b_plus_and_bridging_hydride():
    el, xyz = _b2h6()
    r = predict(el, xyz, total_charge=0)
    fr = r["molecules"][0]["fragments"][0]
    assert fr["bonds_3c2e"] == [(0, 2), (0, 3), (1, 2), (1, 3)]
    assert fr["charge"] == 0
    # 🔴 the signs are the point — without `b_3c` the sum is also 0, with B(−1)/H(+1)
    assert fr["smiles"] == "[H][B+]1([H])[H-][B+]([H])([H])[H-]1"
    # a bridging H cannot be written in two-centre form, so the round trip is refused on purpose
    assert not fr["smiles_ok"]
    assert "3c2e" in fr["smiles_note"]


def test_b_3c_flips_the_sign_of_the_bridge():
    # bridging H — both legs are 3c2e, so `b` drops to 0 and the atom is a hydride
    assert q_atom("H", 2.0, b_3c=2.0) == -1
    assert q_atom("H", 2.0) == +1                          # what it reads without the correction
    # B in that bridge — 2 terminal bonds left and **no** sextet lone pair, so +1 not −1
    assert q_atom("B", 4.0, b_3c=2.0) == +1
    assert q_atom("B", 2.0) == -1                          # a boryl, untouched: it does have one


def test_b_3c_leaves_every_other_boron_alone():
    assert q_atom("B", 3.0) == 0                           # B(OH)₃ · a boronic ester · B₂pin₂
    assert q_atom("B", 4.0) == -1                          # BF₄⁻ · the N→B adduct
    assert q_atom("B", 2.0) == -1                          # a boryl after the ionic cut


def test_a_leg_is_a_bond_to_a_metal_like_neighbour_not_any_bond():
    """`three_c_legs` reports every internal leg; `three_c_unpaired_edges` only the empty ones."""
    # κ²-BH₄ on a metal: the ligand graph is B + 4 H, two of the H also bound to the metal.
    # The `B–H` **is** a leg and is reported — but it holds the pair, so it is not subtracted.
    el = {1: "B", 2: "H", 3: "H", 4: "H", 5: "H"}
    G = nx.Graph()
    G.add_nodes_from([1, 2, 3, 4, 5])
    G.add_edges_from([(1, 2), (1, 3), (1, 4), (1, 5)])
    #    Hydrogen cannot hold two σ bonds, so its one leg carries no pair either — the ligand is
    #    `B(+1)` with two bridging `[H-]`, the same motif diborane has.
    tags = {2: "3c2e", 3: "3c2e"}
    assert three_c_legs(el, G, tags) == {2: [(1, 2)], 3: [(1, 3)]}
    assert three_c_unpaired_edges(el, G, tags) == {(1, 2), (1, 3)}

    # 🔴 a **boron** bridging atom can: a diboranyl `R₂B–BR₂` on a metal has an ordinary `B–B`
    #    and a separate M–B σ. Subtracting that leg read `[B+2]` and put the metal at −2.
    G5 = nx.Graph()
    G5.add_edges_from([(1, 2), (1, 3), (1, 4)])
    el5 = {1: "B", 2: "B", 3: "C", 4: "Br"}
    assert three_c_legs(el5, G5, {1: "3c2e"}) == {1: [(1, 2)]}
    assert three_c_unpaired_edges(el5, G5, {1: "3c2e"}) == set()

    # μ-H between two metals — no internal neighbour at all, so no internal leg
    G2 = nx.Graph()
    G2.add_nodes_from([2])
    assert three_c_legs({2: "H"}, G2, {2: "3c2e"}) == {}

    # 🔴 μ-CH₃ — internal **degree** 3, but H is not metal-like, so it has no internal leg and
    #   its `C–H` bonds are not part of the bridge. Keying on the degree instead read this
    #   carbon as `[C⁻⁴]` and cost .0058 of T10 OS on holdout.
    G4 = nx.Graph()
    G4.add_edges_from([(2, 6), (2, 7), (2, 8)])
    assert three_c_legs({2: "C", 6: "H", 7: "H", 8: "H"}, G4, {2: "3c2e"}) == {}

    # B–H–B — two internal legs, neither holds a pair, so both are reported and both subtracted
    G3 = nx.Graph()
    G3.add_edges_from([(0, 2), (1, 2)])
    el3 = {0: "B", 1: "B", 2: "H"}
    assert three_c_legs(el3, G3, {2: "3c2e"}) == {2: [(0, 2), (1, 2)]}
    assert three_c_unpaired_edges(el3, G3, {2: "3c2e"}) == {(0, 2), (1, 2)}

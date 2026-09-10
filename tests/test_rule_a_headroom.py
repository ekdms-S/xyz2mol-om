"""🔴 Regression — rule A's π headroom counts the **σ M–L bonds**, not only the internal degree.

A ring carbon carrying an H *and* a σ bond to the metal already has four σ bonds and is sp³. The
ring is still planar enough to pass `TAU_P`, so rule A pinned it `Conj` anyway — and a **pinned**
bond is not the ④ matching's to move, so nothing downstream could take the π back. That is where
`b_int(X) + b_ML(X) > CAP(X)` was coming from: on the CSD holdout 42 structures carried exactly
this violation, and in every one the reference calls both ring bonds at that atom `Single`
(`C(H)(N)(N)→M` aminal carbons — KOZFIS C13/C39, ZEJNUZ C8/C16, AGOJOW C0).

⚠️ What must **not** change: an NHC carbene carbon (degree 2 + 1 = 3), a σ-aryl *ipso* carbon
(2 + 1 = 3), and any haptic-bound ring atom (a haptic bond spends nothing, so it has no `bml`
entry at all). Those are the cases a `deg`-only test and this one agree on.
"""

# ruff: noqa: E501
from __future__ import annotations

import networkx as nx

from xyz2mol_om.config import CAP


def _sat(G, el, bml):
    """The rule A headroom set, as `rules.pipeline.predict_T3_EHT` computes it."""
    return {x for x in G.nodes if G.degree(x) + bml.get(x, 0.0) >= CAP.get(el[x], 4)}


def test_an_aminal_carbon_with_a_sigma_metal_bond_has_no_headroom():
    # C0 bonded to N, N and H, plus one σ M–L: four σ bonds, so it cannot also carry π
    G = nx.Graph([(0, 1), (0, 2), (0, 3)])
    el = ["C", "N", "N", "H"]
    assert 0 in _sat(G, el, {0: 1.0})
    assert 0 not in _sat(G, el, {})  # the degree-only test misses it — that was the bug


def test_an_nhc_carbene_carbon_keeps_its_headroom():
    # the same σ M–L, but no H: degree 2 + 1 = 3 < CAP(C)
    G = nx.Graph([(0, 1), (0, 2)])
    assert 0 not in _sat(G, ["C", "N", "N"], {0: 1.0})


def test_a_sigma_aryl_ipso_carbon_keeps_its_headroom():
    G = nx.Graph([(0, 1), (0, 2)])
    assert 0 not in _sat(G, ["C", "C", "C"], {0: 1.0})


def test_a_haptic_ring_atom_keeps_its_headroom():
    # a haptic M–L spends nothing, so it never reaches `bml` — a Cp carbon stays π-capable
    G = nx.Graph([(0, 1), (0, 2), (0, 3)])
    assert 0 not in _sat(G, ["C", "C", "C", "H"], {})

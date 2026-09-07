"""🔴 Regression — the four structural invariants the 2026-09-08 rule review fixed or relied on.

These are cheap, geometry-free unit tests on the decision functions themselves. They exist
because the corresponding defects were all silent: the metrics stayed fine while the emitted
structure was wrong, so nothing failed until someone read a picture.

  1. ④ spends **one** capacity unit per physical bond (the replica graph used to allow two).
  2. `is_cluster_frag` reads **Kekule integers**, so a substituted arene is not a cage.
  3. The ⑤ EHT trust gate fires on `–NO2` (two terminal O) and **not** on nitrate (three).
  4. The ⑥ Kekule matching is **weighted**, so among equal-cardinality Kekule structures the
     one the bond lengths prefer comes out.
"""

# ruff: noqa: E501
from __future__ import annotations

import networkx as nx

from xyz2mol_om.charge import is_cluster_frag, kekulize
from xyz2mol_om.pipeline import _eht_untrusted, _is_nitro
from xyz2mol_om.solvers import _solve_cap


def _chain(el, bonds):
    G = nx.Graph()
    G.add_nodes_from(range(len(el)))
    G.add_edges_from(bonds)
    return G


def test_cap_spends_one_unit_per_bond():
    """Two carbons with headroom 2 each, joined by one bond, must not take two units.

    The capacity reduction gives each atom two replica vertices, and a matching can hold
    `(a,0)-(b,0)` and `(a,1)-(b,1)` at once — two units spent, one `Double` emitted. 96 holdout
    structures hit this.
    """
    el = ["C", "C", "H", "H", "H", "H"]
    bonds = [(0, 1), (0, 2), (0, 3), (1, 4), (1, 5)]
    G = _chain(el, bonds)
    # ③ likelihood strongly prefers `Double` on the C-C bond, `Single` on the C-H bonds
    sc = {(0, 1): {0: 0.0, 1: 9.0, 2: -9.0}}
    sc.update({e: {0: 9.0, 1: -9.0} for e in bonds if e != (0, 1)})
    cls, _ml = _solve_cap(G, el, sc, set(), {}, {}, ml_max=2)
    assert cls[(0, 1)] == 1, "the C=C should be raised"
    # every carbon stays inside its cap — that is what the second unit would have broken
    for x in (0, 1):
        used = sum({0: 1.0, 1: 2.0, 2: 3.0}[cls[(min(x, y), max(x, y))]] for y in G[x])
        assert used <= 4.0 + 1e-9, f"atom {x} spent {used} > cap 4"


def test_substituted_arene_is_not_a_cluster():
    """`b_int > CAP` on `Conj` = 1.5 made a benzene carbon read 4.5 > 4.

    The cluster test exists for carborane cages; an ordinary arene taking that branch replaced
    its formal-charge sum with the EHT number (906 fragment-level cluster calls on holdout, of
    which 747 were this artifact).
    """
    el = ["C"] * 7 + ["H"] * 8              # toluene: ring C0..C5, methyl C6
    ring = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 5)]
    G = _chain(el, ring + [(0, 6), (1, 7), (2, 8), (3, 9), (4, 10), (5, 11),
                           (6, 12), (6, 13), (6, 14)])
    cls = {e: 3 for e in ring}              # every ring bond is `Conj`
    cls.update({e: 0 for e in G.edges if e not in cls and (e[1], e[0]) not in cls})
    orders, _fq = kekulize(G, el, cls)
    assert not is_cluster_frag(G, el, cls, list(G.nodes), orders=orders)


def test_eht_trust_gate_scope():
    """Nitro is skipped, nitrate is not — the detector counts terminal O **exactly**.

    Holdout: the EHT fragment-charge target is wrong for 54 of 64 `R–NO2` and 13 of 13 free
    `NO2-`, and for **0 of 75** nitrate fragments.
    """
    # nitrobenzene fragment core: C0-N1(=O2)(=O3)
    el_nitro = ["C", "N", "O", "O"]
    G_nitro = _chain(el_nitro, [(0, 1), (1, 2), (1, 3)])
    assert _is_nitro(G_nitro, el_nitro, 1)
    assert _eht_untrusted(G_nitro, el_nitro, set(G_nitro.nodes)) == "nitro"

    # nitrate: N0 with three terminal O
    el_no3 = ["N", "O", "O", "O"]
    G_no3 = _chain(el_no3, [(0, 1), (0, 2), (0, 3)])
    assert not _is_nitro(G_no3, el_no3, 0)
    assert _eht_untrusted(G_no3, el_no3, set(G_no3.nodes)) != "nitro"


def test_kekule_matching_follows_the_likelihood():
    """Benzene has two Kekule structures of the same size — the weight must decide which.

    Unweighted, `max_weight_matching(maxcardinality=True)` returned whichever the library
    happened to build, so the emitted double bonds were unrelated to the bond lengths (`EMAXAR`
    put the pi on a 1.440 A C-C instead of a 1.234 A C=O).
    """
    el = ["C"] * 6
    ring = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 5)]
    G = _chain(el, ring)
    cls = dict.fromkeys(ring, 3)
    a, b = [(0, 1), (2, 3), (4, 5)], [(1, 2), (3, 4), (0, 5)]
    for want, other in ((a, b), (b, a)):
        w = {**dict.fromkeys(want, 5.0), **dict.fromkeys(other, 0.0)}
        orders, _ = kekulize(G, el, cls, w=w)
        assert all(orders[e] == 2.0 for e in want), f"the pi should sit on {want}"
        assert all(orders[e] == 1.0 for e in other)

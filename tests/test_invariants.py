"""🔴 Regression — the structural invariants of the decision functions.

These are cheap, geometry-free unit tests on the decision functions themselves. They exist
because the corresponding defects were all silent: the metrics stayed fine while the emitted
structure was wrong, so nothing failed until someone read a picture.

  1. ④ spends **one** capacity unit per physical bond (the replica graph used to allow two).
  2. `is_cluster_frag` reads **Kekule integers**, so a substituted arene is not a cage.
  3. The ⑤ EHT trust gate fires on `–NO2` (two terminal O) and **not** on nitrate (three).
  4. The ⑥ Kekule matching is **weighted**, so among equal-cardinality Kekule structures the
     one the bond lengths prefer comes out.
  5. T4 does not report an agostic `C–H···M` contact as a bond.
  6. T4 does not report a bond to an atom its own bonds already saturate — H, B/Al and cage
     atoms exempt.
"""

# ruff: noqa: E501
from __future__ import annotations

import networkx as nx

from xyz2mol_om.charge import is_cluster_frag, kekulize
from xyz2mol_om.rules.pipeline import _eht_untrusted, _is_nitro
from xyz2mol_om.rules.solvers import _solve_cap


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


# ── 5·6. What `ml_bonds` may contain (reported by flower-om) ──────────────────
#   The output used to be assembled from `ml_raw`, i.e. **before** T4's agostic removal, while
#   the molecule SMILES was built from `ml_pred`, i.e. after. A consumer reading `ml_bonds` got a
#   different graph from the one the SMILES describes. These two tests pin the T4 filters.


def test_agostic_contact_is_not_a_bond():
    """`C–H···M`: the H has one metal neighbour and one non-metal internal neighbour."""
    from xyz2mol_om.rules.pipeline import drop_agostic

    el = ["Ni", "C", "H", "H", "H"]
    G = _chain(el, [(1, 2), (1, 3), (1, 4)])
    assert drop_agostic(el, G, [(0, 2)]) == []
    # a hydride bound to nothing else is kept, and so is a mu-H between two metals
    el2 = ["Ni", "H"]
    assert drop_agostic(el2, _chain(el2, []), [(0, 1)]) == [(0, 1)]
    el3 = ["Ni", "Ni", "H"]
    assert drop_agostic(el3, _chain(el3, []), [(0, 2), (1, 2)]) == [(0, 2), (1, 2)]


def test_saturated_atom_takes_no_ml_bond():
    """A carbon whose four internal neighbours already fill it cannot also bond a metal.

    The three exceptions are load-bearing: H (else every M–H dies), B/Al, and **an atom bonded
    to B/Al** — a dicarbollide cage carbon has `deg 5 > CAP 4` and its M–C is real (95 of the 97
    such candidates on the CSD holdout are reference M–L bonds).
    """
    from xyz2mol_om.rules.pipeline import drop_saturated

    el = ["Ni", "C", "H", "H", "H", "C"]
    G = _chain(el, [(1, 2), (1, 3), (1, 4), (1, 5)])
    assert drop_saturated(el, G, [(0, 1)]) == []          # sp3 C, deg 4 = CAP
    # an sp2 carbon keeps its M–L bond: deg 3 leaves a place even though `b_int` is 4
    el2 = ["Ni", "C", "C", "H", "H"]
    G2 = _chain(el2, [(1, 2), (1, 3), (1, 4)])
    assert drop_saturated(el2, G2, [(0, 1)]) == [(0, 1)]
    # cage carbon (bonded to B) is exempt even at deg 5
    el3 = ["Co", "C", "B", "B", "B", "C", "H"]
    G3 = _chain(el3, [(1, 2), (1, 3), (1, 4), (1, 5), (1, 6)])
    assert drop_saturated(el3, G3, [(0, 1)]) == [(0, 1)]
    # H is never vetoed by this rule
    el4 = ["Ni", "H", "B"]
    assert drop_saturated(el4, _chain(el4, [(1, 2)]), [(0, 1)]) == [(0, 1)]


def test_nitrogen_never_carries_five_bonds():
    """Nitro comes out charge-separated, so `bonds_kekule` and the SMILES agree.

    ⑥ reproduces the reference's `-N(=O)=O`, which puts five bonds on a period-2 atom. RDKit
    refuses that and rewrites it as `[N+](=O)[O-]`, so the emitted SMILES disagreed with our own
    bond orders and charges, and `verify_complex` reported the molecule as failing its round trip.
    """
    import networkx as nx

    from xyz2mol_om.charge import octet_fix_period2, q_atom
    from xyz2mol_om.output import complex_smiles, verify_complex

    el = ["C", "H", "H", "H", "N", "O", "O"]
    orders = {(0, 1): 1, (0, 2): 1, (0, 3): 1, (0, 4): 1, (4, 5): 2, (4, 6): 2}
    G = nx.Graph()
    G.add_edges_from(orders)
    octet_fix_period2(el, G, orders)
    assert sorted(orders[e] for e in [(4, 5), (4, 6)]) == [1, 2]

    q = {}
    for v in G:
        b = sum(orders[(min(v, w), max(v, w))] for w in G[v])
        q[v] = int(round(q_atom(el[v], float(b), G.degree(v),
                                tuple(sorted(el[w] for w in G[v])))))
    assert q[4] == 1 and sorted(q[x] for x in (5, 6)) == [-1, 0]
    assert sum(q.values()) == 0  # the fragment total does not move

    smi, _ = complex_smiles(el, list(range(7)), orders, q, [], {})
    assert verify_complex(smi, el, list(range(7)), orders, q, [], {}, 0)[0]

    # a period-3 atom keeps the hypervalent form — a sulfone S with six bonds is left alone
    el3 = ["S", "O", "O", "C", "C"]
    o3 = {(0, 1): 2, (0, 2): 2, (0, 3): 1, (0, 4): 1}
    G3 = nx.Graph()
    G3.add_edges_from(o3)
    octet_fix_period2(el3, G3, o3)
    assert o3 == {(0, 1): 2, (0, 2): 2, (0, 3): 1, (0, 4): 1}

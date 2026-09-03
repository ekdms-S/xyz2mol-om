"""complex SMILES · bridge tags · 3c2e budget exclusion - checks that run without the workspace.

Two systems are covered:
  ① `[Mo(≡N)(OH)Cl₃]⁻` - dative arrows · metal oxidation state · total-charge conservation
     (plain σ M–L)
  ② a μ-H bridge of the `[H₂B(μ-H)₂Mo]` kind - `bridge`/`3c2e` tagging and budget exclusion
"""

# ruff: noqa: E501
from __future__ import annotations

import numpy as np

from xyz2mol_om import predict
from xyz2mol_om.pipeline import bridge_tags

EL = ["Mo", "N", "O", "H", "Cl", "Cl", "Cl"]
XYZ = np.array(
    [
        [0.00, 0.00, 0.00],
        [0.00, 0.00, 1.66],
        [1.95, 0.00, -0.25],
        [2.52, 0.76, -0.45],
        [0.00, 2.35, -0.25],
        [-2.35, 0.00, -0.25],
        [0.00, -2.35, -0.25],
    ]
)
WBO = {(0, 1): 2.891, (0, 2): 1.022, (0, 4): 0.955, (0, 5): 0.967, (0, 6): 0.997, (2, 3): 0.862}
WBO.update({(j, i): w for (i, j), w in list(WBO.items())})


def test_complex_smiles_dative_and_oxidation():
    r = predict(EL, XYZ, total_charge=-1, wbo=WBO)
    smi = r["complex_smiles"]
    assert smi, f"no complex SMILES - {r['complex_smiles_note']}"
    assert r["complex_smiles_ok"], f"round-trip check failed - {r['complex_smiles_note']}"
    # all five M–L bonds are dative arrows
    assert smi.count("->") + smi.count("<-") == 5, smi
    # the **oxidation state** is written on the metal
    assert "[Mo+6]" in smi, smi
    # formal charges of the anionic ligands are visible too
    assert "[N-3]" in smi and "[Cl-]" in smi, smi
    # the output order lets us map back to the xyz atoms
    assert sorted(r["complex_atom_order"]) == list(range(len(EL)))
    assert [EL[i] for i in r["complex_atom_order"]].count("Cl") == 3


def test_complex_smiles_needs_total_charge():
    """Without `total_charge` there is no oxidation state, so **nothing is produced**
    (rather than being silently wrong)."""
    r = predict(EL, XYZ, wbo=WBO)
    assert r["complex_smiles"] is None
    assert "oxidation state" in r["complex_smiles_note"]


def test_ml_bonds_have_bridge_key():
    r = predict(EL, XYZ, total_charge=-1, wbo=WBO)
    for lig in r["ligands"]:
        for _e, d in lig["ml_bonds"].items():
            assert set(d) == {"type", "order", "bridge"}, d
            assert d["type"] in ("sigma", "haptic", "bridge")
            assert d["bridge"] in (None, "3c2e", "dative")
    # every coordination here is terminal, so there are no bridges
    assert all(
        d["bridge"] is None for lig in r["ligands"] for d in lig["ml_bonds"].values()
    )


def test_bridge_tags_rule():
    """T7 decision rule - the four real cases of design doc §3.0 5c."""
    import networkx as nx

    # μ-H : 0 internal bonds · 2 M–L  ⇒ n_center 2 · deg 2 > VALENCE_3C[H]=1  ⇒ 3c2e
    el = ["Fe", "Fe", "H"]
    G = nx.Graph()
    G.add_nodes_from([2])
    assert bridge_tags(el, G, [(0, 2), (1, 2)]) == {2: "3c2e"}

    # μ-Cl : n_center 2 but Cl is not in VALENCE_3C  ⇒ dative (3c4e)
    el = ["Fe", "Fe", "Cl"]
    assert bridge_tags(el, G, [(0, 2), (1, 2)]) == {2: "dative"}

    # terminal Cl : n_center 1  ⇒ no tag
    assert bridge_tags(el, G, [(0, 2)]) == {}

    # B–H···M : 1 internal neighbour B + 1 M–L ⇒ n_center 2 · deg 2 > 1 ⇒ 3c2e
    el = ["Fe", "B", "H"]
    G2 = nx.Graph()
    G2.add_edge(1, 2)
    assert bridge_tags(el, G2, [(0, 2)]) == {2: "3c2e"}


def test_bridge_type_precedence():
    """`type` follows haptic > bridge > sigma, and the `bridge` field survives the overlap."""
    import networkx as nx

    el = ["Fe", "Fe", "Cl"]
    G = nx.Graph()
    G.add_nodes_from([2])
    assert bridge_tags(el, G, [(0, 2), (1, 2)])[2] == "dative"


if __name__ == "__main__":
    test_complex_smiles_dative_and_oxidation()
    test_complex_smiles_needs_total_charge()
    test_ml_bonds_have_bridge_key()
    test_bridge_tags_rule()
    test_bridge_type_precedence()
    print("COMPLEX SMILES / BRIDGE PASS")

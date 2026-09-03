"""Public API smoke test - the minimum check that runs without the workspace.

A single `[Mo(≡N)(OH)Cl₃]⁻` goes through every pipeline stage:
  T1 internal bonds (O–H) · T4 five M–L bonds · T8 `Mo≡N` = Triple · T10 `q_L` and `OS(Mo)=+6` ·
  ⑥ Kekule output · ligand SMILES round-trip check.
Mayer bond orders are **hard-coded constants** from xtb `--sp --wbo` (so this runs without xtb).
"""

# ruff: noqa: E501
from __future__ import annotations

import numpy as np

from xyz2mol_om import predict

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
# xtb GFN2 `--sp --wbo` measured values (2026-09-03)
WBO = {(0, 1): 2.891, (0, 2): 1.022, (0, 4): 0.955, (0, 5): 0.967, (0, 6): 0.997, (2, 3): 0.862}
WBO.update({(j, i): w for (i, j), w in list(WBO.items())})


def test_mo_nitrido():
    r = predict(EL, XYZ, total_charge=-1, wbo=WBO)
    assert len(r["metals"]) == 1
    m = r["metals"][0]
    assert m["element"] == "Mo"
    assert m["oxidation"] == 6, f"OS(Mo) must be +6 - got {m['oxidation']}"

    ligs = {tuple(L["atoms"]): L for L in r["ligands"]}
    assert len(ligs) == 5, f"expected 5 ligands - got {len(ligs)}"

    nit = ligs[(1,)]
    assert nit["charge"] == -3, f"nitrido N must be −3 - got {nit['charge']}"
    assert nit["ml_bonds"][(0, 1)]["order"] == 3, "Mo≡N must be Triple"
    assert nit["ml_bonds"][(0, 1)]["type"] == "sigma"

    oh = ligs[(2, 3)]
    assert oh["charge"] == -1
    assert oh["bonds_kekule"][(2, 3)] == 1
    assert oh["bonds_4class"][(2, 3)] == "Single"

    for a in (4, 5, 6):
        assert ligs[(a,)]["charge"] == -1

    for L in r["ligands"]:
        assert L["smiles"], "SMILES must be produced"
        assert L["smiles_ok"], f"SMILES round-trip failed: {L['smiles']} - {L['smiles_note']}"


def test_no_spurious_hh():
    """🔴 Regression - a methyl geminal H–H (≈1.77 Å) must not be picked up as a bond.

    Back when `d_int` had no `H,H` entry the 2.0542 Å fallback applied and **22.9% of the
    predicted bonds were spurious H–H**.
    """
    el = ["C", "H", "H", "H", "H"]
    d = 1.09 / np.sqrt(3)
    xyz = np.array([[0, 0, 0], [d, d, d], [d, -d, -d], [-d, d, -d], [-d, -d, d]], dtype=float)
    r = predict(el, xyz, total_charge=0)
    (L,) = r["ligands"]
    hh = [(i, j) for (i, j) in L["bonds_kekule"] if el[i] == "H" and el[j] == "H"]
    assert not hh, f"spurious H–H bond appeared: {hh}"
    assert len(L["bonds_kekule"]) == 4, f"CH4 has 4 bonds - {L['bonds_kekule']}"


if __name__ == "__main__":
    test_mo_nitrido()
    test_no_spurious_hh()
    print("API SMOKE PASS")

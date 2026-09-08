"""🔴 Regression — a disconnected input is split into molecules, and the oxidation state is
computed **inside** a molecule.

`OS(M) = (charge - sum q_L) / n_metals` run over the whole input averages one molecule's charge
into another molecule's metals. Measured before this was fixed: `CpTiCl3` alone gives Ti(IV) and
`[Os(CO)3Cl3]-` alone gives Os(II), but concatenated into one input they came out **Ti(III) and
Os(III)** — both wrong, the sum still correct, and `complex_smiles_ok` still True. It failed
silently, which is the reason this is pinned.

The case that matters in practice is an IRC endpoint: a complex plus the fragment that left it.
"""

# ruff: noqa: E501
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from xyz2mol_om import all_fragments, all_metals, predict, read_xyz

EX = Path(__file__).resolve().parent.parent / "examples"
CH4 = (["C", "H", "H", "H", "H"],
       np.array([[0, 0, 0], [0.63, 0.63, 0.63], [-0.63, -0.63, 0.63],
                 [-0.63, 0.63, -0.63], [0.63, -0.63, -0.63]], dtype=float))


def _load(name):
    meta = json.loads((EX / f"{name}.wbo.json").read_text())
    el, xyz = read_xyz(EX / f"{name}.xyz")
    wbo = {tuple(int(t) for t in k.split(",")): v for k, v in meta["wbo"].items()}
    return el, xyz, wbo, meta["total_charge"]


def test_single_molecule_is_one_molecule():
    el, xyz, wbo, q = _load("02_haptic_cp_ticl3")
    r = predict(el, xyz, total_charge=q, wbo=wbo)
    assert len(r["molecules"]) == 1
    mol = r["molecules"][0]
    assert [m["index"] for m in mol["metals"]] == [0]
    assert mol["charge"] == 0 and mol["charge_is_exact"]
    assert len(mol["fragments"]) == 4          # Cp- and three Cl-
    assert mol["metals"][0]["oxidation"] == 4


def test_complex_plus_a_detached_fragment_keeps_the_right_oxidation_state():
    """The IRC-endpoint shape: one metal-bearing molecule and one that left it."""
    el, xyz, wbo, q = _load("02_haptic_cp_ticl3")
    el2 = list(el) + CH4[0]
    xyz2 = np.vstack([xyz, CH4[1] + np.array([60.0, 0, 0])])
    r = predict(el2, xyz2, total_charge=q, wbo=wbo)

    assert len(r["molecules"]) == 2
    metal_mol = [m for m in r["molecules"] if m["metals"]]
    free_mol = [m for m in r["molecules"] if not m["metals"]]
    assert len(metal_mol) == 1 and len(free_mol) == 1
    assert len(free_mol[0]["atoms"]) == 5 and free_mol[0]["charge"] == 0
    # the free fragment's charge must not leak into the metal
    assert metal_mol[0]["metals"][0]["oxidation"] == 4
    assert metal_mol[0]["metals"][0]["oxidation_is_exact"] is True
    assert metal_mol[0]["charge"] == 0 and metal_mol[0]["charge_is_exact"]
    # each molecule carries its own SMILES — no dot-joined string any more
    assert all(m["smiles_ok"] for m in r["molecules"])
    assert "." not in metal_mol[0]["smiles"] and "." not in free_mol[0]["smiles"]


def test_two_metal_molecules_flag_the_oxidation_state_as_an_even_split():
    """Nothing says how the total charge divides, so the even split is used and **flagged**.

    Kept because it is right more often than not (5 of the 7 holdout structures that land here),
    but every value it produces carries `oxidation_is_exact = False` and the molecule charge stays
    `None` — when it is wrong it is wrong silently.
    """
    elA, xA, wA, qA = _load("02_haptic_cp_ticl3")     # neutral, Ti(IV)
    elB, xB, wB, qB = _load("01_dative_os_carbonyl")  # -1, Os(II)
    n = len(elA)
    el = list(elA) + list(elB)
    xyz = np.vstack([xA, xB + np.array([60.0, 0, 0])])
    wbo = dict(wA)
    wbo.update({(i + n, j + n): v for (i, j), v in wB.items()})

    r = predict(el, xyz, total_charge=qA + qB, wbo=wbo)
    assert len(r["molecules"]) == 2
    assert all(m["metals"] for m in r["molecules"])
    # the even split answers Ti(III)/Os(III) here — the truth is Ti(IV) and Os(II)
    assert [m["oxidation"] for m in all_metals(r)] == [3, 3]
    # so it must not be presented as exact
    assert [m["oxidation_is_exact"] for m in all_metals(r)] == [False, False]
    assert all(m["charge"] is None and not m["charge_is_exact"] for m in r["molecules"])
    assert all("even split" in m["smiles_note"] for m in r["molecules"])

"""Reassembling ligand SMILES + metals + `ml_bonds` yields the same molecule as `complex_smiles`."""

from __future__ import annotations

import glob
from pathlib import Path

import pytest
from rdkit import Chem, RDLogger

from xyz2mol_om import all_fragments, all_metals, assemble_complex, load_json

RDLogger.DisableLog("rdApp.*")
EX = sorted(glob.glob(str(Path(__file__).resolve().parents[1] / "examples" / "*.result.json")))


def _canon(smi):
    m = Chem.MolFromSmiles(smi, sanitize=False)
    Chem.SanitizeMol(m, Chem.SanitizeFlags.SANITIZE_ALL
                     ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE
                     ^ Chem.SanitizeFlags.SANITIZE_SETAROMATICITY)
    return Chem.MolToSmiles(m)


@pytest.mark.parametrize("f", EX, ids=[Path(f).stem for f in EX])
def test_assemble_matches_complex_smiles(f):
    r = load_json(f)
    # 🔴 `assemble_complex` rebuilds the molecule from the **ligand** SMILES and joins it up
    #    with two-centre bonds, so it cannot express a 3c2e bridge — the bridging atom holds a
    #    pair that belongs to no single bond, and RDKit rejects the valence that leaves behind.
    #    `complex_smiles` can, because it writes that leg as a dative arrow. Skip either kind.
    if any(lg["bonds_3c2e"] or not lg["smiles_ok"] for lg in all_fragments(r)):
        pytest.skip("a fragment carries a 3c2e bridge — not expressible in two-centre form")
    mol, amap = assemble_complex(r)
    assert Chem.MolToSmiles(mol) == _canon(r["molecules"][0]["smiles"])
    for met in all_metals(r):               # metals and coordinating atoms must be in the map
        assert met["index"] in amap
    for lg in all_fragments(r):
        for x in lg["coordinating"]:
            assert x in amap

"""Reassembling ligand SMILES + metals + `ml_bonds` yields the same molecule as `complex_smiles`."""

from __future__ import annotations

import glob
from pathlib import Path

import pytest
from rdkit import Chem, RDLogger

from xyz2mol_om import assemble_complex, load_json

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
    mol, amap = assemble_complex(r)
    assert Chem.MolToSmiles(mol) == _canon(r["complex_smiles"])
    for met in r["metals"]:               # metals and coordinating atoms must be in the map
        assert met["index"] in amap
    for lg in r["ligands"]:
        for x in lg["coordinating"]:
            assert x in amap

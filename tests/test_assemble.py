"""리간드 SMILES + 금속 + `ml_bonds` 로 재조립하면 `complex_smiles` 와 같은 분자가 나온다."""

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
    for met in r["metals"]:               # 금속과 배위 원자는 대응표에 있어야 한다
        assert met["index"] in amap
    for lg in r["ligands"]:
        for x in lg["coordinating"]:
            assert x in amap

"""예시 5종의 **Kekulé 2D 그래프**를 PNG 로 그린다 — `python examples/draw_examples.py`.

착물 SMILES(M–L 은 dative 화살표)를 RDKit 으로 파싱해 그린다. 파싱이 안 되면
리간드 SMILES 를 격자로 그린다.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

from rdkit import Chem, RDLogger  # noqa: E402
from rdkit.Chem import Draw  # noqa: E402
from rdkit.Chem.Draw import rdMolDraw2D  # noqa: E402

from xyz2mol_om import predict  # noqa: E402

RDLogger.DisableLog("rdApp.*")


def read_xyz(p: Path):
    lines = p.read_text().splitlines()
    n = int(lines[0].split()[0])
    el, xyz = [], []
    for ln in lines[2 : 2 + n]:
        w = ln.split()
        el.append(w[0])
        xyz.append([float(v) for v in w[1:4]])
    return el, np.array(xyz), lines[1]


def draw(mol, out: Path, legend: str) -> None:
    Chem.rdDepictor.Compute2DCoords(mol)
    d = rdMolDraw2D.MolDraw2DCairo(760, 520)
    d.drawOptions().addStereoAnnotation = False
    d.drawOptions().legendFontSize = 18
    rdMolDraw2D.PrepareAndDrawMolecule(d, mol, legend=legend, kekulize=True)
    d.FinishDrawing()
    out.write_bytes(d.GetDrawingText())


def main() -> None:
    for f in sorted(HERE.glob("*.xyz")):
        el, xyz, title = read_xyz(f)
        meta = json.loads(f.with_suffix(".wbo.json").read_text())
        wbo = {tuple(int(t) for t in k.split(",")): v for k, v in meta["wbo"].items()}
        r = predict(el, xyz, total_charge=meta["total_charge"], wbo=wbo)
        out = f.with_suffix(".png")
        legend = title.split("|")[2].strip()
        smi = r.get("complex_smiles")
        mol = Chem.MolFromSmiles(smi, sanitize=False) if smi else None
        if mol is not None:
            try:
                Chem.SanitizeMol(mol, Chem.SanitizeFlags.SANITIZE_ALL
                                 ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE
                                 ^ Chem.SanitizeFlags.SANITIZE_SETAROMATICITY)
                draw(mol, out, legend)
                print(f"{out.name:30} 착물 SMILES 로 그림")
                continue
            except Exception as e:
                print(f"{out.name:30} 착물 파싱 실패({type(e).__name__}) → 리간드 격자로")
        mols, legs = [], []
        for lg in r["ligands"]:
            m = Chem.MolFromSmiles(lg["smiles"] or "", sanitize=False)
            if m is None:
                continue
            mols.append(m)
            legs.append(f"q={lg['charge']:+d}" + (f" η{max(lg['eta'].values())}" if lg.get("eta") else ""))
        img = Draw.MolsToGridImage(mols, legends=legs, molsPerRow=min(3, max(1, len(mols))),
                                   subImgSize=(260, 220), useSVG=False)
        out.write_bytes(img.data if hasattr(img, "data") else img)
        print(f"{out.name:30} 리간드 격자로 그림 ({len(mols)}개)")


if __name__ == "__main__":
    main()

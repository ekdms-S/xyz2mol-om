"""Draw the **Kekule 2D graph** of the five examples as PNG - `python examples/draw_examples.py`.

The complex SMILES (M–L as dative arrows) is parsed with RDKit and drawn. If it cannot be
parsed, the ligand SMILES are drawn as a grid instead.
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
from rdkit.Chem import rdCoordGen  # noqa: E402
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


METALS_FOR_H = {"Ti", "Zr", "Hf", "V", "Nb", "Ta", "Cr", "Mo", "W", "Mn", "Re", "Fe", "Ru",
                "Os", "Co", "Rh", "Ir", "Ni", "Pd", "Pt", "Cu", "Ag", "Au", "Zn", "Sc", "Y",
                "La", "Ce", "Al", "Ga", "In", "Sn", "Pb", "Mg", "B"}


def display_copy(mol):
    """Drawing copy: drop terminal H. Keep H on a metal (hydrido / 3c2e bridge).

    Every H is explicit in our molecules, so drawing them all makes the picture unreadable.
    """
    keep = set()
    for a in mol.GetAtoms():
        if a.GetSymbol() != "H":
            continue
        nb = list(a.GetNeighbors())
        if len(nb) != 1 or nb[0].GetSymbol() in METALS_FOR_H:
            keep.add(a.GetIdx())
    rw = Chem.RWMol(mol)
    for i in sorted((a.GetIdx() for a in mol.GetAtoms()
                     if a.GetSymbol() == "H" and a.GetIdx() not in keep), reverse=True):
        rw.RemoveAtom(i)
    m = rw.GetMol()
    for a in m.GetAtoms():
        a.SetNoImplicit(True)
    return m


def draw(mol, out: Path, legend: str) -> None:
    m = display_copy(mol)
    rdCoordGen.AddCoords(m)          # far better than the default generator for complexes
    d = rdMolDraw2D.MolDraw2DCairo(1100, 820)
    o = d.drawOptions()
    o.addStereoAnnotation = False
    o.legendFontSize = 20
    o.minFontSize = 13
    o.maxFontSize = 22
    o.bondLineWidth = 2
    o.padding = 0.08
    o.multipleBondOffset = 0.18
    rdMolDraw2D.PrepareAndDrawMolecule(d, m, legend=legend, kekulize=True)
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
                print(f"{out.name:30} drawn from complex SMILES")
                continue
            except Exception as e:
                print(f"{out.name:30} complex parse failed({type(e).__name__}) -> ligand grid")
        mols, legs = [], []
        for lg in r["ligands"]:
            m = Chem.MolFromSmiles(lg["smiles"] or "", sanitize=False)
            if m is None:
                continue
            m = display_copy(m)
            rdCoordGen.AddCoords(m)
            mols.append(m)
            legs.append(f"q={lg['charge']:+d}" + (f" η{max(lg['eta'].values())}" if lg.get("eta") else ""))
        img = Draw.MolsToGridImage(mols, legends=legs, molsPerRow=min(3, max(1, len(mols))),
                                   subImgSize=(340, 300), useSVG=False)
        out.write_bytes(img.data if hasattr(img, "data") else img)
        print(f"{out.name:30} drawn as ligand grid ({len(mols)} fragments)")


if __name__ == "__main__":
    main()

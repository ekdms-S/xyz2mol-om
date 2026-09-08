"""Reassemble a `predict()` result into an **RDKit complex molecule**.

`complex_smiles` gives the complex as one string but **collapses the M-L orders**. Assembling
directly from ligand SMILES + metals + `ml_bonds` lets you put in the orders you want and keeps
the atom correspondence in hand.

    from xyz2mol_om import predict, assemble_complex
    mol, atom_map = assemble_complex(predict(el, xyz, total_charge=0, wbo=wbo))
    #   atom_map : {input atom index -> mol atom index}   (coordinating atoms and metals only)

⚠️ **The atom map `[X:n]` in a ligand SMILES is not the input index** — it is the **n-th
(1-based)** entry of that ligand's sorted `coordinating` list. This function does that mapping.
"""

from __future__ import annotations

from rdkit import Chem

_BT = {1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE, 3: Chem.BondType.TRIPLE}


def assemble_complex(r: dict, ml_dative: bool = True):
    """`(mol, {input index: mol index})`. With `ml_dative=False`, M-L bonds are added with the
    integer order from `ml_bonds[...]["order"]` (haptic has no order, so it is always dative)."""
    m = Chem.RWMol()
    pos: dict[int, int] = {}
    for met in r["metals"]:
        a = Chem.Atom(met["element"])
        a.SetNoImplicit(True)
        a.SetFormalCharge(int(met["oxidation"] or 0))
        pos[met["index"]] = m.AddAtom(a)
    for lg in r["ligands"]:
        sub = Chem.MolFromSmiles(lg["smiles"] or "", sanitize=False)
        if sub is None:
            raise ValueError(f"ligand {lg['index']} SMILES parse failed: {lg['smiles']!r}")
        coord = sorted(lg["coordinating"])
        loc = {}
        for at in sub.GetAtoms():
            b = Chem.Atom(at.GetSymbol())
            b.SetNoImplicit(True)
            b.SetNumExplicitHs(0)
            b.SetFormalCharge(at.GetFormalCharge())
            loc[at.GetIdx()] = m.AddAtom(b)
            n = at.GetAtomMapNum()
            if n:
                pos[coord[n - 1]] = loc[at.GetIdx()]
        for bd in sub.GetBonds():
            m.AddBond(loc[bd.GetBeginAtomIdx()], loc[bd.GetEndAtomIdx()], bd.GetBondType())
    for lg in r["ligands"]:
        for (mi, x), d in lg["ml_bonds"].items():
            if ml_dative or d.get("type") == "haptic" or not d.get("order"):
                m.AddBond(pos[x], pos[mi], Chem.BondType.DATIVE)
            else:
                m.AddBond(pos[x], pos[mi], _BT.get(int(d["order"]), Chem.BondType.SINGLE))
    seen = set()
    for met in r["metals"]:
        for (a, b), o in (met.get("mm_bonds") or {}).items():
            if (a, b) in seen:
                continue
            seen.add((a, b))
            m.AddBond(pos[a], pos[b], _BT.get(int(o), Chem.BondType.SINGLE))
    mol = m.GetMol()
    Chem.SanitizeMol(
        mol,
        Chem.SanitizeFlags.SANITIZE_ALL
        ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE
        ^ Chem.SanitizeFlags.SANITIZE_SETAROMATICITY,
    )
    return mol, pos

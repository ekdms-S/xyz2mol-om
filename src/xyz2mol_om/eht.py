"""Extended Hückel fragment charges — the target values of step ⑤.

⚠️ **Ported from `ognm-bh-workspace/code/analysis/scratch/260830_fit_t10_charge.py`**
(2026-09-03). Function bodies were moved **verbatim** — the decision rules are unchanged.
"""

# ruff: noqa: E501
from __future__ import annotations

import collections

import networkx as nx
import numpy as np

from .config import EHT_CUTOFF, METALS, _EHT_VE


def eht_frag_charges(el, xyz, G):
    """Extended Hückel **charge per connected component (= ligand fragment)**.

    Returns {min atom idx of fragment: q}. Ported verbatim from xyz2mol_tm
    `get_proposed_ligand_charge` (HOMO/LUMO correction loop included) —
    cf/code/xyz2mol_tm/.../xyz2mol_tmc.py:227. A fragment that fails gets no key.
    """
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdEHTTools
    from rdkit.Geometry import Point3D

    RDLogger.DisableLog("rdApp.*")
    out = {}
    for comp in nx.connected_components(G):
        atoms = sorted(comp)
        m = Chem.RWMol()
        idx = {a: m.AddAtom(Chem.Atom(el[a])) for a in atoms}
        conf = Chem.Conformer(len(atoms))
        for a in atoms:
            conf.SetAtomPosition(idx[a], Point3D(*[float(v) for v in xyz[a]]))
        mol = m.GetMol()
        mol.AddConformer(conf)
        try:
            passed, result = rdEHTTools.RunMol(mol)
        except Exception:
            continue
        if not passed:
            continue
        E = list(result.GetOrbitalEnergies())
        nocc = sum(1 for e in E if e < EHT_CUTOFF)
        q = sum(_EHT_VE.get(el[a], 4) for a in atoms) - 2 * nocc
        homo = E[nocc - 1] if nocc >= 1 else float("nan")
        lumo = float("nan") if nocc >= len(E) else E[nocc]
        while q >= 1 and lumo == lumo and lumo < -9:
            nocc += 1
            q -= 2
            if nocc >= len(E):
                break
            lumo = E[nocc]
        while q < -1 and homo == homo and homo > -10.2:
            nocc -= 1
            q += 2
            if nocc < 1:
                break
            homo = E[nocc - 1]
        out[atoms[0]] = q
    return out

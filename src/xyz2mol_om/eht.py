"""확장 Hückel 조각 전하 — ⑤ 단계의 목표값.

⚠️ **`ognm-bh-workspace/code/analysis/scratch/260830_fit_t10_charge.py` 에서 이관한 코드다**
(2026-09-03). 함수 본문은 **그대로** 옮겼다 — 판정 규칙을 바꾸지 않는다.
"""

# ruff: noqa: E501
from __future__ import annotations

import collections

import networkx as nx
import numpy as np

from .config import EHT_CUTOFF, METALS, _EHT_VE


def eht_frag_charges(el, xyz, G):
    """확장 Hückel 로 **연결 성분(=리간드 조각)마다 전하**를 낸다. {조각 최소원자idx: q}.

    xyz2mol_tm `get_proposed_ligand_charge` 를 그대로 옮겼다 (HOMO/LUMO 보정 루프 포함) —
    cf/code/xyz2mol_tm/.../xyz2mol_tmc.py:227. 실패한 조각은 키를 안 만든다.
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

"""`predict()` 결과 → **RDKit 착물 분자**로 되조립한다.

`complex_smiles` 는 착물을 한 문자열로 주지만 **M–L 차수를 뭉갠다**. 리간드 SMILES + 금속 +
`ml_bonds` 로 직접 조립하면 차수를 원하는 대로 넣을 수 있고, 원자 대응도 손에 남는다.

    from xyz2mol_om import predict, assemble_complex
    mol, atom_map = assemble_complex(predict(el, xyz, total_charge=0, wbo=wbo))
    #   atom_map : {입력 원자 인덱스 -> mol 원자 인덱스}   (배위 원자·금속만)

⚠️ **리간드 SMILES 의 원자 맵 `[X:n]` 은 입력 인덱스가 아니다** — 그 리간드의
`coordinating` 을 정렬한 목록의 **n 번째(1-based)** 다. 이 함수가 그 변환을 한다.
"""

from __future__ import annotations

from rdkit import Chem

_BT = {1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE, 3: Chem.BondType.TRIPLE}


def assemble_complex(r: dict, ml_dative: bool = True):
    """`(mol, {입력 인덱스: mol 인덱스})`. `ml_dative=False` 면 M–L 을 `ml_bonds[…]["order"]`
    정수 차수로 넣는다(하프틱은 차수가 없으므로 언제나 dative)."""
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
            raise ValueError(f"리간드 {lg['index']} SMILES 파싱 실패: {lg['smiles']!r}")
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

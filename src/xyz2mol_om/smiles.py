"""리간드 SMILES — **우리 결합차수·전하를 RDKit 이 바꾸지 못하게** 고정해서 만든다.

⚠️ 이 모듈의 존재 이유가 그것 하나다. 순진하게 `RWMol` 을 만들어 `SanitizeMol` 을 부르면
RDKit 이 **배위 원자에 암묵적 수소를 붙이고 형식전하를 자기 규칙대로 다시 매긴다** —
`RO⁻`·`R₂N⁻`·`Cp⁻` 처럼 **금속에 전자쌍을 준 자리**가 우리 출력에서는 음이온인데
RDKit 눈에는 "수소가 모자란 중성 원자" 로 보이기 때문이다.

그래서 원자마다 **셋을 다 명시**하고 그 뒤로 RDKit 이 손대지 못하게 잠근다:

    a.SetNoImplicit(True)          암묵적 H 를 0 으로 못박는다
    a.SetNumExplicitHs(0)          xyz 의 H 는 이미 **실제 원자**로 들어 있다
    a.SetFormalCharge(q)           우리 `q_atom` 값을 그대로 쓴다

그리고 `SanitizeMol` 은 **부분 플래그로만** 부른다 — `SANITIZE_ADJUSTHS` 와
`SANITIZE_SETAROMATICITY` 를 빼서 H 개수와 방향족화가 재계산되지 않게 한다.

반환 SMILES 는 **Kekulé 형**(`kekuleSmiles=True`)이다 — 우리 ⑥ 출력 변환기의 정수 차수를
그대로 읽을 수 있어야 하기 때문이다.
"""

# ruff: noqa: E501
from __future__ import annotations

from rdkit import Chem

_BOND = {1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE, 3: Chem.BondType.TRIPLE}


def ligand_smiles(el, atoms, bonds, charges, coord_atoms=(), with_map=True):
    """리간드 조각 하나 → (SMILES, 원자 대응). 실패하면 `(None, {})`.

    `el`          전체 원소 리스트
    `atoms`       이 조각의 원자 인덱스 (전체 기준)
    `bonds`       {(i, j): 1 | 2 | 3}  — **Kekulé 정수 차수** (⑥ 출력 변환기 산출)
    `charges`     {i: q}               — 우리 `q_atom` 형식전하 (정수)
    `coord_atoms` 금속에 배위한 원자 인덱스 — SMILES 원자 맵 번호로 표시한다
    `with_map`    배위 원자에 `[C:1]` 식 맵 번호를 붙일지

    ⚠️ **원자 순서·형식전하·수소 개수를 전부 명시하고 잠근다** — 모듈 docstring 참조.
    """
    idx = {a: k for k, a in enumerate(sorted(atoms))}
    m = Chem.RWMol()
    for a in sorted(atoms):
        at = Chem.Atom(el[a])
        at.SetNoImplicit(True)  # 🔴 암묵적 H 금지 — xyz 의 H 는 실제 원자로 들어 있다
        at.SetNumExplicitHs(0)
        at.SetFormalCharge(int(round(charges.get(a, 0))))
        if with_map and a in set(coord_atoms):
            at.SetAtomMapNum(1 + sorted(coord_atoms).index(a))
        m.AddAtom(at)
    for (i, j), o in bonds.items():
        if i in idx and j in idx:
            m.AddBond(idx[i], idx[j], _BOND.get(int(round(o)), Chem.BondType.SINGLE))
    mol = m.GetMol()
    try:
        # 🔴 부분 sanitize — H 재계산(ADJUSTHS)·방향족화(SETAROMATICITY)를 **끈다**
        Chem.SanitizeMol(
            mol,
            sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL
            ^ Chem.SanitizeFlags.SANITIZE_ADJUSTHS
            ^ Chem.SanitizeFlags.SANITIZE_SETAROMATICITY,
            catchErrors=True,
        )
        smi = Chem.MolToSmiles(mol, kekuleSmiles=True, canonical=True)
    except Exception:
        return None, {}
    if not smi:
        return None, {}
    return smi, {a: k for a, k in idx.items()}


def verify_roundtrip(smi, el, atoms, bonds, charges):
    """SMILES 를 다시 읽어 **결합차수·형식전하가 보존됐는지** 확인한다.

    반환 `(ok, 이유)`. 원자 대응은 정렬 순서로 맞춘다(`ligand_smiles` 와 같은 순서).
    ⚠️ SMILES 는 정준화되므로 원자 순서가 바뀐다 — **조성·차수 다중집합·전하 합**으로 본다.
    """
    m = Chem.MolFromSmiles(smi, sanitize=False)
    if m is None:
        return False, "파싱 실패"
    try:
        Chem.SanitizeMol(
            m,
            sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL
            ^ Chem.SanitizeFlags.SANITIZE_ADJUSTHS
            ^ Chem.SanitizeFlags.SANITIZE_SETAROMATICITY,
            catchErrors=True,
        )
    except Exception:
        return False, "sanitize 실패"
    want_el = sorted(el[a] for a in atoms)
    got_el = sorted(a.GetSymbol() for a in m.GetAtoms())
    if want_el != got_el:
        return False, f"조성 불일치 {len(want_el)} vs {len(got_el)}"
    wq = sum(int(round(charges.get(a, 0))) for a in atoms)
    gq = sum(a.GetFormalCharge() for a in m.GetAtoms())
    if wq != gq:
        return False, f"전하 합 {wq} → {gq}"
    wb = sorted(int(round(o)) for (i, j), o in bonds.items() if i in set(atoms) and j in set(atoms))
    gb = sorted(int(b.GetBondTypeAsDouble()) for b in m.GetBonds())
    if wb != gb:
        return False, f"차수 다중집합 {wb[:6]}… vs {gb[:6]}…"
    if any(a.GetNumImplicitHs() for a in m.GetAtoms()):
        return False, "RDKit 이 암묵적 H 를 붙였다"
    # 🔴 화학적 타당성 — 2중심 형식으로 표현 불가능한 자리를 잡는다 (2026-09-03).
    #   `H` 가 결합 2개 이상이면 **다리 H(3c2e)** 다. 설계도 §3 5c 에서 원자가 채점에서 빼는
    #   자리이고, SMILES(2중심 형식)로는 옳게 적을 수 없다 — `[H+2]` 같은 것이 나온다.
    for a in m.GetAtoms():
        if a.GetSymbol() == "H" and a.GetDegree() > 1:
            return False, f"다리 H(3c2e) — 2중심 형식으로 표현 불가 (deg {a.GetDegree()})"
        if abs(a.GetFormalCharge()) > 3:  # 나이트라이도 N³⁻ · 카바이드 C³⁻ 는 정상이다
            return False, f"형식전하 |q| > 3 ({a.GetSymbol()} {a.GetFormalCharge():+d})"
    return True, ""

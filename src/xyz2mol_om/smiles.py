"""SMILES — **우리 결합차수·전하를 RDKit 이 바꾸지 못하게** 고정해서 만든다.

리간드 하나(`ligand_smiles`)와 **착물 전체**(`complex_smiles`) 둘 다 낸다. 착물 쪽은 M–L 을
전부 dative 화살표로 적고 금속에 **산화수**를 형식전하로 박는다 — 아래 `complex_smiles` 참조.

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

# 🔴 부분 sanitize — H 재계산(ADJUSTHS)·방향족화(SETAROMATICITY)를 **끈다** (모듈 docstring)
_SANI = (
    Chem.SanitizeFlags.SANITIZE_ALL
    ^ Chem.SanitizeFlags.SANITIZE_ADJUSTHS
    ^ Chem.SanitizeFlags.SANITIZE_SETAROMATICITY
)


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
        Chem.SanitizeMol(mol, sanitizeOps=_SANI, catchErrors=True)
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
        Chem.SanitizeMol(m, sanitizeOps=_SANI, catchErrors=True)
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


# ══════════════════════════════════════════════════════════════════════════════
# ★ complex SMILES — 금속을 포함한 **착물 전체** (2026-09-03 오너 요청)
#   M–L 은 **전부 dative 화살표**(`->`)로 적는다. 결합차수는 여기서 뭉개지고(모든 M–L 이
#   화살표 1개), 실제 차수는 `ml_bonds[(m,x)]["order"]` 로 남는다 — 오너 결정.
#   🔴 **화살표는 연결성 표시일 뿐이다.** 그것 때문에 형식전하가 달라지면 안 된다(오너
#      2026-09-03) — 원자 전하는 전부 우리 `q_atom`·산화수를 그대로 박고, `verify_complex` 가
#      **(원소, 전하) 다중집합**으로 그것을 강제한다. RDKit 이 다시 매기지 못하게 잠근다.
#   왜 dative 인가: RDKit 의 `BondType.DATIVE` 는 **도너 쪽 원자가에 안 세인다**(실측 —
#   `N` 이 σ 3개 + dative 1개인데 전하 0으로 sanitize 통과). 우리 `q_atom` 이 이미 전자쌍
#   기부를 전하로 반영해 놨으므로, M–L 을 보통 결합으로 적으면 도너가 이중으로 세어진다.
#   화살표 방향은 **리간드 → 금속**(도너 → 억셉터)이다.
# ══════════════════════════════════════════════════════════════════════════════


def complex_smiles(el, atoms, bonds, charges, ml_pairs, mm_bonds=(), with_map=False):
    """착물 전체 → `(SMILES, 출력 원자 순서)`. 실패하면 `(None, [])`.

    `el`        전체 원소 리스트
    `atoms`     포함할 원자 인덱스 전량 (금속 + 리간드)
    `bonds`     {(i, j): 1|2|3}  — **리간드 내부** Kekulé 정수 차수
    `charges`   {i: q}           — 리간드 원자는 `q_atom` 형식전하 · **금속은 산화수**
    `ml_pairs`  [(m, x)]         — M–L 결합. 전부 `x -> m` dative 로 적는다
    `mm_bonds`  {(m1, m2): 1|2|3} — M–M 결합 (dative 가 아니라 보통 결합)
    `with_map`  원자마다 `[X:i+1]` 맵 번호(= 입력 원자 인덱스 + 1)를 붙일지

    반환 두 번째는 **SMILES 출력 순서대로 나열한 입력 원자 인덱스**다 — 정준화로 순서가
    바뀌므로 이것이 있어야 SMILES 원자와 xyz 원자를 맞출 수 있다.
    """
    order = sorted(atoms)
    idx = {a: k for k, a in enumerate(order)}
    m = Chem.RWMol()
    for a in order:
        at = Chem.Atom(el[a])
        at.SetNoImplicit(True)  # 🔴 암묵적 H 금지 — xyz 의 H 는 실제 원자로 들어 있다
        at.SetNumExplicitHs(0)
        at.SetFormalCharge(int(round(charges.get(a, 0))))
        if with_map:
            at.SetAtomMapNum(a + 1)
        m.AddAtom(at)
    for (i, j), o in bonds.items():
        if i in idx and j in idx:
            m.AddBond(idx[i], idx[j], _BOND.get(int(round(o)), Chem.BondType.SINGLE))
    for (m1, m2), o in dict(mm_bonds).items():
        if m1 in idx and m2 in idx:
            m.AddBond(idx[m1], idx[m2], _BOND.get(int(round(o)), Chem.BondType.SINGLE))
    seen = set()
    for mm_, x in ml_pairs:
        if mm_ not in idx or x not in idx or (mm_, x) in seen:
            continue
        seen.add((mm_, x))
        m.AddBond(idx[x], idx[mm_], Chem.BondType.DATIVE)  # 리간드 → 금속
    mol = m.GetMol()
    try:
        Chem.SanitizeMol(mol, sanitizeOps=_SANI, catchErrors=True)
        smi = Chem.MolToSmiles(mol, kekuleSmiles=True, canonical=True)
    except Exception:
        return None, []
    if not smi:
        return None, []
    # `_smilesAtomOutputOrder` 는 `[3,2,4,…]` 또는 `[3,2,4,…,]` 로 나온다 (RDKit 판마다 다르다).
    try:
        raw = mol.GetProp("_smilesAtomOutputOrder").strip().strip("[]")
        out = [order[int(t)] for t in raw.split(",") if t.strip()]
    except Exception:
        out = []
    return smi, out


def verify_complex(smi, el, atoms, bonds, charges, ml_pairs, mm_bonds=(), total_charge=None):
    """complex SMILES 왕복 검증. 반환 `(ok, 이유)`.

    보는 것 — **(원소, 형식전하) 다중집합 · 보통결합 차수 다중집합 · dative 개수 · 암묵적 H 0**.
    🔴 전하는 **합이 아니라 원자별로** 본다 — dative 화살표는 연결성 표시일 뿐이고 우리가 박은
       형식전하를 한 톨도 바꾸면 안 된다(오너 2026-09-03). 합만 보면 전하가 다른 원자로
       옮겨간 것을 놓친다.
    `total_charge` 를 주면 **전하 합이 그것과 같은지**까지 본다 (⚠️ 짝수고리 다이아니온처럼
    골격으로 표현 안 되는 잔여 전하가 있으면 여기서 걸린다 — `residual_charge` 참조).
    """
    m = Chem.MolFromSmiles(smi, sanitize=False)
    if m is None:
        return False, "파싱 실패"
    try:
        Chem.SanitizeMol(m, sanitizeOps=_SANI, catchErrors=True)
    except Exception:
        return False, "sanitize 실패"
    # 🔴 **화살표는 연결성 표시일 뿐이다 — 전하를 한 톨도 바꾸면 안 된다** (오너 2026-09-03).
    #   그래서 합이 아니라 **(원소, 형식전하) 다중집합**으로 본다. 합만 보면 `+1` 이 다른
    #   원자로 옮겨간 것을 못 잡는다. 이 검사가 조성 검사도 겸한다.
    want_q = sorted((el[a], int(round(charges.get(a, 0)))) for a in atoms)
    got_q = sorted((a.GetSymbol(), a.GetFormalCharge()) for a in m.GetAtoms())
    if want_q != got_q:
        bad = sorted(set(want_q) ^ set(got_q))[:4]
        return False, f"(원소, 전하) 다중집합 불일치 — 차이 {bad}"
    gq = sum(q for _e, q in got_q)
    if total_charge is not None and gq != int(total_charge):
        return False, f"전하 합이 총전하와 다르다 {gq} vs {int(total_charge)} (잔여 조각 전하)"
    aset = set(atoms)
    wb = sorted(
        [int(round(o)) for (i, j), o in bonds.items() if i in aset and j in aset]
        + [int(round(o)) for (i, j), o in dict(mm_bonds).items() if i in aset and j in aset]
    )
    gb = sorted(
        int(b.GetBondTypeAsDouble())
        for b in m.GetBonds()
        if b.GetBondType() != Chem.BondType.DATIVE
    )
    if wb != gb:
        return False, f"차수 다중집합 {wb[:6]}… vs {gb[:6]}…"
    n_dat = sum(1 for b in m.GetBonds() if b.GetBondType() == Chem.BondType.DATIVE)
    want_dat = len({(mm_, x) for mm_, x in ml_pairs if mm_ in aset and x in aset})
    if n_dat != want_dat:
        return False, f"dative 개수 {want_dat} → {n_dat}"
    if any(a.GetNumImplicitHs() for a in m.GetAtoms()):
        return False, "RDKit 이 암묵적 H 를 붙였다"
    return True, ""

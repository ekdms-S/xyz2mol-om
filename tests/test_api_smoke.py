"""공개 API 스모크 — 워크스페이스 없이도 도는 최소 검증 (배포본에서 이것만으로 확인 가능).

`[Mo(≡N)(OH)Cl₃]⁻` 하나로 파이프라인 전 단계를 지난다:
  T1 내부 결합(O–H) · T4 M–L 5개 · T8 `Mo≡N` = Triple · T10 `q_L` 과 `OS(Mo) = +6` ·
  ⑥ Kekulé 출력 · 리간드 SMILES 왕복 검증.
Mayer 결합차수는 xtb `--sp --wbo` 산출값을 **상수로 박아** 두었다 (xtb 없이 돌게 하려고).
"""

# ruff: noqa: E501
from __future__ import annotations

import numpy as np

from xyz2mol_om import predict

EL = ["Mo", "N", "O", "H", "Cl", "Cl", "Cl"]
XYZ = np.array(
    [
        [0.00, 0.00, 0.00],
        [0.00, 0.00, 1.66],
        [1.95, 0.00, -0.25],
        [2.52, 0.76, -0.45],
        [0.00, 2.35, -0.25],
        [-2.35, 0.00, -0.25],
        [0.00, -2.35, -0.25],
    ]
)
# xtb GFN2 `--sp --wbo` 실측 (2026-09-03)
WBO = {(0, 1): 2.891, (0, 2): 1.022, (0, 4): 0.955, (0, 5): 0.967, (0, 6): 0.997, (2, 3): 0.862}
WBO.update({(j, i): w for (i, j), w in list(WBO.items())})


def test_mo_nitrido():
    r = predict(EL, XYZ, total_charge=-1, wbo=WBO)
    assert len(r["metals"]) == 1
    m = r["metals"][0]
    assert m["element"] == "Mo"
    assert m["oxidation"] == 6, f"OS(Mo) 는 +6 이어야 한다 — 얻은 값 {m['oxidation']}"

    ligs = {tuple(L["atoms"]): L for L in r["ligands"]}
    assert len(ligs) == 5, f"리간드 5개여야 한다 — {len(ligs)}"

    nit = ligs[(1,)]
    assert nit["charge"] == -3, f"나이트라이도 N 은 −3 — {nit['charge']}"
    assert nit["ml_bonds"][(0, 1)]["order"] == 3, "Mo≡N 은 Triple 이어야 한다"
    assert nit["ml_bonds"][(0, 1)]["type"] == "sigma"

    oh = ligs[(2, 3)]
    assert oh["charge"] == -1
    assert oh["bonds_kekule"][(2, 3)] == 1
    assert oh["bonds_4class"][(2, 3)] == "Single"

    for a in (4, 5, 6):
        assert ligs[(a,)]["charge"] == -1

    for L in r["ligands"]:
        assert L["smiles"], "SMILES 가 나와야 한다"
        assert L["smiles_ok"], f"SMILES 왕복 검증 실패: {L['smiles']} — {L['smiles_note']}"


def test_no_spurious_hh():
    """🔴 회귀 — 메틸의 geminal H–H(≈1.77 Å)가 결합으로 잡히면 안 된다.

    `d_int` 에 `H,H` 항목이 없던 시절 폴백 2.0542 Å 가 걸려 **예측 결합의 22.9%가 허위 H–H** 였다.
    """
    el = ["C", "H", "H", "H", "H"]
    d = 1.09 / np.sqrt(3)
    xyz = np.array([[0, 0, 0], [d, d, d], [d, -d, -d], [-d, d, -d], [-d, -d, d]], dtype=float)
    r = predict(el, xyz, total_charge=0)
    (L,) = r["ligands"]
    hh = [(i, j) for (i, j) in L["bonds_kekule"] if el[i] == "H" and el[j] == "H"]
    assert not hh, f"허위 H–H 결합이 생겼다: {hh}"
    assert len(L["bonds_kekule"]) == 4, f"CH4 는 결합 4개 — {L['bonds_kekule']}"


if __name__ == "__main__":
    test_mo_nitrido()
    test_no_spurious_hh()
    print("API SMOKE PASS")

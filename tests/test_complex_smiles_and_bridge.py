"""complex SMILES · bridge 태그 · 3c2e 예산 제외 — 워크스페이스 없이 도는 검증.

두 계로 본다:
  ① `[Mo(≡N)(OH)Cl₃]⁻` — dative 화살표 · 금속 산화수 표기 · 총전하 보존 (M–L 단순 σ)
  ② `[H₂B(μ-H)₂Mo]` 꼴 μ-H 다리 — `bridge`/`3c2e` 태깅과 예산 제외
"""

# ruff: noqa: E501
from __future__ import annotations

import numpy as np

from xyz2mol_om import predict
from xyz2mol_om.pipeline import bridge_tags

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
WBO = {(0, 1): 2.891, (0, 2): 1.022, (0, 4): 0.955, (0, 5): 0.967, (0, 6): 0.997, (2, 3): 0.862}
WBO.update({(j, i): w for (i, j), w in list(WBO.items())})


def test_complex_smiles_dative_and_oxidation():
    r = predict(EL, XYZ, total_charge=-1, wbo=WBO)
    smi = r["complex_smiles"]
    assert smi, f"complex SMILES 가 없다 — {r['complex_smiles_note']}"
    assert r["complex_smiles_ok"], f"왕복 검증 실패 — {r['complex_smiles_note']}"
    # M–L 5개가 전부 dative 화살표다
    assert smi.count("->") + smi.count("<-") == 5, smi
    # 금속에 **산화수**가 박힌다
    assert "[Mo+6]" in smi, smi
    # 음이온 리간드의 형식전하도 보인다
    assert "[N-3]" in smi and "[Cl-]" in smi, smi
    # 출력 순서로 xyz 원자를 되짚을 수 있다
    assert sorted(r["complex_atom_order"]) == list(range(len(EL)))
    assert [EL[i] for i in r["complex_atom_order"]].count("Cl") == 3


def test_complex_smiles_needs_total_charge():
    """`total_charge` 가 없으면 산화수가 없으므로 **만들지 않는다** (조용히 틀리지 않는다)."""
    r = predict(EL, XYZ, wbo=WBO)
    assert r["complex_smiles"] is None
    assert "산화수" in r["complex_smiles_note"]


def test_ml_bonds_have_bridge_key():
    r = predict(EL, XYZ, total_charge=-1, wbo=WBO)
    for lig in r["ligands"]:
        for _e, d in lig["ml_bonds"].items():
            assert set(d) == {"type", "order", "bridge"}, d
            assert d["type"] in ("sigma", "haptic", "bridge")
            assert d["bridge"] in (None, "3c2e", "dative")
    # 이 계는 전부 말단 배위라 다리가 없다
    assert all(
        d["bridge"] is None for lig in r["ligands"] for d in lig["ml_bonds"].values()
    )


def test_bridge_tags_rule():
    """T7 판정식 — 설계도 §3.0 5c 의 실물 4경우."""
    import networkx as nx

    # μ-H : 내부 결합 0 · M–L 2  ⇒ n_center 2 · deg 2 > VALENCE_3C[H]=1  ⇒ 3c2e
    el = ["Fe", "Fe", "H"]
    G = nx.Graph()
    G.add_nodes_from([2])
    assert bridge_tags(el, G, [(0, 2), (1, 2)]) == {2: "3c2e"}

    # μ-Cl : n_center 2 인데 Cl 은 VALENCE_3C 에 없다  ⇒ dative (3c4e)
    el = ["Fe", "Fe", "Cl"]
    assert bridge_tags(el, G, [(0, 2), (1, 2)]) == {2: "dative"}

    # 말단 Cl : n_center 1  ⇒ 태그 없음
    assert bridge_tags(el, G, [(0, 2)]) == {}

    # B–H···M : 내부 이웃 B 1개 + M–L 1개 ⇒ n_center 2 · deg 2 > 1 ⇒ 3c2e
    el = ["Fe", "B", "H"]
    G2 = nx.Graph()
    G2.add_edge(1, 2)
    assert bridge_tags(el, G2, [(0, 2)]) == {2: "3c2e"}


def test_bridge_type_precedence():
    """`type` 은 haptic > bridge > sigma 이고, `bridge` 칸은 겹쳐도 남는다."""
    import networkx as nx

    el = ["Fe", "Fe", "Cl"]
    G = nx.Graph()
    G.add_nodes_from([2])
    assert bridge_tags(el, G, [(0, 2), (1, 2)])[2] == "dative"


if __name__ == "__main__":
    test_complex_smiles_dative_and_oxidation()
    test_complex_smiles_needs_total_charge()
    test_ml_bonds_have_bridge_key()
    test_bridge_tags_rule()
    test_bridge_type_precedence()
    print("COMPLEX SMILES / BRIDGE PASS")

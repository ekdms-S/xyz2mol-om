"""🔴 회귀 — **R7 이 실제로 발동한다**: η⁵-싸이오펜(thienyl) 고리의 S 가 하프틱에 들어간다.

실물 `HOQNOQ` — CSD `chemical_name` = *"(μ2-Diphenylphosphido)-(μ2-**η5**-thien-2-yl)-
hexacarbonyl-di-manganese"* ⇒ **정답 η⁵** 이다. 기하는 `ref_xtb2` · Mayer 결합차수는 같은
구조의 xtb GFN2 `--sp --wbo` 실측값을 **상수로 박아** 두었다 (워크스페이스 없이 돌게 하려고).

무엇을 재나 (`docs/PIPELINE.md` 5′):

  R7 off :  하프틱 = {Mn–C 4개}                 η⁴   ← S 가 π 조각에 없어서 떨어진다
  R7 on  :  하프틱 = {Mn–C 4개, **Mn–S**}       η⁵   ← 정답

왜 S 가 π 조각에서 빠지나: R2 가 `S`(deg 2)의 두 결합에서 `Conj` 를 지우므로 5원 고리에
`Conj` 가 3개만 남고 **S 만 `Double`·`Triple`·`Conj` 어디에도 안 닿는다.**

🔴 **T3 는 한 비트도 안 바뀐다** — 아래 `bonds_4class` 동일성 검사가 그것을 고정한다.
"""

# ruff: noqa: E501
from __future__ import annotations

import numpy as np

import xyz2mol_om.api as api
from xyz2mol_om import predict

# `HOQNOQ` (CSD 유래 · `ref_xtb2` 기하)
EL = [
    "Mn", "Mn", "P", "S", "C", "C", "H", "C", "H", "C", "H", "C", "C", "H", "C", "H", "C",
    "H", "C", "H", "C", "H", "C", "C", "H", "C", "H", "C", "H", "C", "H", "C", "H", "C",
    "O", "C", "O", "C", "O", "C", "O", "C", "O", "O", "C"
]
XYZ = np.array(
    [
        [   5.0977,    0.2869,    1.7015],
        [   5.0006,    3.8564,    1.0655],
        [   4.0454,    1.8077,    0.3411],
        [   4.4852,    1.5687,    3.5050],
        [   5.6048,    2.3238,    2.3664],
        [   6.7457,    1.5389,    2.3113],
        [   7.6350,    1.8364,    1.7674],
        [   6.7153,    0.3510,    3.0896],
        [   7.5685,   -0.3004,    3.2275],
        [   5.5135,    0.1909,    3.7524],
        [   5.2726,   -0.5212,    4.5271],
        [   4.1499,    1.3970,   -1.4278],
        [   4.3863,    2.3293,   -2.4298],
        [   4.5715,    3.3673,   -2.1822],
        [   4.3956,    1.9266,   -3.7548],
        [   4.5875,    2.6517,   -4.5319],
        [   4.1654,    0.6016,   -4.0802],
        [   4.1808,    0.2916,   -5.1144],
        [   3.9207,   -0.3297,   -3.0810],
        [   3.7430,   -1.3633,   -3.3362],
        [   3.9121,    0.0619,   -1.7572],
        [   3.7365,   -0.6502,   -0.9512],
        [   2.2328,    1.8634,    0.5012],
        [   1.5405,    2.7402,   -0.3342],
        [   2.0848,    3.3212,   -1.0685],
        [   0.1705,    2.8632,   -0.2211],
        [  -0.3627,    3.5440,   -0.8682],
        [  -0.5181,    2.1197,    0.7260],
        [  -1.5890,    2.2235,    0.8174],
        [   0.1662,    1.2469,    1.5541],
        [  -0.3693,    0.6656,    2.2901],
        [   1.5397,    1.1168,    1.4461],
        [   2.0757,    0.4293,    2.0887],
        [   6.0435,   -0.4592,    0.3935],
        [   6.7185,   -0.8973,   -0.4304],
        [   3.9763,   -1.0932,    1.6335],
        [   3.2814,   -2.0175,    1.6109],
        [   4.2978,    4.9785,   -0.1326],
        [   3.8078,    5.7060,   -0.8708],
        [   5.9961,    5.0490,    1.9621],
        [   6.6505,    5.7674,    2.5729],
        [   6.2968,    3.3136,   -0.0465],
        [   7.0744,    2.8816,   -0.7637],
        [   2.5552,    4.1882,    2.6460],
        [   3.5281,    4.0809,    2.0536],
    ]
)
# xtb GFN2 `--sp --wbo` 실측 — M–L 쌍만 (`predict` 는 M–L·M–M 에서만 쓴다)
WBO = {
    (0, 1): 0.0, (0, 2): 0.9164, (0, 3): 0.322, (0, 4): 0.1982, (0, 5): 0.196, (0, 6): 0.0,
    (0, 7): 0.2669, (0, 8): 0.0, (0, 9): 0.3179, (0, 10): 0.0, (0, 11): 0.0, (0, 20): 0.0,
    (0, 22): 0.0, (0, 31): 0.0, (0, 32): 0.0, (0, 33): 1.4583, (0, 34): 0.3575, (0, 35):
    1.5298, (0, 36): 0.3796, (0, 41): 0.0, (1, 0): 0.0, (1, 2): 0.743, (1, 3): 0.0, (1, 4):
    0.7143, (1, 5): 0.0, (1, 11): 0.0, (1, 12): 0.0, (1, 22): 0.0, (1, 37): 1.2449, (1, 38):
    0.2646, (1, 39): 1.2972, (1, 40): 0.2691, (1, 41): 1.1211, (1, 42): 0.2115, (1, 43):
    0.2228, (1, 44): 1.1469
}
WBO.update({(j, i): w for (i, j), w in list(WBO.items())})

MN = 0          # 싸이엔일이 하프틱으로 붙은 망간
S_IDX = 3       # 고리의 S — R7 이 되돌리는 원자
RING_C = (4, 5, 7, 9)


def _thienyl(r):
    (L,) = [x for x in r["ligands"] if S_IDX in x["atoms"]]
    return L


def _run(r7):
    old = api.R7RING
    api.R7RING = r7
    try:
        return predict(EL, XYZ, total_charge=None, wbo=WBO)
    finally:
        api.R7RING = old


def test_r7_recovers_eta5_thiophene():
    """R7 on(배포 기본값) 이면 η⁵ · off 면 η⁴ 다."""
    on, off = _thienyl(_run(True)), _thienyl(_run(False))

    hap_on = sorted(k for k, v in on["ml_bonds"].items() if v["type"] == "haptic")
    hap_off = sorted(k for k, v in off["ml_bonds"].items() if v["type"] == "haptic")
    assert hap_off == [(MN, c) for c in RING_C], f"R7 off 하프틱이 예상과 다르다 — {hap_off}"
    assert hap_on == sorted([(MN, S_IDX)] + [(MN, c) for c in RING_C]), (
        f"R7 on 인데 Mn–S 가 하프틱이 아니다 — {hap_on}"
    )
    assert off["eta"] == {MN: 4}, f"R7 off η 는 4 — {off['eta']}"
    assert on["eta"] == {MN: 5}, f"R7 on η 는 5 (CSD 정답 η⁵) — {on['eta']}"


def test_r7_does_not_touch_bond_orders():
    """🔴 R7 은 **T5 의 π 조각 소속 조건만** 면제한다 — T3 4클래스는 그대로여야 한다."""
    on, off = _run(True), _run(False)
    for a, b in zip(on["ligands"], off["ligands"]):
        assert a["bonds_4class"] == b["bonds_4class"], f"R7 이 T3 를 바꿨다 — 조각 {a['index']}"
        assert a["bonds_kekule"] == b["bonds_kekule"], f"R7 이 Kekulé 를 바꿨다 — 조각 {a['index']}"
        assert a["charge"] == b["charge"], f"R7 이 리간드 전하를 바꿨다 — 조각 {a['index']}"


if __name__ == "__main__":
    test_r7_recovers_eta5_thiophene()
    test_r7_does_not_touch_bond_orders()
    print("R7 PASS")

"""🔴 One unpaired electron (`n_unpaired=1`).

`q_atom` already points at the right atom; what it got wrong was the pricing — an unpaired
electron read as a lone pair, so `CH3•` came out `CH3-`. With a metal present the invented `-1`
is cancelled by a `+1` on the metal, so the total charge stayed right while the oxidation state
did not, and no total-based check could see it.

The cases below are the placements and the two refusals.
"""

# ruff: noqa: E501
from __future__ import annotations

import numpy as np
import pytest

from xyz2mol_om import predict

CH3 = [[0, 0, 0], [1.08, 0, 0], [-0.54, 0.93, 0], [-0.54, -0.93, 0]]


def _cu_ch3():
    """Cu(I)Cl and a methyl radical 12 Å apart — the shape an IRC endpoint takes."""
    el = ["Cu", "Cl", "C", "H", "H", "H"]
    xyz = [[0, 0, 0], [2.2, 0, 0]] + [[x, y, z + 12] for x, y, z in CH3]
    wbo = {(0, x): 0.0 for x in range(1, 6)}
    wbo[(0, 1)] = 0.9
    return el, np.array(xyz, float), wbo


def test_methyl_radical_is_neutral_with_an_unpaired_electron():
    r = predict(["C", "H", "H", "H"], np.array(CH3, float), total_charge=0, n_unpaired=1)
    assert r["radical"]["site"] == "organic" and r["radical"]["atom"] == 0
    (mol,) = r["molecules"]
    assert mol["charge"] == 0 and mol["smiles_ok"]
    assert "-" not in mol["smiles"]  # not the carbanion


def test_radical_on_the_ligand_fixes_the_metal_oxidation_state():
    """The reported defect: `Cu(I)Cl + CH3•` came out as Cu(II) with a `CH3-`."""
    el, xyz, wbo = _cu_ch3()
    closed = predict(el, xyz, total_charge=0, wbo=wbo, n_unpaired=0)
    assert [m["oxidation"] for mol in closed["molecules"] for m in mol["metals"]] == [2]
    assert min(mol["charge"] for mol in closed["molecules"]) == -1

    r = predict(el, xyz, total_charge=0, wbo=wbo, n_unpaired=1)
    assert [m["oxidation"] for mol in r["molecules"] for m in mol["metals"]] == [1]
    assert all(mol["charge"] == 0 for mol in r["molecules"])
    assert r["radical"]["site"] == "organic"


def test_radical_on_the_metal_changes_nothing():
    """`Cu(II)Cl2 + CH4`: no negative site outside the complex, so the oxidation state keeps it."""
    el = ["Cu", "Cl", "Cl", "C", "H", "H", "H", "H"]
    xyz = np.array([[0, 0, 0], [2.2, 0, 0], [-2.2, 0, 0], [0, 0, 12],
                    [0.63, 0.63, 12.63], [-0.63, -0.63, 12.63],
                    [0.63, -0.63, 11.37], [-0.63, 0.63, 11.37]], float)  # fmt: skip
    wbo = {(0, x): 0.0 for x in range(1, 8)}
    wbo[(0, 1)] = wbo[(0, 2)] = 0.9
    open_, closed = (predict(el, xyz, total_charge=0, wbo=wbo, n_unpaired=n) for n in (1, 0))
    assert open_["radical"]["site"] == "metal" and open_["radical"]["atom"] is None
    for a, b in zip(open_["molecules"], closed["molecules"]):
        assert (a["charge"], a["smiles"]) == (b["charge"], b["smiles"])


def test_two_candidate_sites_are_refused():
    """A free chloride beside the radical — the graph cannot say which one is which."""
    el = ["Cu", "Cl", "C", "H", "H", "H", "Cl"]
    xyz = np.array([[0, 0, 0], [2.2, 0, 0]] + [[x, y, z + 12] for x, y, z in CH3]
                   + [[0, 0, 25]], float)
    wbo = {(0, x): 0.0 for x in range(1, 7)}
    wbo[(0, 1)] = 0.9
    r = predict(el, xyz, total_charge=-1, wbo=wbo, n_unpaired=1)
    assert r["radical"]["atom"] is None and "candidate sites" in r["radical"]["note"]
    # the closed-shell answer comes back untouched, so a caller can drop it on the note
    closed = predict(el, xyz, total_charge=-1, wbo=wbo, n_unpaired=0)
    assert [m["charge"] for m in r["molecules"]] == [m["charge"] for m in closed["molecules"]]


def test_more_than_one_unpaired_electron_is_refused():
    """Diradicals are out of scope — two electrons on different atoms cannot be placed."""
    with pytest.raises(ValueError, match="n_unpaired"):
        predict(["C", "H", "H", "H"], np.array(CH3, float), total_charge=0, n_unpaired=2)


def test_a_structural_negative_charge_is_not_a_radical_site():
    """A borate's `-1` is required by its valence, not a mispriced unpaired electron.

    Neutralising a four-bond boron leaves `3 - 4 = -1` places for the electron, i.e. a neutral
    B with four bonds, which RDKit rejects outright.
    """
    el = ["Cu", "Cl", "B", "O", "H", "O", "H", "O", "H", "C", "H", "H", "H"]
    xyz = np.array([[0, 0, 0], [2.2, 0, 0],
                    [0, 0, 12], [1.5, 0, 12], [2.0, 0.8, 12],
                    [-0.75, 1.3, 12], [-1.3, 1.8, 12.6],
                    [-0.75, -1.3, 12], [-1.3, -1.8, 12.6],
                    [0, 0, 13.6], [0.9, 0, 14.2], [-0.5, 0.85, 14.2], [-0.5, -0.85, 14.2]],
                   float)  # fmt: skip
    wbo = {(0, x): 0.0 for x in range(1, 13)}
    wbo[(0, 1)] = 0.9
    r = predict(el, xyz, total_charge=-1, wbo=wbo, n_unpaired=1)
    assert r["radical"]["site"] == "metal" and r["radical"]["atom"] is None
    borate = [m for m in r["molecules"] if not m["metals"]][0]
    assert borate["charge"] == -1 and borate["smiles_ok"]


# ── the metal-free reports (UniTS-Lib batch, 2026-09-10) ────────────────────────────────────
#   Six of ten recognition failures were metal-free doublets where **every** formal negative
#   charge became a candidate site. Two different causes hide behind that one symptom, and the
#   pipeline now separates them instead of reporting "N candidate sites".

def _nitromethane():
    el = ["C", "N", "O", "O", "H", "H", "H"]
    xyz = [[0, 0, 0], [0, 0, 1.489], [0, 1.067, 2.093], [0, -1.067, 2.093],
           [1.03, 0, -0.36], [-0.515, 0.892, -0.36], [-0.515, -0.892, -0.36]]  # fmt: skip
    return el, np.array(xyz, float)


def _ozone():
    """🔴 **결함을 픽스처로 쓴다.** 오존의 옳은 Lewis 구조는 `[O-][O+]=O` 인데 파이프라인은
    `[O-]O[O-]` 를 내고, 그래서 `total_charge=0` 에 대해 shortfall 2 가 난다. 여기서 필요한 것은
    «부풀어 오른 금속 없는 구조» 하나뿐이라 이것을 쓴다 — 오존이 고쳐지면 이 픽스처를 옮겨야
    한다. (2026-09-12 이전에는 나이트로메탄이 이 자리였고, 제미널 규칙이 그것을 고쳤다.)"""
    el = ["O", "O", "O"]
    xyz = [[0, 0, 0], [1.09, 0.66, 0], [-1.09, 0.66, 0]]  # fmt: skip
    return el, np.array(xyz, float)


def test_a_charge_inflated_structure_is_reported_as_such_not_as_a_radical_site():
    """부풀어 오른 전하는 **라디칼 자리가 아니다.** 금속이 없으면 산화수가 차이를 흡수해 주지
    않으므로, 그 shortfall 은 결합차수 오류이고 그렇게 말하는 것이 쓸모 있는 답이다. 예전에는
    형식전하마다 라디칼 후보가 되어 "후보 2 개" 라고 답했다."""
    el, xyz = _ozone()
    r = predict(el, xyz, total_charge=0, n_unpaired=1)
    assert r["radical"]["atom"] is None and r["radical"]["site"] is None
    assert "charge shortfall 2" in r["radical"]["note"]
    assert "bond-order/charge error" in r["radical"]["note"]


def test_a_geminal_anion_pair_is_raised_instead_of_inflating_the_charge():
    """★ 제미널 규칙 (2026-09-12) — 공통 이웃을 사이에 둔 같은 부호 음이온 쌍은 **두 결합을
    같이** 올려서 상쇄한다. 교대(±1)로는 원리상 안 된다: 한쪽을 올리면 다른 쪽이 내려간다.

    나이트로메탄이 그 꼴이었다. `N=O` 두 개가 `Single` 로 나와 `N([O-])[O-]`, `total_charge=0`
    에 대해 shortfall **2**. 이제 두 N–O 를 같이 올려 교과서 구조가 나온다. 같은 규칙이
    금속에 붙은 CO₂ (`[O-]-C-[O-]` ⇒ Cu 산화수 +2 부풀음)를 고치는 것이고, 여기서는 금속 없이
    그 산술만 확인한다."""
    el, xyz = _nitromethane()
    r = predict(el, xyz, total_charge=0)
    (mol,) = r["molecules"]
    assert r["charge_balance"]["shortfall"] == 0 and r["charge_balance"]["ok"]
    assert mol["charge"] == 0
    assert "[N+]" in mol["smiles"] and "[O-]" in mol["smiles"]


def test_a_structural_charge_pair_is_not_a_radical_site():
    """Diazomethane is `CH2=N+=N-`: the `N-` is the notation, not an unpaired electron. The
    emitted charges already add up to `total_charge`, so there is no shortfall to explain."""
    el = ["C", "N", "N", "H", "H"]
    xyz = np.array([[0, 0, 0], [0, 0, 1.300], [0, 0, 2.430],
                    [0.94, 0, -0.55], [-0.94, 0, -0.55]], float)  # fmt: skip
    r = predict(el, xyz, total_charge=0, n_unpaired=1)
    (mol,) = r["molecules"]
    assert mol["charge"] == 0 and "[N+]" in mol["smiles"] and "[N-]" in mol["smiles"]
    assert r["radical"]["atom"] is None
    assert "charge shortfall 0" in r["radical"]["note"]


def test_the_metal_free_shortfall_test_does_not_touch_the_metal_case():
    """With a metal the shortfall is identically 0 by construction (the oxidation state absorbs
    it), so the new test must not fire there — `Cu(I)Cl + CH3-` still gets its electron."""
    el, xyz, wbo = _cu_ch3()
    r = predict(el, xyz, total_charge=0, wbo=wbo, n_unpaired=1)
    assert r["radical"]["site"] == "organic" and r["radical"]["note"] == ""


def test_charge_balance_flags_a_charge_inflated_metal_free_structure():
    """`charge_balance` is the same test as above, exposed for `n_unpaired=0` too — the case the
    consumer hits as "Sq = -2 != 0" with no radical involved."""
    el, xyz = _ozone()
    r = predict(el, xyz, total_charge=0, n_unpaired=0)
    assert r["charge_balance"]["shortfall"] == 2 and not r["charge_balance"]["ok"]

    # a healthy metal-free molecule, and a metal-bearing one where the test cannot apply
    r2 = predict(["C", "N", "N", "H", "H"],
                 np.array([[0, 0, 0], [0, 0, 1.300], [0, 0, 2.430],
                           [0.94, 0, -0.55], [-0.94, 0, -0.55]], float),
                 total_charge=0)  # fmt: skip
    assert r2["charge_balance"]["ok"] and r2["charge_balance"]["shortfall"] == 0
    el3, xyz3, wbo3 = _cu_ch3()
    assert predict(el3, xyz3, total_charge=0, wbo=wbo3)["charge_balance"]["shortfall"] is None

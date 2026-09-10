"""🔴 Regression — π 를 옮기면 사라지는 전하는 남겨두지 않는다 (`QSHIFT`).

오너가 지목한 꼴 (`10.1021_jo802516k__05`, Gold-DIGR):

    C0⁻ – C2 = C3 – O4⁻     조각 −2 · Au **+3**      ← P 프레임
    C0 = C2 – C3 = O4       조각 0  · Au **+1**      ← R 프레임

**골격도 M–L 도 하나도 안 변했는데 산화수만 두 단계 뛴다.** 산화적 부가 없이 Au(I)→Au(III) 는
화학이 아니고, 반응 예측 모델에는 "아무 일도 없었는데 그 방향으로 반응이 일어난다" 고 가르치는
것이라 가장 유해한 부류다. 같은 결합 집합 위에 |전하| 가 더 작은 **유효한** 배치가 존재하므로
이것은 모호함이 아니라 해결 가능한 실패다.

⚠️ 게이트가 이 규칙의 절반이다. 두 음이온 자리가 **둘 다** 금속에 배위하면 그것은 잘못 놓인 π 가
아니라 진짜 이음이온 킬레이트다 — 다이싸이올렌 · 벤젠다이싸이올레이트 · 카테콜레이트 ·
아미디네이트. 게이트 없이 돌리면 holdout 에서 `Σq_L` 이 15 건 맞→틀이 된다.
"""

# ruff: noqa: E501
from __future__ import annotations

import networkx as nx

from xyz2mol_om.charge import shift_pi_to_cancel


def _chain():
    """`C0–C1=C2–O3` — 아크롤레인 골격. 양 끝이 각각 −1 이고, π 를 옮기면 둘 다 사라진다.

    H 를 명시한다. `q_atom` 은 `orders` 의 차수 합만 보므로, H 를 빼면 말단 탄소가 전부
    카바니온이 되어 픽스처가 무의미해진다.
    """
    el = ["C", "C", "C", "O", "H", "H", "H", "H"]
    bonds = [(0, 1), (1, 2), (2, 3), (0, 4), (0, 5), (1, 6), (2, 7)]
    G = nx.Graph(bonds)
    orders = dict.fromkeys(bonds, 1)
    orders[(1, 2)] = 2          # 전하가 생기는 배치: C0⁻–C1=C2–O3⁻
    return el, G, orders


def test_shifts_the_pi_and_cancels_both_charges():
    el, G, orders = _chain()
    moved = shift_pi_to_cancel(orders, el, G, {})
    assert moved, "π 를 옮겨 전하를 없앨 수 있는데 그대로 두었다"
    assert orders[(0, 1)] == 2 and orders[(1, 2)] == 1 and orders[(2, 3)] == 2


def test_a_dianionic_chelate_is_left_alone():
    # 양 끝이 **둘 다** 금속에 배위하면 진짜 이음이온 킬레이트다
    el, G, orders = _chain()
    before = dict(orders)
    assert shift_pi_to_cancel(orders, el, G, {}, coord={0, 3}) == []
    assert orders == before


def test_one_coordinating_end_still_shifts():
    # 오너가 지목한 케이스가 이쪽이다 — O 만 배위하고 C 는 배위하지 않는다
    el, G, orders = _chain()
    assert shift_pi_to_cancel(orders, el, G, {}, coord={3})
    assert orders[(0, 1)] == 2


def test_the_valence_cap_still_wins():
    # C0 가 M–L 로 이미 예산을 쓰고 있으면 π 를 더 얹을 수 없다
    el, G, orders = _chain()
    before = dict(orders)
    assert shift_pi_to_cancel(orders, el, G, {0: 1.0}) == []
    assert orders == before


def test_a_neutral_chain_is_untouched():
    el, G, orders = _chain()
    orders.update({(0, 1): 2, (1, 2): 1, (2, 3): 2})
    before = dict(orders)
    assert shift_pi_to_cancel(orders, el, G, {}) == []
    assert orders == before

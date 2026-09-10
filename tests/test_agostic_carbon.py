"""🔴 Regression — `C–H···M` 의 **탄소 쪽** M–L 도 지운다 (`AGOC`), 단 조각이 떨어지지 않을 때만.

`drop_agostic` 은 M–**H** 만 지운다. 그것은 맞다. 문제는 남는 M–C 다 — 아고스틱에서 금속은
탄소의 네 번째 결합손을 차지한 것이 아니라 **이미 있는 C–H 결합의 전자쌍을 빌려 쓴다.** 그런데
이 형식에는 `sigma`(원자가 1 소비)와 `haptic`(0 소비)뿐이라 남은 M–C 를 `sigma` 로 셀 수밖에
없고, 그 1 이 탄소를 질식시켜 방향족을 깨고 금속 산화수를 2 밀어올린다.

⚠️ **떼면 그 조각이 금속에서 완전히 떨어지는 경우에는 떼지 않는다.** 실측 30.9% 가 그 경우이고,
거기서 떼면 σ 과대평가라는 오류를 «해리했다» 는 더 나쁜 오류로 바꾼다.
"""

# ruff: noqa: E501
from __future__ import annotations

import networkx as nx
import numpy as np

from xyz2mol_om.rules.pipeline import drop_agostic_carbon


def _case(d_mh, d_mc, extra_ml=False):
    """M ··· H–C 삼각형. `extra_ml` 이면 같은 조각이 다른 원자로도 금속에 붙어 있다."""
    el = ["Ru", "C", "H", "C", "N"]
    #     0     1    2    3    4
    xyz = np.zeros((5, 3))
    xyz[2] = [d_mh, 0.0, 0.0]                       # H
    xyz[1] = [d_mh, 0.0, 1.0]                       # C — d(M,C) 를 맞춘다
    xyz[1] = xyz[2] + (xyz[1] - xyz[2]) / np.linalg.norm(xyz[1] - xyz[2]) * 1.1
    xyz[1] = xyz[1] / np.linalg.norm(xyz[1]) * d_mc
    xyz[3] = [0.0, 3.0, 0.0]
    xyz[4] = [0.0, 2.0, 0.0]
    G = nx.Graph([(1, 2), (1, 3), (3, 4)])          # C–H · C–C · C–N (한 조각)
    ml = [(0, 1)] + ([(0, 4)] if extra_ml else [])
    return el, xyz, G, ml


def test_the_carbon_side_is_dropped_when_the_fragment_stays_attached():
    el, xyz, G, ml = _case(d_mh=1.80, d_mc=2.28, extra_ml=True)
    out = drop_agostic_carbon(el, xyz, G, ml, cut=2.0)
    assert (0, 1) not in out, "아고스틱 M–C 를 남겼다"
    assert (0, 4) in out, "다른 M–L 까지 건드렸다"


def test_it_is_kept_when_dropping_would_orphan_the_fragment():
    # 그 M–C 가 유일한 연결이면 떼지 않는다 — 떼면 자유 분자가 되어 더 나쁘다
    el, xyz, G, ml = _case(d_mh=1.80, d_mc=2.28, extra_ml=False)
    assert drop_agostic_carbon(el, xyz, G, ml, cut=2.0) == ml


def test_a_plain_sigma_aryl_is_untouched():
    # H 가 금속에서 더 멀면 아고스틱이 아니다 — 진짜 M–C 결합이다
    el, xyz, G, ml = _case(d_mh=2.60, d_mc=2.05, extra_ml=True)
    assert drop_agostic_carbon(el, xyz, G, ml, cut=2.0) == ml


def test_a_long_contact_is_untouched():
    # H 가 더 가깝긴 하지만 절대적으로 멀면 아고스틱이라 부르지 않는다 (2.38 대 2.44 같은 간발의 차)
    el, xyz, G, ml = _case(d_mh=2.38, d_mc=2.44, extra_ml=True)
    assert drop_agostic_carbon(el, xyz, G, ml, cut=2.0) == ml


def test_off_by_default_switch():
    el, xyz, G, ml = _case(d_mh=1.80, d_mc=2.28, extra_ml=True)
    assert drop_agostic_carbon(el, xyz, G, ml, cut=0) == ml

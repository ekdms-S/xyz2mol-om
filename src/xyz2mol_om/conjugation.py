"""`Conj` 후보 규칙 — 규칙 A · R2 · R3 · R4 (R5 는 파이프라인에 있다).

⚠️ **`ognm-bh-workspace/code/analysis/scratch/260830_fit_t10_charge.py` 에서 이관한 코드다**
(2026-09-03). 함수 본문은 **그대로** 옮겼다 — 판정 규칙을 바꾸지 않는다.
"""

# ruff: noqa: E501
from __future__ import annotations

import networkx as nx
import numpy as np

from .config import CAP, R3MODE, R3RING, R4RING, RULEA, TAU_P, _LP_DEG
from .geometry import plane_rms


def rule_a_ok(n):
    return {"ge5": n >= 5, "eq6": n == 6, "off": False}[RULEA]

def lp_donor(elem, deg):
    """그 원자의 중성 σ 골격이 이미 찼는가 = π 에 고립쌍으로만 참여하는 자리인가."""
    d = _LP_DEG.get(elem)
    return d is not None and deg >= d

def conj_forbidden(G, el, q_frag=None, xyz=None):
    """`Conj` 가 될 수 없는 결합 집합. `q_frag` = {원자: 그 조각의 EHT 전하}(피리디늄 면제)."""
    bad = set()
    donors = set()
    for x in G.nodes():
        if not lp_donor(el[x], G.degree(x)):
            continue
        if el[x] == "N" and q_frag is not None and q_frag.get(x, 0) > 0:
            continue  # 피리디늄 N⁺ 예외
        donors.add(x)
        for y in G[x]:
            bad.add((min(x, y), max(x, y)))
    if R4RING:  # R4 — 4n 전탄소 고리(4·8원) 중 **비평면**인 것은 Kekulé
        for r_ in nx.cycle_basis(G):
            if len(r_) in (4, 8) and all(el[x] == "C" for x in r_):
                if xyz is None or plane_rms(xyz[np.array(r_)]) > TAU_P:
                    for a, b in zip(r_, r_[1:] + r_[:1]):
                        bad.add((min(a, b), max(a, b)))
    if R3RING and donors:  # R3 — 그 원자를 낀 5원 고리는 통째로 Kekulé
        for r_ in nx.cycle_basis(G):
            if len(r_) != 5:
                continue
            din = [x for x in r_ if x in donors]
            if not din:
                continue
            if R3MODE == "N" and not all(el[x] == "N" for x in din):
                continue
            if R3MODE == "mono" and not (
                len(din) == 1 and all(el[x] == "C" for x in r_ if x not in din)
            ):
                continue
            for a, b in zip(r_, r_[1:] + r_[:1]):
                bad.add((min(a, b), max(a, b)))
    return bad

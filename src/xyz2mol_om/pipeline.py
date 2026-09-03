"""★ 채택 파이프라인 — `predict_T3_EHT` (설계도 §3 `1c`).

⚠️ **`ognm-bh-workspace/code/analysis/scratch/260830_fit_t10_charge.py` 에서 이관한 코드다**
(2026-09-03). 함수 본문은 **그대로** 옮겼다 — 판정 규칙을 바꾸지 않는다.
"""

# ruff: noqa: E501
from __future__ import annotations

import collections

import networkx as nx
import numpy as np

from .config import (CAP, EHTCOST, EHTMINFRAG, EHTSKIP, LNORM_ON, LNORM_SKIP_CONJ, LPA,
                     LPCOND, LPCOND_NOCONJ, R2CONJ, R5SOLO, ROPW, TAU_P, USE_ROP)
from .charge import _qfrag
from .conjugation import conj_forbidden, rule_a_ok
from .eht import eht_frag_charges
from .geometry import plane_rms
from .likelihood import deg_cell
from .solvers import _kek_val, _solve_cap, _solve_sc, r6_swap


def predict_T3_EHT(el, xyz, G, scores4, bml=None, ml_sc=None, q_eht=None, coord=None, rop=None):
    """★ 채택안 `D_eht` — 설계도 §3 `1c` 전 단계. 반환 `(내부 클래스, M–L 클래스)`.

    `bml`   {배위원자: M–L 결합차수 합}  — 용량 예산에 들어간다 (기준선 Single)
    `ml_sc` {(금속,배위원자): {클래스: 점수}} — 주면 M–L 차수도 같이 최적화한다
    `q_eht` {조각 최소원자idx: 전하} — 안 주면 `eht_frag_charges` 로 직접 계산한다
    `rop`   {(i,j): EHT 겹침 밀도} — `USE_ROP=1` 이고 scores4 항목이 5-튜플이면 두 번째 차원
    `coord` 배위 원자 집합 — 공액 판정 탐색에서 **미달 벌점을 면제**한다(M–L 이 흡수하므로)

    ⚠️ `bml` 에 **하프틱 M–L 을 넣으면 안 된다** — 하프틱은 차수를 안 매기고 π 계에 공유되어
       원자에 귀속되지 않는다(§3 `5a`). 넣으면 Cp 탄소의 예산이 헛되이 소모된다.
    """
    bml = bml or {}
    if q_eht is None:
        q_eht = eht_frag_charges(el, xyz, G)
    # ① 규칙 A — 평면 고리(크기>=5) 중 **π 여유가 남은** 원자에만 걸린다
    sat = {x for x in G.nodes if G.degree(x) >= CAP.get(el[x], 4)}
    ringA = set()
    for r in nx.cycle_basis(G):
        if rule_a_ok(len(r)) and plane_rms(xyz[np.array(r)]) <= TAU_P:
            ringA |= {
                (min(a, b), max(a, b))
                for a, b in zip(r, r[1:] + r[:1])
                if a not in sat and b not in sat
            }
    # ② 원소쌍별 4클래스 거리 우도
    sc = {}
    for a, b in G.edges:
        e = (min(a, b), max(a, b))
        k = tuple(sorted((el[a], el[b])))
        if k not in scores4:
            continue
        ent = scores4[k]
        med, scl, lp = ent[0], ent[1], ent[2]
        d = float(np.linalg.norm(xyz[a] - xyz[b]))
        # 🔴 `LPCOND` — 이 결합의 **끝점 차수 셀** 사전확률로 갈아끼운다 (셀이 없으면 전역 `lp`).
        #    `LPCOND_NOCONJ` 면 `Conj`(클래스 3)만 전역으로 되돌린다. 판정은 `LPCOND` 주석 참조.
        lp_e = lp
        if LPCOND and len(ent) >= 6 and ent[5]:
            _cell = deg_cell(el, a, b, {x: G.degree(x) for x in (a, b)})
            _lc = ent[5].get(_cell)
            if _lc is not None:
                lp_e = {c: (lp[c] if (LPCOND_NOCONJ and c == 3) else v) for c, v in _lc.items()}
        sc[e] = {
            c: -abs(d - med[c]) / scl[c]
            + LPA * lp_e.get(c, lp[c])
            - (float(np.log(2 * scl[c])) if LNORM_ON and not (LNORM_SKIP_CONJ and c == 3) else 0.0)
            for c in med
        }
        if USE_ROP and rop is not None and len(ent) >= 5 and e in rop:
            rmed, rscl = ent[3], ent[4]
            rv = rop[e]
            for c in list(sc[e]):
                if c in rmed:
                    sc[e][c] += ROPW * (-abs(rv - rmed[c]) / rscl[c])
    # ③ 공액 집합 — 포화 원자에 닿는 결합은 Single 만 허용한 우도 위에서 정한다
    sc_sat = {
        e: ({0: v[0]} if (e[0] in sat or e[1] in sat) and 0 in v else dict(v))
        for e, v in sc.items()
    }
    if R2CONJ:  # ★ R2 — 피롤형 헤테로원자에서 `Conj` 를 금지한다 (위 주석)
        qf = {}
        if q_eht:
            for comp0 in nx.connected_components(G):
                qf.update(dict.fromkeys(comp0, q_eht.get(min(comp0), 0)))
        _bad = conj_forbidden(G, el, qf, xyz)
        ringA -= _bad
    conj = {e for e, v in _solve_sc(G, el, sc_sat, ringA, coord or set(), bml).items() if v == 3}
    if R2CONJ:
        conj -= _bad
    if R5SOLO and conj:  # R5 — 고립 `Conj`(이웃에 `Conj` 가 없는 결합)는 비편재가 아니다
        Gj = nx.Graph()
        Gj.add_edges_from(conj)
        conj = {e for e in conj if Gj.degree(e[0]) > 1 or Gj.degree(e[1]) > 1}
    # ④ 원자가 상한 경성 제약 — 정확 해 (M–L 은 Triple 까지)
    cls, mlout = _solve_cap(G, el, sc, conj, bml, ml_sc, ml_max=2)
    # ⑤ EHT 조각 전하 목표
    for comp0 in nx.connected_components(G):
        comp = set(comp0)
        tgt = q_eht.get(min(comp))
        if tgt is None:
            continue
        if len(comp) < EHTMINFRAG:
            continue  # 작은 조각은 EHT 목표를 믿지 않는다 (위 `EHTMINFRAG` 주석)
        if EHTSKIP and "".join(sorted(el[x] for x in comp)) in EHTSKIP:
            continue  # 이 조성에서는 EHT 목표가 계통적으로 틀린다 (위 `EHTSKIP` 주석)
        edges = [e for e in cls if e[0] in comp and cls[e] != 3 and e in sc]
        snap = {e: cls[e] for e in edges}
        cost = 0.0
        for _ in range(12):
            d = tgt - round(_qfrag(G, el, cls, comp))
            if d == 0 or abs(d) % 2 == 1:
                break
            use = collections.defaultdict(float, _kek_val(G, el, cls))
            for x in comp:
                use[x] += bml.get(x, 0.0)
            best = None
            for e in edges:
                c0 = cls[e]
                if d > 0:
                    if c0 >= 2 or any(CAP.get(el[x], 4) - use[x] < 1 - 1e-9 for x in e):
                        continue
                    c1 = c0 + 1
                else:
                    if c0 <= 0:
                        continue
                    c1 = c0 - 1
                g = sc[e].get(c1, -1e9) - sc[e].get(c0, 0.0)
                if best is None or g > best[0]:
                    best = (g, e, c1)
            if best is None:
                break
            cost += -best[0]
            if EHTCOST >= 0 and cost > EHTCOST:  # 목표를 포기하고 되돌린다
                for e, c in snap.items():
                    cls[e] = c
                break
            cls[best[1]] = best[2]
    r6_swap(
        G, el, xyz, cls, bml
    )  # ⑥ R6 — 같은 중심 동일 원소의 거리↔차수 순서 정합 (교환이라 총합 불변)
    return cls, mlout

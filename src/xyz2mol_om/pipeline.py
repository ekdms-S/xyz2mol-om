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
                     LPCOND, LPCOND_NOCONJ, R2CONJ, R5SOLO, ROPW, TAU_P, USE_ROP, R7MIN, R7RING, THETA_HAPTIC,)
from .charge import _qfrag
from .conjugation import conj_forbidden, lp_donor, rule_a_ok
from .eht import eht_frag_charges
from .geometry import plane_rms
from .likelihood import deg_cell
from .solvers import _kek_val, _solve_cap, _solve_sc, r6_swap
from .ml_order import load_b_ml_mayer, ml_order_scores, ml_order_scores_dist


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


# ★★ 통일 진입점 (2026-09-03) — **T3 · M–L 차수 · 하프틱(T5) · R7 을 한 함수에서 낸다.**
#   왜: 같은 판정을 채점기(`260831_propagation_prior_cv.py`)와 배포(`xyz2mol-om`)가 각자
#   조립하다가 **네 자리**에서 갈렸다(2026-09-03 실측, 결합 0.12~0.90%):
#     ① ④ 상한 해에 넘기는 `b_ML` 예산에서 haptic 을 뺐나  ② M–L 차수 후보에서 haptic 을 뺐나
#     ③ agostic(`C–H···M`) 을 뺐나                      ④ T5 의 Y 후보가 조각 이웃인가 전부인가
#   ⇒ **조립을 호출자에게 맡기지 않는다.** 호출자는 T4 후보(`ml_raw`)와 Mayer(`wbo`)만 준다.
#   ⚠️ 이 함수는 배포 `xyz2mol-om/src/xyz2mol_om/pipeline.py` 의 같은 이름 함수와 **본문이 같아야
#      한다.** 한쪽만 고치지 말 것.
MLIKE_EXTRA = {"B", "Al"}  # metal-like = 금속 ∪ {B, Al} (설계도 §3.1 (c))


def drop_agostic(el, G, ml_raw):
    """`C–H···M` 만 뺀다 — μ-H 와 `B–H···M`(보로하이드라이드)은 진짜 3c2e 라 남긴다 (설계도 §3.0 [T4]).

    판정  뺀다 ⟺ el[X] = H  AND  metal-like 이웃 1개  AND  내부 이웃에 metal-like 아닌 것이 있다
    """
    nmet = collections.Counter(x for _m, x in ml_raw)
    out = []
    for m, x in ml_raw:
        if el[x] == "H":
            n_like = nmet[x] + sum(1 for y in G[x] if el[y] in MLIKE_EXTRA)
            if n_like == 1 and any(el[y] not in MLIKE_EXTRA for y in G[x]):
                continue
        out.append((m, x))
    return out


def _angle_ok(xyz, m, x, y, theta):
    """∠(M–X–Y) < theta ?  Y 는 호출자가 고른다."""
    v1, v2 = xyz[m] - xyz[x], xyz[y] - xyz[x]
    cs = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12))
    return float(np.degrees(np.arccos(max(-1.0, min(1.0, cs))))) < theta


def _closest_mid(xyz, x, m, cand):
    """X 의 이웃 후보 중 **결합 중점이 M 에 가장 가까운** 것."""
    return min(cand, key=lambda q: float(np.linalg.norm((xyz[x] + xyz[q]) / 2 - xyz[m])))


def predict_T3_T5(el, xyz, G, scores4, ml_raw, wbo, bml_model=None, bml_fb=None,
                  q_eht=None, rop=None):
    """T4 후보와 Mayer 만 받아 **T3 4클래스 · M–L 차수 · 하프틱**을 끝까지 낸다.

    반환 `(cls, mlout, hap, ml_pred)`
      `cls`     {(i,j): 0 Single · 1 Double · 2 Triple · 3 Conj}   내부 결합
      `mlout`   {(m,x): 클래스}                                    M–L 차수 (하프틱 제외)
      `hap`     {(m,x)}                                            하프틱 M–L
      `ml_pred` [(m,x)]                                            agostic 을 뺀 T4 결합

    2-pass 인 이유: 하프틱 M–L 은 **차수를 안 매기고 예산도 안 쓴다**(설계도 §3 5a). 그런데
    하프틱 여부는 T3(π 조각)를 알아야 정해진다. 그래서 **1회차는 예산 0** 으로 T3 를 풀어
    π 후보를 얻고, **각도만으로** 하프틱을 예비 판정한 뒤, 그것을 뺀 예산으로 2회차를 푼다.
    ⚠️ 예비 판정은 예산용이다 — **최종 하프틱은 2회차 결과의 π 조각으로 다시 정한다.**
    """
    if bml_model is None:
        bml_model, bml_fb = load_b_ml_mayer()
    coord = {x for _m, x in ml_raw}
    ml_pred = drop_agostic(el, G, ml_raw)
    # 1회차 — 예산 0 · M–L 최적화 없음
    cls0, _ = predict_T3_EHT(el, xyz, G, scores4, {}, None, q_eht, coord, rop)
    unsat0 = {x for e, v in cls0.items() if v in (1, 2, 3) for x in e}
    hap_pre = set()
    for m, x in ml_pred:
        nb = list(G[x])
        if x in unsat0 and nb and _angle_ok(xyz, m, x, _closest_mid(xyz, x, m, nb), THETA_HAPTIC):
            hap_pre.add((m, x))
    keep = [p for p in ml_pred if p not in hap_pre]
    bml = collections.defaultdict(float)
    for _m, x in keep:
        bml[x] += 1.0  # M–L 기준선 = Single
    # 🔴 `wbo` 가 없으면 **거리 폴백**으로 M–L 차수를 매긴다 (2026-09-03).
    #   Mayer 가 없으면 T8 은 전 결합을 `Single` 로 내보낸다(`Double` F1 **0.0000** · 실측
    #   300구조 TP 0 / FN 40). 거리 단조 임계 폴백은 같은 풀·refcode 5-fold CV 에서
    #   `Double` **0.6976** · `Triple` 0.6515 (Mayer 판 .7318 / .7230 · 자명 기준선 0.0000).
    #   ⚠️ `wbo` 가 **있으면 쓰지 않는다** — 거리가 확실히 진다(`Double` −0.034).
    if wbo:
        ml_sc = ml_order_scores(el, keep, wbo, bml_model, bml_fb)
    else:
        ml_sc = ml_order_scores_dist(el, keep, xyz)
    # 2회차 — 이것이 출력이다
    cls, mlout = predict_T3_EHT(el, xyz, G, scores4, dict(bml), ml_sc, q_eht, coord, rop)
    # T5 — 최종 하프틱. Y 후보는 **같은 π 조각 이웃**이다 (실측 2026-09-03: 조각 이웃 F1 .9810 ·
    #      내부 이웃 전부 .9803 — 정밀도가 조각 이웃 쪽이 높다).
    pi = nx.Graph()
    pi.add_edges_from(e for e, v in cls.items() if v in (1, 2, 3))
    pifrag = {x: i for i, c in enumerate(nx.connected_components(pi)) for x in c}
    hap, hap_by_m = set(), collections.defaultdict(set)
    for m, x in ml_pred:
        if x not in pifrag:
            continue
        nb = [y for y in G[x] if pifrag.get(y) == pifrag[x]]
        if nb and _angle_ok(xyz, m, x, _closest_mid(xyz, x, m, nb), THETA_HAPTIC):
            hap.add((m, x))
            hap_by_m[m].add(x)
    # R7 — 하프틱 고리 안의 R2 도너를 π 후보로 되돌린다 (2026-09-03 채택 · 판정은 설계도 §3.1-R7)
    if R7RING:
        mlset = set(ml_pred)
        donors = {x for x in G if lp_donor(el[x], G.degree(x))}
        for r5 in (nx.cycle_basis(G) if donors else []):
            if len(r5) != 5:
                continue
            din = [x for x in r5 if x in donors]
            if not din:
                continue
            for m in list(hap_by_m):
                if len(set(r5) & hap_by_m[m]) < R7MIN:
                    continue
                for x in din:
                    if x in hap_by_m[m] or (m, x) not in mlset:
                        continue
                    nb = [y for y in G[x] if y in pifrag]
                    if nb and _angle_ok(xyz, m, x, _closest_mid(xyz, x, m, nb), THETA_HAPTIC):
                        hap.add((m, x))
                        hap_by_m[m].add(x)
    for e in hap:  # 하프틱에는 차수를 안 매긴다
        mlout.pop(e, None)
    return cls, mlout, hap, ml_pred

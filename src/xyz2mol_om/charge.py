"""형식전하 — 원자별 `q_atom` · 공액 조각 `frag_charge` · 조각 합 `_qfrag` · 출력 변환기 `kekulize`.

⚠️ **`ognm-bh-workspace/code/analysis/scratch/260830_fit_t10_charge.py` 에서 이관한 코드다**
(2026-09-03). 함수 본문은 **그대로** 옮겼다 — 판정 규칙을 바꾸지 않는다.
"""

# ruff: noqa: E501
from __future__ import annotations

import collections

import networkx as nx

from .config import (ALT, CAP, FULL, HUCKEL, NAMEEL, ORD4, PAT, PATM, QHV, R, ROMAN, VAL)


def q_atom(e, b, deg=None, nb=()):
    """(a) q_i = v + b − 정원 (옥텟 가정 `lp = 4 − b`).

    ★ (a′) 옥텟이 깨지는 자리만 덮는다 (2026-08-30, §5.0.8 ④ · V2d).
    `deg` = 리간드 *내부* 이웃 수 · `nb` = 그 이웃들의 원소 튜플.
    🔴 **이웃 원소 조건이 필수다** — 없이 `(원소, deg, b)` 만으로 걸면 회귀가 난다(실측):
      · `("N",2,4)` 를 무조건 −1 로 두면 **아자이드 N₃ 중심 N**(이웃 N,N)과
        **이소시아나이드 N**(이웃 C,C)까지 걸려 리간드 전하가 −2 어긋난다.
      · `("C",2,2)` 를 무조건 0 으로 두면 **CF₂**(이웃 F,F)·`C7H6`(이웃 C,C)이 틀린다.
    """
    # 🔴 `QHV=1` — **초원자가 일반화** (2026-09-03). `lp = 4 − b` 를 `lp = max(0, 4 − b)` 로 둔다
    #   ⇒ `q = v − b − 2·lp`. `b ≤ 4` 에서는 `v + b − 8` 과 **항등**(현행과 완전히 동일)이고,
    #   `b > 4` 에서만 `q = v − b` 가 된다. 손예외 `S(=O)₂`(b 6) · `P=O`(b 5) 를 흡수하고,
    #   **나이트로 `–N(=O)=O`(b 5)** 를 `+2` → `0` 으로 고친다(`ACARAZ` 기전).
    #   실측(정답 배정 · train 26,075 · 조각 94,117): 초원자가 원자 1,427 · 그 조각의
    #   EHT 목표 불일치 **50.7% → 28.5%** (626 → 352). `b ≤ 4` 자리는 한 건도 안 바뀐다.
    if QHV and b > 4.0 + 1e-9:
        return VAL.get(e, 4) - b
    nO, nN = nb.count("O"), nb.count("N")
    if e == "C" and deg == 2 and b == 2.0 and nN + nO >= 1:
        # 헤테로원자 안정화 카벤 — 6전자. 옥텟식 −2 (2026-09-02 오너 지적으로 `nO` 추가).
        #   NHC `:C(NR)₂`(이웃 N) 는 처음부터 잡혔지만 **Fischer 카벤 `:C(OR)R`(이웃 O)** 가
        #   빠져 −2 로 떨어지고 있었다 — 실측 **374건**(전부 M 배위 · `C–O` 이웃).
        #   ⚠️ Schrock 알킬리덴(`C–H` 465 · `C–C` 397 · `H–H` 111)은 `nN+nO = 0` 이라
        #      **−2 로 남는다 — 그게 맞다.** 헤테로원자 조건이 둘을 가른다.
        #   ⚠️ `S` 는 넣지 않았다 — `C–S` 95건이 전부 M 배위가 아니라 카벤 공여자가 아니다.
        return 0
    if e == "S" and deg == 3 and b == 4.0 and nO >= 1:
        return 0  # 술폭사이드 S=O — 10전자.                      옥텟식 +2
    if e == "S" and deg == 4 and b == 6.0 and nO >= 2:
        return 0  # 술폰·술포네이트 중심 — 12전자.                 옥텟식 +4
    if e == "P" and deg == 4 and b == 5.0 and nO + nN >= 1:
        return 0  # 포스핀 옥사이드 P=O · 포스핀이미드 P=N — 10전자. 옥텟식 +2
    if e == "N" and deg == 2 and b == 4.0 and nO == 2:
        return -1  # 나이트로(두 N=O) — 10전자.                   옥텟식 +1
    return VAL.get(e, 4) + b - FULL.get(e, 8)

def frag_charge(el, atoms, edges, orders, deg=None, nbrs=None, out=None):
    """공액 조각 하나의 전하 — (b) 규칙.

    단환 전탄소 **`CmHm`** → Hückel `z = m − (4n+2)`   ← 고리 원자마다 외부 결합이 정확히 1 이어야 한다
    그 외                  → Kekulé 최대화(최대 매칭) 후 (a) 합

    🔴 버그 2건 수정 (2026-08-30, §5.0.8 ②):
      B1  치환된 전탄소 고리에도 Hückel 을 걸어 **페닐 C6H5 가 0** 이었다(정답 −1).
          `CmHm` 확인을 넣으니 Kekulé 로 내려가 −1 이 된다.
      B2  `min(HUCKEL, key=|m−h|)` 가 **m=8 에서 6·10 동점 → 앞의 6** 을 골라
          **COT 가 +2** 였다(정답 −2). 동점이면 **큰 쪽**을 고른다.
    """
    ring = len(edges) == len(atoms) and all(
        sum(1 for a, b in edges if a == v or b == v) == 2 for v in atoms
    )
    huckel = None
    if ring and all(el[a] == "C" for a in atoms):
        m = len(atoms)
        if all(orders.get(v, 0.0) == 1.0 for v in atoms):  # ← B1: CmHm 확인
            d = min(abs(m - h) for h in HUCKEL)
            huckel = m - max(h for h in HUCKEL if abs(m - h) == d)  # ← B2: 동점이면 큰 쪽
            if out is None:
                return huckel
            # ⑥ 출력 변환기 — Hückel 분기도 **골격은 매칭으로 낸다**(2026-09-02).
            #   전하만 Hückel 이 정하고, S/D/T 골격은 아래 매칭이 준다.
            #   ⚠️ 짝수 고리 다이아니온(η⁴-C₄R₄²⁻·η⁸-COT²⁻)은 골격이 **중성 Kekulé** 라
            #      원자별 전하를 골격에서 유추할 수 없다 — 조각 전하로 내야 한다.
    G = nx.Graph()
    G.add_nodes_from(atoms)
    G.add_edges_from(edges)
    match = nx.max_weight_matching(G, maxcardinality=True)
    md = {}
    for a, b in match:
        md[(min(a, b), max(a, b))] = 2.0
    if out is not None:
        for a, b in edges:
            out[(min(a, b), max(a, b))] = md.get((min(a, b), max(a, b)), 1.0)
    if huckel is not None:
        return huckel
    q = 0
    for v in atoms:
        b = sum(md.get((min(v, w), max(v, w)), 1.0) for w in G[v])
        b += orders.get(v, 0.0)  # 조각 밖으로 나가는 결합의 차수
        q += q_atom(el[v], b, None if deg is None else deg.get(v), (nbrs or {}).get(v, ()))
    return q

def _qfrag(G, el, cls, comp):
    """조각 하나의 전하 — 본문 q 규칙과 같다(공액 조각은 `frag_charge`)."""
    pc = {e for e, v in cls.items() if v == 3 and e[0] in comp}
    Gc = nx.Graph()
    Gc.add_edges_from(pc)
    ca = set(Gc.nodes)
    DEG = {v: G.degree(v) for v in comp}
    NB = {v: tuple(sorted(el[w] for w in G[v])) for v in comp}
    q = 0.0
    for cm in nx.connected_components(Gc) if pc else []:
        sub = [(a, b) for a, b in Gc.subgraph(cm).edges]
        outer = {
            v: sum(
                0.0 if (min(v, w), max(v, w)) in pc else ORD4[cls.get((min(v, w), max(v, w)), 0)]
                for w in G[v]
                if w not in cm
            )
            for v in cm
        }
        q += frag_charge(el, list(cm), sub, outer, DEG, NB)
    for v in comp:
        if v in ca:
            continue
        q += q_atom(
            el[v], sum(ORD4[cls.get((min(v, w), max(v, w)), 0)] for w in G[v]), DEG[v], NB[v]
        )
    return q

def kekulize(G, el, cls, bml=None):
    """⑥ **출력 변환기** — 4클래스 예측(`Conj` 포함)을 정수 S/D/T 로 되돌린다 (2026-09-02).

    반환 `(orders, frag_q)`
      `orders` {(i,j): 1|2|3}          — 출력용 정수 결합차수 (내부결합 전량)
      `frag_q` {조각 대표원자(min idx): 전하} — **골격에서 유추할 수 없는 전하**를 담는다

    **왜 별도 반환이 필요한가.** 홀수 π 계(Cp⁻·알릴⁻)는 매칭이 원자 하나를 남기고 그 자리가
    그대로 −1 이 되므로 골격만으로 전하가 읽힌다. 그런데 **짝수 고리 다이아니온**
    (η⁴-C₄R₄²⁻ · η⁸-COT²⁻)은 완전매칭이 되어 **골격이 중성 Kekulé** 인데 실제 전하는 −2 다.
    −2 는 결합 패턴이 아니라 **전자 2개**라 어떤 S/D/T 배정으로도 표현되지 않는다.
    ⇒ 그런 조각은 **원자별로 찍지 말고 리간드 전하로** 낸다(Hückel 분기가 그 전하를 준다).

    ⚠️ 전하·원자가 경로(`_qfrag`)와 **같은 매칭**을 쓴다 — 별도 매칭을 돌리면 출력과 전하가
       어긋난다(`relabel.py` 의 거리우도 가중 매칭을 재활용하면 안 되는 이유).
    """
    bml = bml or {}
    orders = {}
    for e, v in cls.items():
        if v != 3:
            orders[(min(e), max(e))] = int(ORD4[v])
    pc = {e for e, v in cls.items() if v == 3}
    Gc = nx.Graph()
    Gc.add_edges_from(pc)
    DEG = {v: G.degree(v) for v in G.nodes}
    NB = {v: tuple(sorted(el[w] for w in G[v])) for v in G.nodes}
    frag_q = {}
    for cm in nx.connected_components(Gc) if pc else []:
        sub = [(a, b) for a, b in Gc.subgraph(cm).edges]
        outer = {
            v: sum(
                0.0 if (min(v, w), max(v, w)) in pc else ORD4[cls.get((min(v, w), max(v, w)), 0)]
                for w in G[v]
                if w not in cm
            )
            for v in cm
        }
        out = {}
        q = frag_charge(el, list(cm), sub, outer, DEG, NB, out=out)
        for e, o in out.items():
            orders[e] = int(o)
        # 골격이 주는 전하와 조각 전하가 어긋나면(짝수 고리 다이아니온) 리간드 전하로 낸다
        q_skel = 0
        for v in cm:
            b = sum(out.get((min(v, w), max(v, w)), 1.0) for w in Gc[v]) + outer.get(v, 0.0)
            q_skel += q_atom(el[v], b, DEG.get(v), NB.get(v, ()))
        if round(q_skel) != round(q):
            frag_q[min(cm)] = q
    return orders, frag_q

def parse_os(m, nm):
    """금속 `m` 의 산화수를 이름에서 읽는다. 혼합원자가·불명이면 None (채점 제외)."""
    nm = (nm or "").lower()
    stems = [NAMEEL.get(m, m.lower())] + ALT.get(m, [])
    ok = lambda x: any(x.startswith(y) or y in x for y in stems)  # noqa: E731
    if [g for g in PATM.findall(nm) if ok(g[0])]:
        return None  # 혼합원자가 — 채점 제외
    f = {ROMAN[r] for x, r in PAT.findall(nm) if ok(x)}
    return f.pop() if len(f) == 1 else None

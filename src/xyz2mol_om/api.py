"""상위 API — `xyz` → 결합 · 차수 · 리간드 전하 · 금속 산화수.

한 함수만 알면 된다:

    from xyz2mol_om import predict
    r = predict(elements, coords, total_charge=0, wbo=None)

`r` (dict)
    `bonds`      {(i, j): 1.0 | 2.0 | 3.0}      배위자 **내부** 결합과 차수 (Kekulé 정수)
    `conj`       {(i, j)}                        비편재(`Conj`)로 판정된 결합
    `ml_bonds`   {(m, x): 1 | 2 | 3}             M–L 결합과 차수 (하프틱 제외)
    `haptic`     {(m, x)}                        하프틱 M–L
    `eta`        {(m, 조각idx): k}               η^k
    `q_ligand`   {조각 최소원자idx: q}           리간드 조각 전하
    `os_metal`   {m: OS}                          금속 산화수 (`total_charge` 를 주면)
    `frag_q`     {조각 최소원자idx: q}           골격으로 표현 안 되는 잔여 조각 전하

⚠️ **`wbo`(Mayer 결합차수)가 없으면 M–L 판정의 거부권과 T8 차수를 쓸 수 없다.**
   `xtb --sp` 산출물을 `{(금속idx, 원자idx): w}` 로 넘긴다. 없으면 M–L 은 거리로만 잡고
   차수는 전부 `Single` 로 둔다(설계도 §3 3·5a).
"""

# ruff: noqa: E501
from __future__ import annotations

import collections

import networkx as nx
import numpy as np

from .charge import _qfrag, kekulize
from .config import METALS, ORD4, RCOV, THETA_HAPTIC
from .connectivity import load_dint
from .eht import eht_frag_charges
from .likelihood import load_scores4
from .ml_order import load_b_ml_mayer
from .pipeline import predict_T3_EHT


def _ml_candidates(el, xyz, dbond, c1g, wbo):
    """T4 — M–X 결합 유무. `d < d_bond(M,X)` AND `w > w_veto(M,X)`.

    ⚠️ agostic 제외(`C–H···M`)는 설계도 §3 3 의 규칙이다.
    """
    idx = [i for i, e in enumerate(el) if e not in METALS]
    mets = [i for i, e in enumerate(el) if e in METALS]
    raw = []
    for m in mets:
        for x in idx:
            d = float(np.linalg.norm(xyz[x] - xyz[m]))
            tb, wv = dbond.get((el[m], el[x]), (c1g * (RCOV.get(el[x], 1.6) + RCOV.get(el[m], 1.6)), 0.0))
            if d < tb and (wbo or {}).get((m, x), 1.0) > wv:
                raw.append((m, x))
    return raw


def predict(elements, coords, total_charge=None, wbo=None, scores4=None, dint=None):
    """`xyz` → 결합·차수·전하·산화수. 인자·반환은 모듈 docstring 참조."""
    el = list(elements)
    xyz = np.asarray(coords, dtype=float)
    sc4 = scores4 if scores4 is not None else load_scores4()
    d_int, d_fb = dint if dint is not None else load_dint()

    # ① T1 — 배위자 내부 결합 (거리)
    idx = [i for i, e in enumerate(el) if e not in METALS]
    G = nx.Graph()
    G.add_nodes_from(idx)
    for ii in range(len(idx)):
        for jj in range(ii + 1, len(idx)):
            a, b = idx[ii], idx[jj]
            if float(np.linalg.norm(xyz[a] - xyz[b])) < d_int.get(tuple(sorted((el[a], el[b]))), d_fb):
                G.add_edge(a, b)

    # ② T4 — M–L 결합 (거리 + Mayer 거부권)
    import csv as _csv

    from .config import DATA

    dbond, c1g = {}, 1.3002
    for r in _csv.DictReader(open(DATA / "d_bond.csv")):
        if r["M"] == "*":
            c1g = float(r["d_bond"])
        else:
            dbond[(r["M"], r["X"])] = (float(r["d_bond"]), float(r["w_veto"]))
    ml_raw = _ml_candidates(el, xyz, dbond, c1g, wbo)

    # ③ T5 — haptic 판정은 π 조각을 알아야 하므로 T3 뒤로 미룬다. 예산에는 Single 기준선.
    BML, BML_FB = load_b_ml_mayer()
    ml_sc = {}
    for m, x in ml_raw:
        w = (wbo or {}).get((m, x))
        ent = BML.get((el[m], el[x]))
        if ent is None or w is None or ent[0] == "const":
            ml_sc[(m, x)] = {(ent[1] if (ent and ent[0] == "const") else BML_FB): 0.0}
        elif ent[0] == "thr":
            _, t1, t2, kk = ent
            sm = {0: 0.0}
            if t1 != float("inf"):
                sm[1] = kk * (w - t1)
                if t2 != float("inf"):
                    sm[2] = sm[1] + kk * (w - t2)
            ml_sc[(m, x)] = sm
        else:
            _, med, scl, lp = ent
            ml_sc[(m, x)] = {c: -abs(w - med[c]) / scl[c] + lp[c] for c in med}
    bml = collections.defaultdict(float)
    for _m, x in ml_raw:
        bml[x] += 1.0

    # ④ T3 — 채택 파이프라인 (규칙 A · R2~R5 · 상한 정확 해 · EHT 조각 전하)
    q_eht = eht_frag_charges(el, xyz, G)
    cls, mlout = predict_T3_EHT(
        el, xyz, G, sc4, dict(bml), ml_sc, q_eht, {x for _m, x in ml_raw}
    )

    # ⑤ T5·T6 — haptic 과 η^k (π 조각 = Conj ∪ Double ∪ Triple 의 연결 성분)
    conj = {e for e, v in cls.items() if v == 3}
    pi = nx.Graph()
    pi.add_edges_from(e for e, v in cls.items() if v in (1, 2, 3))
    pifrag = {x: i for i, c in enumerate(nx.connected_components(pi)) for x in c}
    hap, eta = set(), collections.Counter()
    for m, x in ml_raw:
        if x not in pifrag:
            continue
        nb = [y for y in G[x] if y in pifrag and pifrag[y] == pifrag[x]]
        if not nb:
            continue
        y = min(nb, key=lambda q: np.linalg.norm((xyz[x] + xyz[q]) / 2 - xyz[m]))
        v1, v2 = xyz[m] - xyz[x], xyz[y] - xyz[x]
        cs = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12))
        if np.degrees(np.arccos(max(-1.0, min(1.0, cs)))) < THETA_HAPTIC:
            hap.add((m, x))
            eta[(m, pifrag[x])] += 1

    # ⑥ 출력 변환기 — 4클래스 → 정수 S/D/T + 잔여 조각 전하
    orders, frag_q = kekulize(G, el, cls, dict(bml))
    q_lig = {}
    for comp in nx.connected_components(G):
        q_lig[min(comp)] = round(_qfrag(G, el, cls, set(comp)))

    os_metal = {}
    mets = [i for i, e in enumerate(el) if e in METALS]
    if total_charge is not None and mets:
        num = total_charge - sum(q_lig.values())
        if num % len(mets) == 0:
            os_metal = dict.fromkeys(mets, num // len(mets))

    return {
        "bonds": orders,
        "conj": conj,
        # `mlout` 은 클래스 코드(0 S · 1 D · 2 T)다 — 차수로 바꿔 내보낸다. 하프틱은 차수를 안 매긴다.
        "ml_bonds": {k: v + 1 for k, v in mlout.items() if (min(k), max(k)) not in {(min(a, b), max(a, b)) for a, b in hap}},
        "haptic": hap,
        "eta": dict(eta),
        "q_ligand": q_lig,
        "frag_q": frag_q,
        "os_metal": os_metal,
        "_cls4": cls,
        "_graph": G,
    }

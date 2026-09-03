"""상위 API — `xyz` → **금속별 / 리간드별** 결합 · 차수 · 전하 · 산화수.

    from xyz2mol_om import predict
    r = predict(elements, coords, total_charge=0, wbo=wbo)

반환 구조 (dict)

    r["metals"]  = [ {                      금속 하나
          "index":        int,              전체 좌표 기준 원자 인덱스
          "element":      str,
          "oxidation":    int | None,       산화수 (total_charge 를 줘야 나온다)
          "mm_bonds":     {(m1, m2): 1|2|3|4},   M–M 결합 차수
      }, … ]

    r["ligands"] = [ {                      리간드 조각 하나
          "index":        int,              조각 번호 (0부터)
          "atoms":        [int, …],         전체 좌표 기준 원자 인덱스
          "bonds_4class": {(i,j): "Single"|"Double"|"Triple"|"Conj"},
          "bonds_kekule": {(i,j): 1|2|3},   ⑥ 출력 변환기 산출 (정수)
          "smiles":       str | None,       Kekulé SMILES. 배위 원자는 원자 맵 `[X:n]`
          "smiles_ok":    bool,             왕복 검증(차수·전하·H·화학적 타당성) 통과 여부
          "smiles_note":  str,              실패 사유 (통과면 "")
          "coordinating": [int, …],         금속에 배위한 원자
          "ml_bonds":     {(m, x): {"type": "sigma"|"haptic", "order": 1|2|3|None}},
          "eta":          {m: k},           그 금속에 대한 η^k (haptic 일 때)
          "charge":       int,              리간드 전하 q_L
          "residual_charge": int | None,    골격으로 표현 안 되는 잔여 전하 (있으면)
      }, … ]

    r["total_charge"] = 입력 총전하 (그대로)

⚠️ **`wbo`(Mayer 결합차수)가 없으면** M–L 판정이 거리만 쓰고 차수는 전부 `Single` 이 된다.
   `{(금속 인덱스, 원자 인덱스): w}` 로 넘긴다 (xtb `--sp` 산출물).
⚠️ **SMILES 는 우리 차수·전하를 그대로 고정해서 만든다** — RDKit 이 배위 원자에 암묵적 수소를
   붙이거나 형식전하를 다시 매기지 못하게 잠근다(`smiles.py`). `smiles_ok=False` 면 그 리간드는
   왕복 검증에 실패한 것이므로 **SMILES 를 쓰지 말고 `bonds_kekule` 을 쓴다.**
"""


# ruff: noqa: E501
from __future__ import annotations

import collections

import networkx as nx
import numpy as np

from .charge import _qfrag, kekulize, q_atom
from .config import METALS, ORD4, RCOV, THETA_HAPTIC
from .smiles import ligand_smiles, verify_roundtrip
from .connectivity import load_dint
from .eht import eht_frag_charges
from .likelihood import load_scores4
from .ml_order import load_b_ml_mayer, ml_order_scores
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
    # 🔴 M–L 차수 점수표는 **공유 헬퍼 한 곳**에서만 만든다 (워크스페이스와 동일).
    ml_sc = ml_order_scores(el, ml_raw, wbo, BML, BML_FB)
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

    # ⑦ M–M 결합 (T4 가 양 끝 금속이라 한 것) — 차수는 아직 거리 경계 미탑재라 1 로 둔다
    mets = [i for i, e in enumerate(el) if e in METALS]
    mm = {}
    for a in range(len(mets)):
        for b in range(a + 1, len(mets)):
            m1, m2 = mets[a], mets[b]
            d = float(np.linalg.norm(xyz[m1] - xyz[m2]))
            tb, wv = dbond.get(
                (el[m1], el[m2]),
                (c1g * (RCOV.get(el[m1], 1.6) + RCOV.get(el[m2], 1.6)), 0.0),
            )
            if d < tb and (wbo or {}).get((m1, m2), (wbo or {}).get((m2, m1), 1.0)) > wv:
                mm[(m1, m2)] = 1

    # ── 리간드 조각 단위로 묶는다
    NAME4 = {0: "Single", 1: "Double", 2: "Triple", 3: "Conj"}
    hapset = {(min(a, b), max(a, b)) for a, b in hap}
    coord_of = collections.defaultdict(set)  # 조각 대표 -> 배위 원자
    ligands = []
    q_all = {}
    for li, comp0 in enumerate(nx.connected_components(G)):
        comp = sorted(comp0)
        cs = set(comp)
        key = comp[0]
        b4 = {e: NAME4[v] for e, v in cls.items() if e[0] in cs}
        bk = {e: int(o) for e, o in orders.items() if e[0] in cs}
        qL = round(_qfrag(G, el, cls, cs))
        q_all[key] = qL
        coord = sorted({x for _m, x in ml_raw if x in cs})
        coord_of[key] = coord
        # 원자별 형식전하 — SMILES 에 그대로 박는다
        qat = {}
        for x in comp:
            bsum = sum(bk.get((min(x, w), max(x, w)), 1) for w in G[x])
            qat[x] = int(round(q_atom(el[x], float(bsum), G.degree(x),
                                      tuple(sorted(el[w] for w in G[x])))))
        smi, _map = ligand_smiles(el, comp, bk, qat, coord)
        ok, why = False, "SMILES 생성 실패"
        if smi:
            ok, why = verify_roundtrip(smi, el, comp, bk, qat)
        mlb_out = {}
        eta_out = {}
        for m, x in ml_raw:
            if x not in cs:
                continue
            e = (min(m, x), max(m, x))
            is_h = e in hapset
            mlb_out[(m, x)] = {
                "type": "haptic" if is_h else "sigma",
                "order": None if is_h else int(mlout.get((m, x), 0)) + 1,
            }
        for (m, fr), k in eta.items():
            if any(x in cs for x in comp if pifrag.get(x) == fr):
                eta_out[m] = k
        ligands.append({
            "index": li,
            "atoms": comp,
            "bonds_4class": b4,
            "bonds_kekule": bk,
            "smiles": smi,
            "smiles_ok": ok,
            "smiles_note": why,          # 실패 사유 (통과면 "")
            "coordinating": coord,
            "ml_bonds": mlb_out,
            "eta": eta_out,
            "charge": qL,
            "residual_charge": frag_q.get(key),
        })

    os_metal = {}
    if total_charge is not None and mets:
        num = total_charge - sum(q_all.values())
        if num % len(mets) == 0:
            os_metal = dict.fromkeys(mets, num // len(mets))

    return {
        "metals": [
            {
                "index": m,
                "element": el[m],
                "oxidation": os_metal.get(m),
                "mm_bonds": {k: v for k, v in mm.items() if m in k},
            }
            for m in mets
        ],
        "ligands": ligands,
        "total_charge": total_charge,
    }

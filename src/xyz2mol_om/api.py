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
import warnings

import networkx as nx
import numpy as np

from .charge import _qfrag, kekulize, q_atom
from .config import METALS, RCOV
from .smiles import ligand_smiles, verify_roundtrip
from .connectivity import load_dint
from .eht import eht_frag_charges
from .likelihood import load_scores4
from .pipeline import predict_T3_T5


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


MLIKE_EXTRA = {"B", "Al"}  # metal-like = 금속 ∪ {B, Al} (설계도 §3.1 (c))


def _drop_agostic(el, G, ml_raw):
    """`C–H···M` 만 뺀다 — μ-H 와 `B–H···M`(보로하이드라이드)은 진짜 3c2e 라 남긴다.

    판정  뺀다 ⟺ el[X] = H  AND  metal-like 이웃이 1개  AND  내부 이웃에 metal-like 아닌 것이 있다
    설계도 §3.0 [T4] 의 agostic 규칙. 채점기(`260831_propagation_prior_cv.py`)와 같은 식이다.
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


def predict(elements, coords, total_charge=None, wbo=None, scores4=None, dint=None):
    """`xyz` → 결합·차수·전하·산화수. 인자·반환은 모듈 docstring 참조."""
    el = list(elements)
    xyz = np.asarray(coords, dtype=float)
    if not wbo:
        # T4 거부권(`w > w_veto`)과 T8(M–L 차수)의 유일한 입력이 Mayer 결합차수다.
        # 없으면 거리 폴백으로 진행한다 — 성능이 떨어진다(모듈 docstring 참조).
        warnings.warn(
            "wbo(Mayer 결합차수)가 없다 — M–L 판정이 거리만 쓴다. "
            "T4 거부권이 꺼지고 M–L 차수는 거리 폴백(`b_ml_dist.csv`)으로 매긴다: "
            "refcode 5-fold CV 기준 M–L `Double` F1 0.698 (Mayer 판 0.732). "
            "xtb GFN2 `--sp --wbo` 로 얻어 `wbo={(금속idx, 원자idx): w}` 로 넘기면 개선된다.",
            UserWarning,
            stacklevel=2,
        )
    sc4 = scores4 if scores4 is not None else load_scores4()
    d_int, d_fb = dint if dint is not None else load_dint()

    # ① T1 — 배위자 내부 결합 (거리)
    idx = [i for i, e in enumerate(el) if e not in METALS]
    G = nx.Graph()
    G.add_nodes_from(idx)
    for ii in range(len(idx)):
        for jj in range(ii + 1, len(idx)):
            a, b = idx[ii], idx[jj]
            # 🔴 두 가드가 **먼저** 걸린다 (2026-09-03 정합 · 채점기와 동일):
            #   ① `H–H` 는 아예 후보가 아니다
            #   ② `d > 1.8·(r_cov(a)+r_cov(b))` 는 후보가 아니다 — 적합된 컷오프가 **없는**
            #      원소쌍은 전역 폴백 `d_int = 2.0542 Å` 를 쓰는데, 그것이 너무 길어
            #      **수소결합 접촉을 공유결합으로 만든다.** 실측(`DEKKEJ` · 2026-09-03):
            #      `F···H` 1.99 Å 12건이 결합으로 잡혔다(공유 `F–H` 는 0.92 Å · 정답에 없다).
            #      그 12개가 리간드 조각을 잇는 바람에 `C=O` 4개가 `Single` 로 뒤집혔다.
            if el[a] == "H" and el[b] == "H":
                continue
            d_ab = float(np.linalg.norm(xyz[a] - xyz[b]))
            if d_ab > 1.8 * (RCOV.get(el[a], 1.0) + RCOV.get(el[b], 1.0)):
                continue
            if d_ab < d_int.get(tuple(sorted((el[a], el[b]))), d_fb):
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

    # ③④⑤ T3 · M–L 차수 · T5(하프틱) · R7 — **한 함수**가 전부 낸다 (2026-09-03 통일).
    #   왜 호출자가 조립하지 않나: 예산에서 haptic·agostic 을 빼는지, M–L 차수 후보를 어떻게
    #   고르는지, T5 의 Y 후보가 무엇인지를 호출자마다 다르게 조립하다가 채점기와 **네 자리**가
    #   갈렸다(실측 2026-09-03 · 설계도 §6.5). 이제 `ml_raw` 와 `wbo` 만 넘긴다.
    q_eht = eht_frag_charges(el, xyz, G)
    cls, mlout, hap, ml_pred = predict_T3_T5(el, xyz, G, sc4, ml_raw, wbo, q_eht=q_eht)
    bml = collections.defaultdict(float)
    for _m, x in ml_pred:
        if (_m, x) not in hap:
            bml[x] += 1.0  # 출력 변환기·전하도 **같은 예산**을 쓴다 (하프틱 제외)

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
        # 🔴 η^k 는 **리간드 단위**로 센다 (2026-09-03 정합). 채점기(`len(comp ∩ hall)`)와
        #    정답지(`n_haptic_bound`)가 둘 다 리간드 단위다. π 조각별로 세면 R2·R3 로 Kekulé 가
        #    된 5원 고리가 **조각 2개로 갈려** η 가 쪼개진다 — `ZEGVIQ` 는 M–L 5개가 전부
        #    haptic 인데도 옛 셈으로 **η2** 가 나왔다(정답 η5 · 실측 2026-09-03).
        for m in {m0 for m0, x0 in hap if x0 in cs}:
            eta_out[m] = sum(1 for m0, x0 in hap if m0 == m and x0 in cs)
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

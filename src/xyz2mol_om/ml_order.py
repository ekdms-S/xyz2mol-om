"""T8 — M–L 결합 차수 (Mayer 단조 임계).

⚠️ **`ognm-bh-workspace/code/analysis/scratch/260830_fit_t10_charge.py` 에서 이관한 코드다**
(2026-09-03). 함수 본문은 **그대로** 옮겼다 — 판정 규칙을 바꾸지 않는다.
"""

# ruff: noqa: E501
from __future__ import annotations

import csv
import os
from pathlib import Path

import numpy as np

from .config import CLS, DATA, T8FORM


B_ML_CSV = DATA / "b_ml_mayer.csv"
# `lik1`/`thr` 용 — 세 형태를 한 파일에 담는다 (260901_export_t8_forms.py 산출)
B_ML_FORMS_CSV = DATA / "b_ml_t8forms.csv"

def _load_b_ml_forms(form, path=None):
    """`b_ml_t8forms.csv` → ({(M,X): ("lik",med,scl,lp) | ("thr",t1,t2,k) | ("const",c)}, 폴백).

    `thr` 점수:  s(0)=0 · s(1)=k(w−t1) · s(2)=k(w−t1)+k(w−t2),  k = 1/scl_pool
    ⇒ argmax 가 임계 규칙과 정확히 같고, 정확 해가 쓰는 증분 `s(c+1)−s(c)` 의 부호도 맞는다.
    임계가 `inf` 인 클래스는 그 원소쌍에 없는 클래스다(적합이 그렇게 골랐다).
    """
    p = Path(path) if path else B_ML_FORMS_CSV
    mdl, fb = {}, 0
    if not p.exists():
        return mdl, fb
    acc = {}
    for r in csv.DictReader(open(p)):
        if r["M"] == "*":
            fb = int(r["const_cls"])
            continue
        k = (r["M"], r["X"])
        if r["class"] == "const":
            mdl[k] = ("const", int(r["const_cls"]))
            continue
        if r["class"] == "thr":
            if form == "thr":
                t1, t2 = (float(x) for x in r["thr"].split("|"))
                mdl[k] = ("thr", t1, t2, 1.0 / float(r["scl_pool"]))
            continue
        if form == "thr":
            continue
        c = CLS[r["class"]]
        med, scl, lp = acc.setdefault(k, ({}, {}, {}))
        med[c] = float(r["med_w"])
        scl[c] = float(r["scl_pool"]) if form == "lik1" else float(r["scl"])
        lp[c] = float(r["logprior"])
    for k, (med, scl, lp) in acc.items():
        if k in mdl:
            continue
        mdl[k] = ("const", max(lp, key=lp.get)) if len(med) < 2 else ("lik", med, scl, lp)
    return mdl, fb

def load_b_ml_mayer(path=None):
    """`b_ml_mayer.csv` → ({(M,X): ("lik", med, scl, lp) | ("const", c)}, 전역폴백 c).

    `const` 행은 그 쌍이 한 클래스뿐이거나 표본이 얇아(n < 60) 우도를 못 세운 쌍이다.
    """
    if T8FORM != "lik" and path is None:
        return _load_b_ml_forms(T8FORM)
    p = Path(path) if path else B_ML_CSV
    mdl, fb = {}, 0
    if not p.exists():
        return mdl, fb
    acc = {}
    for r in csv.DictReader(open(p)):
        if r["M"] == "*":
            fb = int(r["const_cls"])
            continue
        k = (r["M"], r["X"])
        if r["class"] == "const":
            mdl[k] = ("const", int(r["const_cls"]))
            continue
        c = CLS[r["class"]]
        med, scl, lp = acc.setdefault(k, ({}, {}, {}))
        med[c], scl[c], lp[c] = float(r["med_w"]), float(r["scl"]), float(r["logprior"])
    for k, (med, scl, lp) in acc.items():
        mdl[k] = ("const", max(lp, key=lp.get)) if len(med) < 2 else ("lik", med, scl, lp)
    return mdl, fb

def predict_T8(m_el, x_el, w, model, fallback=0):
    """M–L 결합차수 클래스 (0 Single · 1 Double · 2 Triple).

    판정 ⟺  argmax_c [ −|w − med[c]| / scl[c] + lp[c] ],  w = xtb GFN2 **Mayer** 결합차수.
    그 원소쌍의 우도가 없으면 그 쌍의 `const` → 전역 폴백(Single) 순으로 내려간다.
    `w` 가 없으면(wbo 미산출) 폴백.
    """
    e = model.get((m_el, x_el))
    if e is None or w is None:
        return e[1] if (e is not None and e[0] == "const") else fallback
    if e[0] == "const":
        return e[1]
    if e[0] == "thr":
        _, t1, t2, _k = e
        return 2 if w >= t2 else (1 if w >= t1 else 0)
    _, med, scl, lp = e
    return max(med, key=lambda c: -abs(w - med[c]) / scl[c] + lp[c])


def ml_order_scores(el, ml_pairs, wbo, bml_model=None, fb=None):
    """M–L 차수 **점수표** `{(m, x): {클래스: 점수}}` — ④ 정확 해가 M–L 을 같이 최적화할 때 쓴다.

    🔴 **한 곳에서만 만든다** (2026-09-03). 예전에는 CV 스크립트·도구 대조·CRW·라이브러리가
    각자 만들었고, 어떤 곳은 아예 안 만들어(`ml_sc=None`) M–L 차수를 T8 argmax 로 **고정**했다.
    그러면 같은 입력에 다른 답이 나온다.

    `wbo` {(금속, 원자): Mayer w} · `bml_model`/`fb` = `load_b_ml_mayer()` 산출(없으면 직접 읽는다)
    ⚠️ **haptic 은 호출자가 미리 빼서 넘긴다** — 하프틱에는 차수를 안 매긴다(§3 5a).
    """
    if bml_model is None:
        bml_model, fb = load_b_ml_mayer()
    out = {}
    for m, x in ml_pairs:
        w = (wbo or {}).get((m, x), (wbo or {}).get((x, m)))
        ent = bml_model.get((el[m], el[x]))
        if ent is None or w is None or ent[0] == "const":
            c0 = ent[1] if (ent and ent[0] == "const") else fb
            out[(m, x)] = {c0: 0.0}
        elif ent[0] == "thr":
            _, t1, t2, kk = ent
            sm = {0: 0.0}
            if t1 != float("inf"):
                sm[1] = kk * (w - t1)
                if t2 != float("inf"):
                    sm[2] = sm[1] + kk * (w - t2)
            out[(m, x)] = sm
        else:
            _, med, scl, lp = ent
            out[(m, x)] = {c: -abs(w - med[c]) / scl[c] + lp[c] for c in med}
    return out


# ============================================================================
# T8 폴백 — `wbo`(Mayer) 없이 **거리만으로** M–L 차수를 매긴다 (2026-09-03).
#
# 왜: `predict(..., wbo=None)` 이면 위의 Mayer 경로가 전부 폴백(`Single`)으로 떨어진다.
#     그 자리를 원소쌍별 거리 모델로 메운다.
#
# 판정 (채택 형태 = `thr`, `T8DISTFORM=lik` 로 우도 형태로 바꿀 수 있다)
#     Double 이상 ⟺ d(M,X) <= t1(M,X)
#     Triple     ⟺ d(M,X) <= t2(M,X)      (t2 <= t1, 둘 다 Å)
#   `t = -inf` 인 클래스는 그 원소쌍에서 적합이 고르지 않은 클래스다(= 절대 예측 안 함).
#   점수 형태(④ 정확 해가 증분을 필요로 하므로):
#     s(Single)=0 · s(Double)=k*(t1-d) · s(Triple)=s(Double)+k*(t2-d),  k = 1/scl_pool
#   ⇒ argmax 가 위 임계 규칙과 정확히 같고, 증분 `s(c+1)-s(c)` 의 부호도 맞는다.
#   Mayer 판(`ml_order_scores`)과 **부호만 반대**다 — Mayer 는 클수록, 거리는 짧을수록 높은 차수.
#
# 성능 (정답지 CSD `bond_type` · 기하 `ref_xtb2` · train 표본 158,048 · refcode 5-fold **CV** ·
#       2026-09-03 · 재현 `ognm-bh-workspace/code/analysis/scratch/260903_fit_t8_dist.py`):
#       변형                     Single   Double   Triple   정확도   파라미터
#       자명 (전부 Single)        0.9815   0.0000   0.0000   0.9636        0
#       거리 단조임계 (이 코드)    0.9915   0.6976   0.6515   0.9821      464
#       Mayer 단조임계 (현행)      0.9928   0.7318   0.7230   0.9846      464
#   ⚠️ **T8 단독 측정이다** — CSD 정답 M–L 결합에 차수만 매겼다. `wbo` 가 없으면 T4(결합 유무)
#      게이트도 같이 달라지므로 파이프라인 전체 성능은 이 표로 알 수 없다.
# ============================================================================

B_ML_DIST_CSV = DATA / "b_ml_dist.csv"
T8DISTFORM = os.environ.get("T8DISTFORM", "thr")


def load_b_ml_dist(path=None, form=None):
    """`b_ml_dist.csv` → ({(M,X): ("thr",t1,t2,k) | ("lik",med,scl,lp) | ("const",c)}, 전역폴백 c).

    `form="thr"`(기본) 이면 원소쌍별 단조 임계, `form="lik"` 이면 클래스별 (중앙, MAD, 로그사전).
    `const` 행은 그 쌍이 한 클래스뿐이거나 표본이 얇아(n < 60) 모델을 못 세운 쌍이다.
    ⚠️ `med_d`·`t1`·`t2` 의 단위는 **Å** 이고, `scl_pool` 도 Å 다.
    """
    form = form or T8DISTFORM
    p = Path(path) if path else B_ML_DIST_CSV
    mdl, fb = {}, 0
    if not p.exists():
        return mdl, fb
    acc = {}
    with open(p) as fh:
        for r in csv.DictReader(fh):
            if r["M"] == "*":
                fb = int(r["const_cls"])
                continue
            k = (r["M"], r["X"])
            if r["class"] == "const":
                mdl[k] = ("const", int(r["const_cls"]))
                continue
            if r["class"] == "thr":
                if form == "thr":
                    t1, t2 = (float(x) for x in r["thr"].split("|"))
                    mdl[k] = ("thr", t1, t2, 1.0 / float(r["scl_pool"]))
                continue
            if form == "thr":
                continue
            c = CLS[r["class"]]
            med, scl, lp = acc.setdefault(k, ({}, {}, {}))
            med[c], scl[c], lp[c] = float(r["med_d"]), float(r["scl"]), float(r["logprior"])
    for k, (med, scl, lp) in acc.items():
        if k in mdl:
            continue
        mdl[k] = ("const", max(lp, key=lp.get)) if len(med) < 2 else ("lik", med, scl, lp)
    return mdl, fb


def ml_order_scores_dist(el, ml_pairs, xyz, model=None, fb=None):
    """M–L 차수 **점수표** `{(m, x): {클래스: 점수}}` — `ml_order_scores` 의 거리 폴백판.

    반환 형식은 `ml_order_scores` 와 **동일**하다(같은 자리에 꽂아 쓴다).
    `xyz` = 좌표 배열 (Å) · `model`/`fb` = `load_b_ml_dist()` 산출(없으면 직접 읽는다)
    ⚠️ **haptic 은 호출자가 미리 빼서 넘긴다** — 하프틱에는 차수를 안 매긴다(§3 5a).
    """
    if model is None:
        model, fb = load_b_ml_dist()
    out = {}
    for m, x in ml_pairs:
        d = float(np.linalg.norm(np.asarray(xyz[m], dtype=float) - np.asarray(xyz[x], dtype=float)))
        ent = model.get((el[m], el[x]))
        if ent is None or ent[0] == "const":
            c0 = ent[1] if (ent and ent[0] == "const") else fb
            out[(m, x)] = {c0: 0.0}
        elif ent[0] == "thr":
            _, t1, t2, kk = ent
            sm = {0: 0.0}
            if t1 != float("-inf"):  # `-inf` = 이 쌍에서 Double 을 안 낸다
                sm[1] = kk * (t1 - d)
                if t2 != float("-inf"):
                    sm[2] = sm[1] + kk * (t2 - d)
            out[(m, x)] = sm
        else:
            _, med, scl, lp = ent
            out[(m, x)] = {c: -abs(d - med[c]) / scl[c] + lp[c] for c in med}
    return out


def predict_T8_dist(m_el, x_el, d, model, fallback=0):
    """M–L 결합차수 클래스 (0 Single · 1 Double · 2 Triple) — 거리 폴백 단독 판정.

    판정 ⟺  `Triple` if d <= t2 else `Double` if d <= t1 else `Single`  (`thr` 형태)
             argmax_c [ -|d - med[c]| / scl[c] + lp[c] ]                 (`lik` 형태)
    그 원소쌍의 모델이 없으면 그 쌍의 `const` → 전역 폴백(Single) 순으로 내려간다.
    """
    e = model.get((m_el, x_el))
    if e is None or d is None:
        return e[1] if (e is not None and e[0] == "const") else fallback
    if e[0] == "const":
        return e[1]
    if e[0] == "thr":
        _, t1, t2, _k = e
        return 2 if d <= t2 else (1 if d <= t1 else 0)
    _, med, scl, lp = e
    return max(med, key=lambda c: -abs(d - med[c]) / scl[c] + lp[c])

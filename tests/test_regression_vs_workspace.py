"""회귀 검증 — 이관본이 **워크스페이스 원본과 같은 답**을 내는지 결합 단위로 대조한다.

원본 `ognm-bh-workspace/code/analysis/scratch/260830_fit_t10_charge.py` 의 `predict_T3_EHT` 와
이관본 `xyz2mol_om.predict_T3_EHT` 에 **같은 입력**(같은 우도·같은 M–L 점수·같은 EHT 전하)을
넣고 4클래스 배정이 한 결합도 다르지 않은지 본다.

⚠️ 우도는 **패키지에 실린 `scores4.json`**(train 26,075 적합)을 쓴다. 원본은 호출자가 그때그때
적합해 넘기므로, 같은 산출물을 양쪽에 넣어야 비교가 성립한다.

워크스페이스가 없으면 `skip` 한다 — 배포본에서는 이 테스트가 자동으로 건너뛰어진다.
`N=<개수>` 로 대조 구조 수를 정한다(기본 40).
"""

# ruff: noqa: E501
from __future__ import annotations

import collections
import csv
import os
import sys
from pathlib import Path

import networkx as nx
import numpy as np
WS = Path("/raid/abcd0105/projects/hynix/snu-t3/ognm-bh-workspace")

try:  # pytest 가 없어도 직접 실행할 수 있게 한다
    import pytest

    pytestmark = pytest.mark.skipif(not WS.exists(), reason="워크스페이스가 없다 (배포본)")
except ModuleNotFoundError:  # pragma: no cover
    pytest = None


def _load_ws():
    sys.path.insert(0, str(WS / "code/analysis/reusable"))
    sys.path.insert(0, str(WS / "code/analysis/scratch"))
    from importlib import import_module

    return import_module("260830_fit_t10_charge")


def test_same_bond_orders():
    import xyz2mol_om as X

    T10 = _load_ws()
    from tm_bond_merged import RUNS, load_merged, load_split

    n_want = int(os.environ.get("N", "40"))
    cat, home = load_merged()
    split = load_split(cat)
    train = sorted(rc for rc, v in split.items() if v == "train")

    sc4 = X.load_scores4()
    d_int, d_fb = X.load_dint()
    dbond, c1g = {}, 1.3002
    for r in csv.DictReader(open(WS / "experiments/_comparisons/2026-08-30-tm-bond-refit/d_bond.csv")):
        if r["M"] == "*":
            c1g = float(r["d_bond"])
        else:
            dbond[(r["M"], r["X"])] = (float(r["d_bond"]), float(r["w_veto"]))

    n_done = n_bond = n_diff = 0
    for rc in train:
        if n_done >= n_want:
            break
        p = home[rc] / f"ref_xtb2/xyz/{rc}.xyz"
        if not p.exists():
            continue
        el, xyz = T10.read_xyz(p)
        idx = [i for i, e in enumerate(el) if e not in T10.METALS]
        G = nx.Graph()
        G.add_nodes_from(idx)
        for ii in range(len(idx)):
            for jj in range(ii + 1, len(idx)):
                a, b = idx[ii], idx[jj]
                if float(np.linalg.norm(xyz[a] - xyz[b])) < d_int.get(tuple(sorted((el[a], el[b]))), d_fb):
                    G.add_edge(a, b)
        if not G.number_of_edges():
            continue
        bml = collections.defaultdict(float)
        for m in (i for i, e in enumerate(el) if e in T10.METALS):
            for x in idx:
                d = float(np.linalg.norm(xyz[x] - xyz[m]))
                tb, _wv = dbond.get(
                    (el[m], el[x]), (c1g * (T10.RCOV.get(el[x], 1.6) + T10.RCOV.get(el[m], 1.6)), 0.0)
                )
                if d < tb:
                    bml[x] += 1.0
        q_eht = X.eht_frag_charges(el, xyz, G)
        a_cls, _ = T10.predict_T3_EHT(el, xyz, G, sc4, dict(bml), None, q_eht, set(bml))
        b_cls, _ = X.predict_T3_EHT(el, xyz, G, sc4, dict(bml), None, q_eht, set(bml))
        assert set(a_cls) == set(b_cls), f"{rc}: 결합 집합이 다르다"
        for e in a_cls:
            n_bond += 1
            n_diff += a_cls[e] != b_cls[e]
        n_done += 1

    print(f"\n대조 구조 {n_done} · 결합 {n_bond:,} · 불일치 {n_diff}")
    assert n_done > 0, "대조한 구조가 없다"
    assert n_diff == 0, f"이관본이 원본과 다르다 — 결합 {n_diff}/{n_bond}"


if __name__ == "__main__":
    test_same_bond_orders()
    print("REGRESSION PASS")

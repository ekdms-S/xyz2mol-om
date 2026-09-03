"""Regression check - compares, bond by bond, that the ported code gives **the same answer as the
workspace original**.

The original `predict_T3_EHT` in
`ognm-bh-workspace/code/analysis/scratch/260830_fit_t10_charge.py` and the ported
`xyz2mol_om.predict_T3_EHT` are fed **the same input** (same likelihoods · same M–L scores · same
EHT charges), and we check that not a single bond gets a different 4-class assignment.

⚠️ The likelihoods come from the **`scores4.json` shipped with the package** (fit on 26,075 train
bonds). The original expects the caller to fit them on the fly, so the same artifact must be fed
to both sides for the comparison to be valid.

The test is skipped when the workspace is absent - in the distributed package it skips
automatically. `N=<count>` sets the number of structures compared (default 40).
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

try:  # so this can also be run directly, without pytest
    import pytest

    pytestmark = pytest.mark.skipif(not WS.exists(), reason="workspace not present (dist package)")
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
        assert set(a_cls) == set(b_cls), f"{rc}: bond sets differ"
        for e in a_cls:
            n_bond += 1
            n_diff += a_cls[e] != b_cls[e]
        n_done += 1

    print(f"\nstructures compared {n_done} · bonds {n_bond:,} · mismatches {n_diff}")
    assert n_done > 0, "no structures were compared"
    assert n_diff == 0, f"ported code differs from the original - {n_diff}/{n_bond} bonds"


if __name__ == "__main__":
    test_same_bond_orders()
    print("REGRESSION PASS")

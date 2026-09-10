"""🔴 Regression — an η² is written across a **Double**, not a `Single` (`ETAPI`).

A haptic M–L bond *is* the metal binding a π bond side-on, so an η² across a `Single` contradicts
itself. It is worse than untidy for a model trained on this output, which then learns that haptic
can appear on a single bond (owner: "이 표현을 보고 학습하는 모델이 … single에서도 나올 수 있는
것처럼 학습해버리기 때문").

⚠️ **η² only, and that limit is arithmetic, not a compromise.** A ring of `k` atoms has a maximum
matching of `⌊k/2⌋`, so `⌈k/2⌉` of its bonds *must* be `Single`: η⁵ Cp is 60% single and η⁶ arene
50%, and the emitted output already sits exactly on both floors. η² is the only η whose floor is
0%, and it was the only one out of line — 17% of 350 on Gold-DIGR.
"""

# ruff: noqa: E501
from __future__ import annotations

import numpy as np

from xyz2mol_om import all_fragments, predict


def _eta2_orders(el, xyz, **kw):
    """Every internal bond whose two ends are both haptic to one metal, with its Kekule order."""
    r = predict(el, xyz, **kw)
    out = []
    for fg in all_fragments(r):
        hap = {}
        for (m, x), d in fg["ml_bonds"].items():
            if d["type"] == "haptic":
                hap.setdefault(m, set()).add(x)
        for e, o in fg["bonds_kekule"].items():
            if any(len(xs) == 2 and e[0] in xs and e[1] in xs for xs in hap.values()):
                out.append((e, o))
    return out


def test_a_side_on_ethene_keeps_its_double():
    # Zeise-like: an η²-alkene on Pt. The C=C must stay a Double under the haptic pair.
    el = ["Pt", "Cl", "Cl", "Cl", "C", "C", "H", "H", "H", "H"]
    xyz = np.array([
        [0.000, 0.000, 0.000],
        [0.000, 2.310, 0.000],
        [2.310, 0.000, 0.000],
        [-2.310, 0.000, 0.000],
        [-0.700, -2.020, 0.000],
        [0.700, -2.020, 0.000],
        [-1.250, -2.300, 0.900],
        [-1.250, -2.300, -0.900],
        [1.250, -2.300, 0.900],
        [1.250, -2.300, -0.900],
    ])
    got = _eta2_orders(el, xyz, total_charge=-1)
    assert got, "η² 짝이 하나도 안 잡혔다 — 픽스처가 의도를 못 만든다"
    assert all(o >= 2 for _e, o in got), f"η² 가 Single 위에 걸렸다: {got}"

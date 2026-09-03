"""🔴 Regression - **agostic·haptic are excluded from the T3 `b_ML` budget** + **η^k is counted
per ligand**.

Real structure `ZEGVIQ` (CSD · `ref_xtb2` geometry · Mayer values are xtb GFN2 `--sp --wbo`
measurements on the same structure, hard-coded here).
Ground truth: two double bonds in the 5-membered ring (`C6=C7` · `C22=C23`) · ligand
`n_haptic_bound` = **5**.

Why the old code was wrong - it charged **every** T4 candidate `b_ML` 1.0:
  five ring atoms are attached to Cr, so all five drew on the budget, the `CAP` headroom
  vanished, and the ④ upper-bound exact solution pushed the whole ring to **`Single`** ⇒ with no
  π fragment left, T5 could not form η.
  Measured (train sample of 1,999 structures · 2026-09-03): without this exclusion **520 bonds
  (0.90%)** disagree with the §5 scorer, and accuracy at those sites was **16.5% vs 70.9%**.
  With the exclusion it drops to **71 bonds (0.12%)**.

Also, counting η **per π fragment** splits this Kekule ring into two fragments, so **η2** came
out even though all five M–L bonds are haptic. The ground truth (`n_haptic_bound`) and the
scorer are both **per ligand**.
"""

# ruff: noqa: E501
from __future__ import annotations

import numpy as np

from xyz2mol_om import predict

EL = [
    "Cr", "O", "O", "N", "C", "C", "C", "C", "C", "C", "C", "C", "C", "H", "H", "H", "H", "H",
    "H", "H", "H", "C", "C", "C", "C", "C", "O", "C", "H", "C", "H", "H", "C", "H", "H", "H",
    "H"
]
XYZ = np.array(
    [
        [    7.0098,     9.3315,     5.4596],
        [    6.9272,    11.2607,     7.7247],
        [    4.1242,     9.3299,     6.1887],
        [    8.8900,     9.3315,     4.5862],
        [    6.9897,    10.5256,     6.8297],
        [    5.2360,     9.3329,     5.8567],
        [    8.2578,    10.4760,     4.1044],
        [    7.1721,    10.0446,     3.3466],
        [    8.9514,    11.8070,     4.1722],
        [    9.9134,    12.0603,     2.9597],
        [   10.3945,    10.7208,     2.4923],
        [   11.3858,    10.0248,     3.1781],
        [    9.6176,    10.0211,     1.5728],
        [    9.8456,     9.3315,     4.9218],
        [    6.5167,    10.6723,     2.7642],
        [    8.2130,    12.6089,     4.2146],
        [    9.5331,    11.8442,     5.0969],
        [   10.7285,    12.7075,     3.2827],
        [    9.3648,    12.5619,     2.1622],
        [   12.0832,    10.5629,     3.8049],
        [    8.9288,    10.5580,     0.9372],
        [    6.9892,     8.1380,     6.8301],
        [    8.2578,     8.1870,     4.1044],
        [    7.1721,     8.6184,     3.3466],
        [   11.3857,     8.6382,     3.1781],
        [    9.6177,     8.6419,     1.5727],
        [    6.9257,     7.4024,     7.7247],
        [    8.9515,     6.8560,     4.1722],
        [    6.5168,     7.9907,     2.7642],
        [   10.3946,     7.9422,     2.4922],
        [   12.0833,     8.1001,     3.8048],
        [    8.9289,     8.1049,     0.9371],
        [    9.9134,     6.6027,     2.9596],
        [    8.2131,     6.0540,     4.2146],
        [    9.5332,     6.8188,     5.0969],
        [   10.7286,     5.9554,     3.2827],
        [    9.3649,     6.1010,     2.1622],
    ]
)
WBO = {
    (0, 1): 0.3303,
    (0, 2): 0.3419,
    (0, 3): 0.2417,
    (0, 4): 1.2402,
    (0, 5): 1.2647,
    (0, 6): 0.2660,
    (0, 7): 0.2300,
    (0, 8): 0.0000,
    (0, 13): 0.0000,
    (0, 14): 0.0000,
    (0, 21): 1.2404,
    (0, 22): 0.2661,
    (0, 23): 0.2299,
    (0, 26): 0.3303,
    (0, 27): 0.0000,
    (0, 28): 0.0000,
}


def _ring_ligand(r):
    return max(r["ligands"], key=lambda lg: len(lg["bonds_4class"]))


def test_ring_keeps_double_bonds():
    """The ring is not flattened to all-`Single` - two double bonds at the same positions as
    the CSD ground truth."""
    lg = _ring_ligand(predict(EL, XYZ, wbo=WBO))
    b = lg["bonds_4class"]
    assert b[(6, 7)] == "Double", b[(6, 7)]
    assert b[(22, 23)] == "Double", b[(22, 23)]
    for e in ((3, 6), (3, 22), (7, 23)):
        assert b[e] == "Single", (e, b[e])


def test_eta_is_per_ligand():
    """All five M–L bonds are haptic and η = 5 (counting per π fragment would give 2)."""
    lg = _ring_ligand(predict(EL, XYZ, wbo=WBO))
    types = [d["type"] for d in lg["ml_bonds"].values()]
    assert types == ["haptic"] * 5, types
    assert lg["eta"] == {0: 5}, lg["eta"]

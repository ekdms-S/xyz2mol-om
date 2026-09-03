"""🔴 Regression - **R7 really fires**: the S of an η⁵-thiophene (thienyl) ring turns haptic.

Real structure `HOQNOQ` - CSD `chemical_name` = *"(μ2-Diphenylphosphido)-(μ2-**η5**-thien-2-yl)-
hexacarbonyl-di-manganese"* ⇒ the **ground truth is η⁵**. Geometry is `ref_xtb2` · Mayer bond
orders are xtb GFN2 `--sp --wbo` measurements on the same structure, **hard-coded** here (so this
runs without the workspace).

What is measured (`docs/PIPELINE.md` 5′):

  R7 off :  haptic = {4 Mn–C}                   η⁴   ← S drops out, absent from the π fragment
  R7 on  :  haptic = {4 Mn–C, **Mn–S**}         η⁵   ← ground truth

Why S falls out of the π fragment: R2 clears `Conj` from both bonds of `S` (deg 2), so only three
`Conj` bonds remain in the 5-membered ring and **S alone touches no `Double`·`Triple`·`Conj`.**

🔴 **T3 does not change by a single bit** - the `bonds_4class` identity check below pins it down.
"""

# ruff: noqa: E501
from __future__ import annotations

import numpy as np

import xyz2mol_om.pipeline as pipeline
from xyz2mol_om import predict

# `HOQNOQ` (from CSD · `ref_xtb2` geometry)
EL = [
    "Mn", "Mn", "P", "S", "C", "C", "H", "C", "H", "C", "H", "C", "C", "H", "C", "H", "C",
    "H", "C", "H", "C", "H", "C", "C", "H", "C", "H", "C", "H", "C", "H", "C", "H", "C",
    "O", "C", "O", "C", "O", "C", "O", "C", "O", "O", "C"
]
XYZ = np.array(
    [
        [   5.0977,    0.2869,    1.7015],
        [   5.0006,    3.8564,    1.0655],
        [   4.0454,    1.8077,    0.3411],
        [   4.4852,    1.5687,    3.5050],
        [   5.6048,    2.3238,    2.3664],
        [   6.7457,    1.5389,    2.3113],
        [   7.6350,    1.8364,    1.7674],
        [   6.7153,    0.3510,    3.0896],
        [   7.5685,   -0.3004,    3.2275],
        [   5.5135,    0.1909,    3.7524],
        [   5.2726,   -0.5212,    4.5271],
        [   4.1499,    1.3970,   -1.4278],
        [   4.3863,    2.3293,   -2.4298],
        [   4.5715,    3.3673,   -2.1822],
        [   4.3956,    1.9266,   -3.7548],
        [   4.5875,    2.6517,   -4.5319],
        [   4.1654,    0.6016,   -4.0802],
        [   4.1808,    0.2916,   -5.1144],
        [   3.9207,   -0.3297,   -3.0810],
        [   3.7430,   -1.3633,   -3.3362],
        [   3.9121,    0.0619,   -1.7572],
        [   3.7365,   -0.6502,   -0.9512],
        [   2.2328,    1.8634,    0.5012],
        [   1.5405,    2.7402,   -0.3342],
        [   2.0848,    3.3212,   -1.0685],
        [   0.1705,    2.8632,   -0.2211],
        [  -0.3627,    3.5440,   -0.8682],
        [  -0.5181,    2.1197,    0.7260],
        [  -1.5890,    2.2235,    0.8174],
        [   0.1662,    1.2469,    1.5541],
        [  -0.3693,    0.6656,    2.2901],
        [   1.5397,    1.1168,    1.4461],
        [   2.0757,    0.4293,    2.0887],
        [   6.0435,   -0.4592,    0.3935],
        [   6.7185,   -0.8973,   -0.4304],
        [   3.9763,   -1.0932,    1.6335],
        [   3.2814,   -2.0175,    1.6109],
        [   4.2978,    4.9785,   -0.1326],
        [   3.8078,    5.7060,   -0.8708],
        [   5.9961,    5.0490,    1.9621],
        [   6.6505,    5.7674,    2.5729],
        [   6.2968,    3.3136,   -0.0465],
        [   7.0744,    2.8816,   -0.7637],
        [   2.5552,    4.1882,    2.6460],
        [   3.5281,    4.0809,    2.0536],
    ]
)
# xtb GFN2 `--sp --wbo` measured - M–L pairs only (`predict` uses them only for M–L·M–M)
WBO = {
    (0, 1): 0.0, (0, 2): 0.9164, (0, 3): 0.322, (0, 4): 0.1982, (0, 5): 0.196, (0, 6): 0.0,
    (0, 7): 0.2669, (0, 8): 0.0, (0, 9): 0.3179, (0, 10): 0.0, (0, 11): 0.0, (0, 20): 0.0,
    (0, 22): 0.0, (0, 31): 0.0, (0, 32): 0.0, (0, 33): 1.4583, (0, 34): 0.3575, (0, 35):
    1.5298, (0, 36): 0.3796, (0, 41): 0.0, (1, 0): 0.0, (1, 2): 0.743, (1, 3): 0.0, (1, 4):
    0.7143, (1, 5): 0.0, (1, 11): 0.0, (1, 12): 0.0, (1, 22): 0.0, (1, 37): 1.2449, (1, 38):
    0.2646, (1, 39): 1.2972, (1, 40): 0.2691, (1, 41): 1.1211, (1, 42): 0.2115, (1, 43):
    0.2228, (1, 44): 1.1469
}
WBO.update({(j, i): w for (i, j), w in list(WBO.items())})

MN = 0          # the manganese the thienyl is haptically bound to
S_IDX = 3       # the ring S - the atom R7 restores
RING_C = (4, 5, 7, 9)


def _thienyl(r):
    (L,) = [x for x in r["ligands"] if S_IDX in x["atoms"]]
    return L


def _run(r7):
    # the R7 flag lives in `pipeline`, where the unified function is (since the 2026-09-03 merge)
    old = pipeline.R7RING
    pipeline.R7RING = r7
    try:
        return predict(EL, XYZ, total_charge=None, wbo=WBO)
    finally:
        pipeline.R7RING = old


def test_r7_recovers_eta5_thiophene():
    """R7 on (the shipped default) gives η⁵ · off gives η⁴."""
    on, off = _thienyl(_run(True)), _thienyl(_run(False))

    hap_on = sorted(k for k, v in on["ml_bonds"].items() if v["type"] == "haptic")
    hap_off = sorted(k for k, v in off["ml_bonds"].items() if v["type"] == "haptic")
    assert hap_off == [(MN, c) for c in RING_C], f"R7 off haptic set unexpected - {hap_off}"
    assert hap_on == sorted([(MN, S_IDX)] + [(MN, c) for c in RING_C]), (
        f"R7 is on but Mn–S is not haptic - {hap_on}"
    )
    assert off["eta"] == {MN: 4}, f"R7 off η must be 4 - {off['eta']}"
    assert on["eta"] == {MN: 5}, f"R7 on η must be 5 (CSD ground truth η⁵) - {on['eta']}"


def test_r7_does_not_touch_bond_orders():
    """🔴 R7 waives **only T5's π-fragment membership condition** - the T3 4-class assignment
    must stay identical."""
    on, off = _run(True), _run(False)
    for a, b in zip(on["ligands"], off["ligands"]):
        assert a["bonds_4class"] == b["bonds_4class"], f"R7 changed T3 - fragment {a['index']}"
        assert a["bonds_kekule"] == b["bonds_kekule"], f"R7 changed Kekule - fragment {a['index']}"
        assert a["charge"] == b["charge"], f"R7 changed the ligand charge - fragment {a['index']}"


if __name__ == "__main__":
    test_r7_recovers_eta5_thiophene()
    test_r7_does_not_touch_bond_orders()
    print("R7 PASS")

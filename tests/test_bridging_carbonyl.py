"""mu-CO — a bridging carbonyl is a **ketonic** bridge: `C=O` plus two 2c2e M-C bonds.

What is pinned: `b_use` counts pass-1 bond **orders**, but a bridging pi acceptor gets the
`piacc_relief` correction (`config.PIACC_BRIDGE`), so the C of a bridging CO scores
`3 - 1 + 2 = 4 = CAP(C)` and is tagged **`dative`**, keeping `C=O`. Turning the correction off
reproduces the 2026-09-06 branch answer (`3c2e` + `C#O`), which the CSD labels contradict:
mu2 CO is `Double` 311/318, terminal CO is `Triple` 19,808/21,318 (train, distance-filtered).

The charge follows the **mu-L** convention (`charge.q_atom`, the `piacc` branch): only the
first M-C bond is a pair donated by the ligand, the second is the metal's backbonding pair, so
`q_L = 0` and this molecule is Co(0). Measured against CSD `chemical_name` roman numerals:
mu-CO structures **0/21 -> 17/21**, control (no mu-CO) 353/400 = 0.882.

⚠️ Bond order and charge come from **different** formalisms on purpose - ketonic (mu-X2) for
the bond, mu-L for the charge - because that is what the two reference labels use. See
PIPELINE.md.

Structure: Co2(CO)8, the C2v bridged isomer, relaxed with GFN2-xTB (Co-Co 2.514 A · experiment
2.52). **Not a CSD structure** — it is embedded here so the test needs no data file. `wbo` is
not passed; for this molecule the distance fallback gives the same answer.
"""

from xyz2mol_om import pipeline, predict

# Co2(CO)8 (C2v, bridged) · GFN2-xTB optimized
CO2CO8 = [
    ("Co", -1.256796, 0.000060, -0.054383),
    ("Co", 1.256796, 0.000059, -0.054383),
    ("C", 0.000000, 1.332100, 0.550107),
    ("O", 0.000000, 2.381057, 1.045921),
    ("C", -0.000000, -1.331906, 0.550520),
    ("O", -0.000000, -2.380515, 1.046952),
    ("C", -2.190664, 0.000948, 1.537419),
    ("O", -2.853588, -0.000706, 2.459116),
    ("C", -1.917796, -1.322133, -1.069178),
    ("O", -2.350789, -2.163700, -1.699772),
    ("C", -1.917998, 1.320387, -1.071268),
    ("O", -2.350524, 2.164777, -1.698685),
    ("C", 2.190664, 0.000948, 1.537419),
    ("O", 2.853588, -0.000705, 2.459116),
    ("C", 1.917796, -1.322134, -1.069177),
    ("O", 2.350789, -2.163701, -1.699772),
    ("C", 1.917998, 1.320387, -1.071268),
    ("O", 2.350524, 2.164777, -1.698685),
]


def _run():
    el = [a[0] for a in CO2CO8]
    xyz = [list(a[1:]) for a in CO2CO8]
    return predict(el, xyz, total_charge=0)


def _bridging(r):
    return [lg for lg in r["ligands"] if len({m for m, _ in lg["ml_bonds"]}) >= 2]


def test_mu_co_is_tagged_dative():
    lgs = _bridging(_run())
    assert len(lgs) == 2, "Co2(CO)8 has two bridging carbonyls"
    for lg in lgs:
        assert {d["bridge"] for d in lg["ml_bonds"].values()} == {"dative"}
        assert {d["type"] for d in lg["ml_bonds"].values()} == {"bridge"}


def test_mu_co_is_a_ketonic_bridge():
    for lg in _bridging(_run()):
        assert list(lg["bonds_4class"].values()) == ["Double"], "CSD: mu2 CO is `Double` 311/318"


def test_terminal_co_is_unaffected():
    r = _run()
    term = [lg for lg in r["ligands"] if len({m for m, _ in lg["ml_bonds"]}) == 1]
    assert len(term) == 6
    for lg in term:
        assert list(lg["bonds_4class"].values()) == ["Triple"]
        assert lg["charge"] == 0
        assert all(d["bridge"] is None for d in lg["ml_bonds"].values())


def test_oxidation_state_is_zero():
    r = _run()
    assert [m["oxidation"] for m in r["metals"]] == [0, 0], "Co2(CO)8 is Co(0)"
    for lg in _bridging(r):
        assert lg["charge"] == 0, "a bridging CO is a neutral 2e donor, like a terminal one"
        assert lg["smiles"] == "O=[C:1]", "ketonic C=O, and the carbon carries no formal charge"


def test_the_correction_is_what_keeps_the_double(monkeypatch):
    """`PIACC_BRIDGE=0` restores the uncorrected branch answer, so the knob stays A/B-measurable.

    Without the correction `b_use = 3 + 2 = 5 > CAP(C)` tags the carbon `3c2e`, the 3c2e budget
    (`BML3C_COST` 1.0) frees one unit of headroom, and the distance likelihood - which prefers
    `Triple` at 1.166 A - takes it.
    """
    monkeypatch.setattr(pipeline, "PIACC_BRIDGE", False)
    for lg in _bridging(_run()):
        assert {d["bridge"] for d in lg["ml_bonds"].values()} == {"3c2e"}
        assert list(lg["bonds_4class"].values()) == ["Triple"]

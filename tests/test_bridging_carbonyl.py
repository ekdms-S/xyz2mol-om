"""mu-CO — the bridging carbonyl must come out as `3c2e` with an intact `C≡O`.

Why this test exists (2026-09-06). `bridge_tags` used to measure `deg` as the **number** of
internal bonds, so the C of a bridging CO scored `1 + 2 = 3 <= 4` and was tagged `dative`. And
even with the tag fixed nothing moved, because the tag reached only `BMLSKIP3C` (off, rejected).
Two changes together are what fix it — `deg` counts pass-1 bond **orders**, and an atom taking
part in a 3c2e spends `BML3C_COST` (1.0) in total instead of one unit per M-L bond. This pins
both, and the last test pins the old behaviour so the change stays A/B-measurable.

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


def test_mu_co_is_tagged_3c2e():
    lgs = _bridging(_run())
    assert len(lgs) == 2, "Co2(CO)8 has two bridging carbonyls"
    for lg in lgs:
        assert {d["bridge"] for d in lg["ml_bonds"].values()} == {"3c2e"}
        assert {d["type"] for d in lg["ml_bonds"].values()} == {"bridge"}


def test_mu_co_keeps_its_triple_bond_and_stays_neutral():
    for lg in _bridging(_run()):
        assert list(lg["bonds_4class"].values()) == ["Triple"]
        assert lg["charge"] == 0, "a bridging CO is a neutral 2e donor, like a terminal one"
        assert lg["smiles"] == "[O+]#[C-:1]", "identical to the terminal CO ligand SMILES"


def test_oxidation_state_is_zero():
    assert [m["oxidation"] for m in _run()["metals"]] == [0, 0], "Co2(CO)8 is Co(0)"


def test_terminal_co_is_unaffected():
    r = _run()
    term = [lg for lg in r["ligands"] if len({m for m, _ in lg["ml_bonds"]}) == 1]
    assert len(term) == 6
    for lg in term:
        assert list(lg["bonds_4class"].values()) == ["Triple"]
        assert lg["charge"] == 0
        assert all(d["bridge"] is None for d in lg["ml_bonds"].values())


def test_per_bond_cost_reproduces_the_old_answer(monkeypatch):
    """`BML3C_COST < 0` = one unit per M-L bond, the behaviour before 2026-09-06.

    The tag is `3c2e` either way — this is the evidence that the tag alone changes nothing and
    that the budget is what moves the result.
    """
    monkeypatch.setattr(pipeline, "BML3C_COST", -1.0)
    r = _run()
    for lg in _bridging(r):
        assert {d["bridge"] for d in lg["ml_bonds"].values()} == {"3c2e"}
        assert list(lg["bonds_4class"].values()) == ["Double"]
        assert lg["charge"] == -2
    assert [m["oxidation"] for m in r["metals"]] == [2, 2]

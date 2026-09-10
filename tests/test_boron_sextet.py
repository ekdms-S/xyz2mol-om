"""🔴 Regression — boron fills a **sextet**, so trivalent boron is neutral, not −2.

The default formal charge `q = v + b − 8` assumes the octet (`lp = 4 − b`). Boron carries no lone
pair until it reaches four bonds, so its quota is 6. Under the octet formula every trivalent
boron — a boronic acid, a boronate ester, B₂pin₂ — came out `[B-2]`, and that −2 is not a local
label: it is a real charge on the fragment, so a metal-bearing molecule pays for it with **+2 on
the oxidation state**. Measured on Gold-DIGR: all 55 trivalent borons in a 1,200-reaction sample
were wrong, and `10.1039_D2CY01506D__46_TS8b` put Pd at +4 instead of +2 because of it.

⚠️ Four-coordinate boron is **unchanged** — `3 + 4 − 8` and `3 − 4` are both −1 — which is what
keeps BF₄⁻ and the `[N+]=[B-]` adduct where they were.
"""

# ruff: noqa: E501
from __future__ import annotations

from xyz2mol_om.charge import q_atom


def test_trivalent_boron_is_neutral():
    assert q_atom("B", 3.0) == 0  # B(OH)₃ · a boronic ester · B₂pin₂ — the octet formula said −2


def test_four_coordinate_boron_is_still_minus_one():
    assert q_atom("B", 4.0) == -1  # BF₄⁻ · the N→B adduct — unchanged by the sextet quota


def test_a_boryl_ligand_is_minus_one_after_the_ionic_cut():
    # M–BR₂ keeps its two internal bonds and one lone pair once the M–L bond is cut away
    assert q_atom("B", 2.0) == -1


def test_the_sextet_only_touches_boron():
    assert q_atom("C", 3.0) == -1  # a carbanion — the octet formula, untouched
    assert q_atom("N", 3.0) == 0

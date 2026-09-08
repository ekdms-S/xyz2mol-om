"""`pi_suppressed` — the ⑥ output wrote `Single` between two anionic atoms where the ③ distance
likelihood preferred `Double`.

Why it is reported: a `Single` there costs the fragment **two** extra negative charges (a lone
pair on each end instead of the π bond), and with the fragment on a metal that lands on the metal
as **oxidation state +2**. Measured on Gold-DIGR (`dev/analysis/scratch/260909_golddigr_os_out_of_range.py`).
"""

# ruff: noqa: E501
from __future__ import annotations

import json

import numpy as np

from xyz2mol_om import all_fragments, predict
from xyz2mol_om.charge import pi_suppressed

from test_api_smoke import EL, WBO, XYZ


def test_flags_a_single_between_two_anions_the_likelihood_wanted_double():
    bk = {(0, 1): 1}
    assert pi_suppressed(bk, {0: -1, 1: -1}, {(0, 1): 3.4}) == [(0, 1)]


def test_silent_when_any_of_the_three_conditions_fails():
    q, w = {0: -1, 1: -1}, {(0, 1): 3.4}
    assert pi_suppressed({(0, 1): 2}, q, w) == []            # already Double
    assert pi_suppressed({(0, 1): 1}, {0: -1, 1: 0}, w) == []  # one end neutral
    assert pi_suppressed({(0, 1): 1}, {0: 1, 1: 1}, w) == []   # both ends positive
    assert pi_suppressed({(0, 1): 1}, q, {(0, 1): -3.4}) == []  # likelihood prefers Single
    assert pi_suppressed({(0, 1): 1}, q, {}) == []             # no likelihood for this pair


def test_predict_reports_it_per_fragment():
    r = predict(EL, XYZ, total_charge=-1, wbo=WBO)
    for fr in all_fragments(r):
        assert fr["pi_suppressed"] == [], fr["pi_suppressed"]


def test_survives_the_json_round_trip_as_tuples():
    r = predict(EL, XYZ, total_charge=-1, wbo=WBO)
    fr = all_fragments(r)[0]
    fr["pi_suppressed"] = [(1, 2)]           # the Mo fixture has none - plant one
    from xyz2mol_om import from_jsonable, to_jsonable
    back = from_jsonable(json.loads(json.dumps(to_jsonable(r))))
    got = back["molecules"][0]["fragments"][0]["pi_suppressed"]
    assert got == [(1, 2)], got              # tuples, so they key into `bonds_kekule`


def test_the_margin_it_reads_is_the_likelihood_not_the_matching_penalty():
    """④ subtracts `1e6` from `w` on the `Conj` edges it granted headroom against, to hold the ⑥
    matching to its promise. That term is a constraint, not a likelihood, and it lands on exactly
    the edges a π-suppression report is about — so the report reads `w_raw_out`, filled before it.
    """
    import networkx as nx

    from xyz2mol_om import load_scores4
    from xyz2mol_om.rules import predict_T3_EHT

    el = ["C", "C", "H", "H", "H", "H"]                       # ethene
    xyz = np.array([[0.000, 0.000, 0.000], [1.331, 0.000, 0.000],
                    [-0.575, 0.927, 0.000], [-0.575, -0.927, 0.000],
                    [1.906, 0.927, 0.000], [1.906, -0.927, 0.000]])
    G = nx.Graph()
    G.add_edges_from([(0, 1), (0, 2), (0, 3), (1, 4), (1, 5)])
    w, w_raw = {}, {}
    predict_T3_EHT(el, xyz, G, load_scores4(), {}, None, None, None, None,
                   w_out=w, w_raw_out=w_raw)
    assert w_raw, "w_raw_out must be filled"
    assert set(w_raw) == set(w)
    # the penalty only ever subtracts, so the raw margin is never below the one ⑥ sees
    assert all(w_raw[e] >= w[e] for e in w_raw)
    assert all(v > -1e5 for v in w_raw.values()), "a `−1e6` promise leaked into the raw margin"


def test_neutralising_one_end_silences_it():
    """Why the report is built after radical placement: putting the unpaired electron on an atom
    returns that atom's charge to 0, and then the bond is no longer a two-anion bond. Computed
    before that block, the flag would describe charges the result does not carry."""
    bk, w = {(0, 1): 1}, {(0, 1): 3.4}
    assert pi_suppressed(bk, {0: -1, 1: -1}, w) == [(0, 1)]   # before placement
    assert pi_suppressed(bk, {0: 0, 1: -1}, w) == []          # after it landed on atom 0


def test_an_element_pair_with_no_fitted_double_is_never_flagged():
    """`As–C` and `B–B` are shipped in `scores4` with only `Single` and `Conj` fitted. Reading the
    missing `Double` as 0.0 turns `0 − score[Single]` positive on nearly every such bond, so the
    raw margin must skip those edges entirely — otherwise a caller told to discard flagged
    structures throws away valid ones."""
    import networkx as nx

    from xyz2mol_om import load_scores4
    from xyz2mol_om.rules import predict_T3_EHT

    el = ["As", "C", "H", "H", "H"]
    xyz = np.array([[0.0, 0.0, 0.0], [2.04, 0.0, 0.0], [2.4, 1.03, 0.0],
                    [2.4, -0.51, 0.89], [2.4, -0.51, -0.89]])
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2), (1, 3), (1, 4)])
    assert 1 not in load_scores4()[("As", "C")][0], "fixture assumes As-C has no Double fitted"
    w_raw = {}
    predict_T3_EHT(el, xyz, G, load_scores4(), {}, None, None, None, None, w_raw_out=w_raw)
    assert (0, 1) not in w_raw
    assert pi_suppressed({(0, 1): 1}, {0: -1, 1: -1}, w_raw) == []

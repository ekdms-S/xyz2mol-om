"""`Conj` candidate rules — rule A · R2 · R3 · R4 (R5 lives in the pipeline).

⚠️ **Ported from `ognm-bh-workspace/code/analysis/scratch/260830_fit_t10_charge.py`**
(2026-09-03). Function bodies were moved **verbatim** — the decision rules are unchanged.
"""

# ruff: noqa: E501
from __future__ import annotations

import networkx as nx
import numpy as np

from .config import CAP, R3MODE, R3RING, R4RING, RULEA, TAU_P, _LP_DEG
from .geometry import plane_rms


def rule_a_ok(n):
    return {"ge5": n >= 5, "eq6": n == 6, "off": False}[RULEA]

def lp_donor(elem, deg):
    """Is the atom's neutral σ skeleton already full = does it join π only via a lone pair?"""
    d = _LP_DEG.get(elem)
    return d is not None and deg >= d

def conj_forbidden(G, el, q_frag=None, xyz=None):
    """Set of bonds that cannot be `Conj`. `q_frag` = {atom: EHT charge of its fragment}
    (pyridinium exemption)."""
    bad = set()
    donors = set()
    for x in G.nodes():
        if not lp_donor(el[x], G.degree(x)):
            continue
        if el[x] == "N" and q_frag is not None and q_frag.get(x, 0) > 0:
            continue  # pyridinium N⁺ exception
        donors.add(x)
        for y in G[x]:
            bad.add((min(x, y), max(x, y)))
    if R4RING:  # R4 — a 4n all-carbon ring (4-/8-membered) that is **non-planar** is Kekule
        for r_ in nx.cycle_basis(G):
            if len(r_) in (4, 8) and all(el[x] == "C" for x in r_):
                if xyz is None or plane_rms(xyz[np.array(r_)]) > TAU_P:
                    for a, b in zip(r_, r_[1:] + r_[:1]):
                        bad.add((min(a, b), max(a, b)))
    if R3RING and donors:  # R3 — a 5-membered ring containing such an atom is Kekule as a whole
        for r_ in nx.cycle_basis(G):
            if len(r_) != 5:
                continue
            din = [x for x in r_ if x in donors]
            if not din:
                continue
            if R3MODE == "N" and not all(el[x] == "N" for x in din):
                continue
            if R3MODE == "mono" and not (
                len(din) == 1 and all(el[x] == "C" for x in r_ if x not in din)
            ):
                continue
            for a, b in zip(r_, r_[1:] + r_[:1]):
                bad.add((min(a, b), max(a, b)))
    return bad

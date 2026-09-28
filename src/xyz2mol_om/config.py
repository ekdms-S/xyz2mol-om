"""Element tables and rule settings — **these values are the pipeline**.

The rules that read these numbers are stated in `docs/PIPELINE.md`; this file holds the numbers.
Every setting is an environment variable read once, at import, so a run can be reproduced from the
command line without editing the package. A boolean setting is on when its variable is `1` and off
for any other value.
"""

# ruff: noqa: E501
from __future__ import annotations

import os
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data"  # fitted tables shipped with the package

# ═══ Element tables ═══════════════════════════════════════════════════════════════════════════

CLS = {"Single": 0, "Double": 1, "Triple": 2}  # bond-class name → class index
ORD4 = [1.0, 2.0, 3.0, 1.5]  # 0 Single · 1 Double · 2 Triple · 3 Conj

VAL = {  # valence electrons
    "H": 1, "B": 3, "C": 4, "N": 5, "O": 6, "F": 7, "Si": 4, "P": 5,
    "S": 6, "Cl": 7, "As": 5, "Se": 6, "Br": 7, "Te": 6, "I": 7,
}  # fmt: skip
FULL = {"H": 2}  # filled-shell quota. 8 by default, 2 only for H
RCOV = {  # covalent radii (Å, Cordero) — fallback distance threshold for a pair not in the fit
    "H": 0.31, "B": 0.84, "C": 0.76, "N": 0.71, "O": 0.66, "F": 0.57, "Si": 1.11, "P": 1.07,
    "S": 1.05, "Cl": 1.02, "As": 1.19, "Se": 1.20, "Br": 1.20, "Te": 1.38, "I": 1.39,
    # Li, Na, Ge are neither centres nor in the fitted ligand tables; listed so they do not
    #   take the default radius of an unlisted element.
    "Li": 1.28, "Na": 1.66, "Ge": 1.20,
    # centres (Cordero 2008; Mn · Fe · Co low-spin). Without them every metal took the 1.6 Å
    #   default, so an unfitted M–M pair was cut at 1.30 × 3.2 = 4.16 Å.
    "Mg": 1.41, "Al": 1.21, "Sc": 1.70, "Ti": 1.60, "V": 1.53, "Cr": 1.39, "Mn": 1.39,
    "Fe": 1.32, "Co": 1.26, "Ni": 1.24, "Cu": 1.32, "Zn": 1.22, "Ga": 1.22, "Y": 1.90,
    "Zr": 1.75, "Nb": 1.64, "Mo": 1.54, "Ru": 1.46, "Rh": 1.42, "Pd": 1.39, "Ag": 1.45,
    "In": 1.42, "Sn": 1.39, "La": 2.07, "Ce": 2.04, "Hf": 1.75, "Ta": 1.70, "W": 1.62,
    "Re": 1.51, "Os": 1.44, "Ir": 1.41, "Pt": 1.36, "Au": 1.36, "Pb": 1.46,
}  # fmt: skip

CAP = {  # valence ceiling used by ④ — `b_int + b_ML <= CAP`
    "H": 1, "B": 4, "C": 4, "N": 5, "O": 4, "F": 4, "Si": 6, "P": 6,
    "S": 6, "Cl": 7, "As": 6, "Se": 6, "Br": 6, "Te": 6, "I": 6,
}  # fmt: skip
# `CAPSET` — which ceiling `CAP` holds. `octet` (default) = the table above, an octet ceiling
#   (period-2 elements at 4, pentavalent N allowed) · `mid` = O 3, N 4 · `tight` = O 3, N 4, F 2,
#   i.e. `b_max = 8 - v + q` with cations allowed. Any other value stops the program.
_CAPSET = os.environ.get("CAPSET", "octet")
if _CAPSET == "tight":
    CAP = dict(CAP, O=3, N=4, F=2)
elif _CAPSET == "mid":
    CAP = dict(CAP, O=3, N=4)
elif _CAPSET != "octet":
    raise SystemExit(f"CAPSET={_CAPSET!r} must be one of octet|mid|tight")

VALENCE_3C = {"H": 1, "C": 4, "Si": 4, "B": 3}  # closed-shell budget of a bridging atom (T7)
HUCKEL = [2, 6, 10, 14, 18]  # 4n+2 aromatic electron counts

TAU_P = 0.05  # ring planarity tolerance (Å, out-of-plane rms) — Rule A and R4

# ═══ Centres ══════════════════════════════════════════════════════════════════════════════════

METALS = set(  # the centre elements (see `centers`)
    "Ti Zr Hf Nb Ta V La Sc Y Ce Cr Mo W Mn Re Fe Ru Os Co Rh Ir Ni Pd Pt "
    "Cu Ag Au Zn Al Ga In Sn Pb Mg".split()
)
# `METALS_HARD` — alias of `METALS`, exported by the package.
METALS_HARD = METALS
# `MLIKE_EXTRA` — metal-**like** for the T7 bridge rule only (`docs/PIPELINE.md` 5″). These are
#   not centres; they count toward `n_center` as *internal* neighbours so that the H of a `B–H–B`
#   is recognised as a bridge with no metal in sight. Lives here because both `rules.pipeline`
#   (which tags) and `charge.formal` (which prices the tag) need it, and `charge` cannot import
#   from `rules`.
MLIKE_EXTRA = {"B", "Al"}

def centers(el):
    """The centre atoms — every atom whose element is in `METALS`. No conditions.

    Boron is not a centre: like Si it is a ligand atom everywhere, so a `B–H` bond has the same
    shape in `B₂H₆` as inside a carborane. The bridging H of a borane is tagged `3c2e` through
    the `MLIKE_EXTRA` internal-neighbour term of T7 (`rules/pipeline.bridge_tags`).
    """
    return {i for i, e in enumerate(el) if e in METALS}

# ═══ ③ distance likelihood ════════════════════════════════════════════════════════════════════

# `LPCOND` — take the class prior from the endpoints' internal-degree cell instead of the element
#   pair alone (a `C–O` at `deg(C)=3, deg(O)=1` is a carbonyl, which the pair prior underrates).
#   Default on.
LPCOND = os.environ.get("LPCOND", "1") == "1"
# `LPCOND_NOCONJ` — with `LPCOND`, keep `Conj` on the element-pair prior (the `C–C` deg-3/3 cell
#   would otherwise pull `Double` into `Conj`). Default on.
LPCOND_NOCONJ = os.environ.get("LPCOND_NOCONJ", "1") == "1"
# `LPCOND_NMIN` — samples a degree cell needs for its own prior when `fit_scores4` builds the
#   table; smaller cells fall back to the element-pair prior. Integer, default 300.
#   ⚠️ Read only when fitting: the shipped `data/scores4.json` was fitted at 300.
LPCOND_NMIN = int(os.environ.get("LPCOND_NMIN", "300"))
# `LPA` — prior temperature: `score = distance term + LPA·ln P(c)`. Float, default 0.8; 0 drops
#   the prior. **The one fitted parameter in the rules.**
LPA = float(os.environ.get("LPA", "0.8"))
# `LNORM` — add the Laplace normalisation term `-ln(2·scl)` to the likelihood: `0` off (default) ·
#   `1` on · `2` on except for `Conj`. Any other value acts as `0`.
LNORM = os.environ.get("LNORM", "0")
LNORM_ON = LNORM in ("1", "2")
LNORM_SKIP_CONJ = LNORM == "2"
# `USE_ROP` — add the extended-Hückel overlap population as a second likelihood dimension,
#   weighted by `ROPW` (float, default 1.0). Default off; needs a precomputed overlap cache.
USE_ROP = os.environ.get("USE_ROP", "0") == "1"
ROPW = float(os.environ.get("ROPW", "1.0"))

# ═══ Charge and output ════════════════════════════════════════════════════════════════════════

# `QHV` — the hypervalent branch of the formal charge: above `b = 4` use `q = v − b` instead of the
#   octet form, so nitro `-N(=O)=O`, sulfone S and perchlorate Cl read 0. Sites with `b <= 4` are
#   unaffected. Default on.
QHV = os.environ.get("QHV", "1") == "1"
# `T8FORM` — M–L bond-order model: `thr` monotone Mayer-index thresholds (default) · `lik1`
#   likelihood with a pooled scale · `lik` likelihood from `data/b_ml_mayer.csv`.
T8FORM = os.environ.get("T8FORM", "thr")

# ═══ T4 — which M–L contacts are bonds ════════════════════════════════════════════════════════

# `SATVETO` — no M–X bond to a non-metal whose internal neighbours already fill its valence
#   (`deg_int(X) >= CAP(X)`), with H, B/Al, and any atom bonded to B/Al exempt. Default on.
#   Counts neighbours, not bond orders (see `rules.pipeline.drop_saturated`).
SATVETO = os.environ.get("SATVETO", "1") == "1"
# `HALW` — Mayer floor for an M–halogen bond **whose halogen already carries an internal covalent
#   bond** (`CF₃`, triflate, `BF₄⁻`, `PF₆⁻`); a terminal halide is untouched. Float, default 0.30;
#   0 = off; needs `wbo`. The fitted `d_bond.csv` has almost no Mayer veto for M–halogen pairs.
HALW = float(os.environ.get("HALW", "0.30"))
HALOGENS = {"F", "Cl", "Br", "I", "At"}  # elements `HALW` applies to

# `AGOC` — for an agostic `C–H···M` contact, also drop the M–C candidate when the carbon carries an
#   H with `d(M,H) < d(M,C)` and `d(M,H) < AGOC` (Å). Float, default 2.0; 0 = off.
#   ⚠️ A (metal, fragment) pair is left untouched if the drop would leave it no M–L bond.
AGOC = float(os.environ.get("AGOC", "2.0"))

# ═══ Joint solve (`rules.joint2`) ═════════════════════════════════════════════════════════════

# `JOINT_MAX` — above this many MILP variables the joint solve gives up (`predict` raises).
#   Integer, default 5000.
JOINT_MAX = int(os.environ.get("JOINT_MAX", "5000"))
# `JOINT_TIME` — seconds one MILP solve may take; a solve stopped there keeps the best solution it
#   found, and the search for more candidates stops once four times this has passed. Float, 20.
JOINT_TIME = float(os.environ.get("JOINT_TIME", "20"))
# `JOINTQ` — weight of the charged-atom penalty (|FC| of each non-coordinating atom) against the
#   distance score. Float, default 2.0.
JOINTQ = float(os.environ.get("JOINTQ", "2.0"))
# `JOINTCAT` — extra penalty (in units of `JOINTQ`) on a carbenium carbon (sextet, +1). Only
#   carbons of a planar all-carbon ring get that level; it is what lets a Hückel cation (tropylium,
#   cyclopropenium) be written at all. Float, default 1.0.
JOINTCAT = float(os.environ.get("JOINTCAT", "1.0"))
# `JOINTSYM` — penalty per unit of oxidation-state difference between two metals of the same
#   element, so a tie goes to the even split (Re2Cl8 -> III/III, Co2(CO)8 -> 0/0). Float,
#   default 0.5 — mixed valence is rare, and it must outweigh the `JOINTOSW` prior.
JOINTSYM = float(os.environ.get("JOINTSYM", "0.5"))
# `JOINTOSW` — weight of the oxidation-state prior in the joint solve: each candidate costs
#   `JOINTOSW · -ln(p / p_max)`, with `p` the element's state frequency in the train-split CSD names
#   (`data/os_prior.json`, add-one smoothed over the hard range). Float, default 0.1 — a
#   tie-breaker. 0 = off.
JOINTOSW = float(os.environ.get("JOINTOSW", "0.1"))
# `JOINTRAD` — cost (in units of `JOINTQ`) of putting an unpaired electron on a ligand atom rather
#   than on a metal, when `n_unpaired > 0`. Float, default 0.5.
JOINTRAD = float(os.environ.get("JOINTRAD", "0.5"))
# `JOINTADJ` — penalty (in units of `JOINTQ`) per pair of adjacent same-sign charges. Float, 1.0.
JOINTADJ = float(os.environ.get("JOINTADJ", "1.0"))
# `JOINTK` — how many candidates (distinct signatures) the K-best search draws. Integer, 5.
JOINTK = int(os.environ.get("JOINTK", "5"))
# `JOINTTIE` — candidates within this many `JOINTQ` of the best MILP score are ranked by Mayer
#   consistency. Float, 1.0.
JOINTTIE = float(os.environ.get("JOINTTIE", "1.0"))
# `JOINTCONJEPS` — a bond reads `Conj` when flipping its S/D alternation changes the score by at
#   most this much. Float, 2.0.
JOINTCONJEPS = float(os.environ.get("JOINTCONJEPS", "2.0"))
# `JOINTCHAINW` — a haptic chain of k atoms should carry a fixed number of charged atoms (odd k one,
#   even k none, a 4n ring none or two); each charged atom off that count costs JOINTCHAINW·JOINTQ.
JOINTCHAINW = float(os.environ.get("JOINTCHAINW", "1.0"))
# `JOINTLIGSYM` — per unit charge difference between two ligands with the same element graph
#   (the ligand counterpart of `JOINTSYM`). 0 turns it off.
JOINTLIGSYM = float(os.environ.get("JOINTLIGSYM", "0.5"))
# `JOINTFAR` — a contact M···X whose neighbour Y on the same metal is this many times nearer is a
#   "far contact" and is dropped.
JOINTFAR = float(os.environ.get("JOINTFAR", "1.3"))
# `JOINTLOWQ` — candidates within this much MILP score of the best are ones the geometry cannot
#   tell apart; among them the smaller total |ligand charge| wins (the less charged state is the
#   more stable one: bpy over bpy²⁻). 0 = off.
JOINTLOWQ = float(os.environ.get("JOINTLOWQ", "0.3"))
# `JOINTCUT` — cost (in units of `JOINTQ`) of reading a T4 sigma contact as no bond. Only a contact
#   whose donor is left with no pair to give when the solve runs without V1 has that choice; the
#   small cost keeps T4's call where both readings score the same. Float, default 0.05.
JOINTCUT = float(os.environ.get("JOINTCUT", "0.05"))

# `JOINTCUTW` · `JOINTCUTD` — a contact is offered the cut only while T4's own evidence for it is
#   weak: Mayer below JOINTCUTW and distance above JOINTCUTD × the T4 cutoff (`d_bond`, itself about
#   1.2 × r_M + r_X). Either firm keeps T4's call — the ligand having no pair to give does not
#   overturn it. Among the contacts the solve cut on train (label suspects out) 13 of 14 not-bonded had
#   Mayer <= 0.26 and distance >= 0.88 of the cutoff (the 14th, UNEBIA La–N 2.06 Å, is a bond the
#   label leaves out); the distance is set at 0.9 of the cutoff (owner decision).
JOINTCUTW = float(os.environ.get("JOINTCUTW", "0.30"))
JOINTCUTD = float(os.environ.get("JOINTCUTD", "0.90"))
# `JOINTFIRM` — whether that guard applies (the default for `predict(firm_contacts=...)`). Off, a
#   firm contact whose donor has no pair to give is offered the cut like a weak one, and its η²
#   partner needs the metal over the bond (below 90°). Default on.
JOINTFIRM = os.environ.get("JOINTFIRM", "1") == "1"

# Group numbers of the d-block centres, for the oxidation-state candidates of the joint solve.
_GROUP = {"Sc": 3, "Y": 3, "Ti": 4, "Zr": 4, "Hf": 4, "V": 5, "Nb": 5, "Ta": 5, "Cr": 6, "Mo": 6,
          "W": 6, "Mn": 7, "Re": 7, "Fe": 8, "Ru": 8, "Os": 8, "Co": 9, "Rh": 9, "Ir": 9,
          "Ni": 10, "Pd": 10, "Pt": 10, "Cu": 11, "Ag": 11, "Au": 11, "Zn": 12}  # fmt: skip
_MAXOS = {"Mg": 2, "Al": 3, "Ga": 3, "In": 3, "Sn": 4, "Pb": 4, "La": 3, "Ce": 4}  # fmt: skip

def os_range(el):
    """Hard oxidation-state range of a centre, `(lo, hi)`.

    d-block groups 3–10: `OS = group - d` with `d` in `[0, 10]`. Groups 11–12 take `[0, group]`
    instead — Au(0) clusters and Zn(I) dimers are real, and `group - 10` would exclude them.
    Other centres: `[0, highest common state]`.
    """
    if el in _GROUP:
        g = _GROUP[el]
        return (g - 10, g) if g <= 10 else (0, g)
    return (0, _MAXOS.get(el, 4))

# `WMIN` — T4: a global Mayer floor for M–L candidates, on top of the per-element-pair `w_veto`.
#   Float, default 0.0 (off). ⚠️ It also removes haptic M–C bonds, which are weak by construction.
WMIN = float(os.environ.get("WMIN", "0.0"))

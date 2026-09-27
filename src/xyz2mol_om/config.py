"""Element tables and rule switches — **these values are the pipeline**.

The rules that read these numbers are stated in `docs/PIPELINE.md`; this file holds the numbers
and the switches that turn the optional rules on and off. Every switch is an environment variable
read once, at import, so a run can be reproduced from the command line without editing the
package. A boolean switch is on when its variable is `1` and off for any other value.

Switches that default to off are optional variants outside the documented default pipeline.
"""

# ruff: noqa: E501
from __future__ import annotations

import os
import re
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data"  # fitted tables shipped with the package

# ═══ Element tables ═══════════════════════════════════════════════════════════════════════════

CLS = {"Single": 0, "Double": 1, "Triple": 2}  # bond-class name → class index
ORD = [1.0, 2.0, 3.0]  # bond order of classes 0–2
ORD4 = [1.0, 2.0, 3.0, 1.5]  # 0 Single · 1 Double · 2 Triple · 3 Conj

VAL = {  # valence electrons
    "H": 1, "B": 3, "C": 4, "N": 5, "O": 6, "F": 7, "Si": 4, "P": 5,
    "S": 6, "Cl": 7, "As": 5, "Se": 6, "Br": 7, "Te": 6, "I": 7,
}  # fmt: skip
FULL = {"H": 2}  # filled-shell quota. 8 by default, 2 only for H
VTGT = {  # neutral-atom bond-order target, used by the under-valence penalty
    "H": 1, "B": 3, "C": 4, "N": 3, "O": 2, "F": 1, "Si": 4, "P": 3,
    "S": 2, "Cl": 1, "Br": 1, "I": 1, "Se": 2, "As": 3, "Te": 2,
}  # fmt: skip
RCOV = {  # covalent radii (Å, Cordero) — fallback distance threshold for a pair not in the fit
    "H": 0.31, "B": 0.84, "C": 0.76, "N": 0.71, "O": 0.66, "F": 0.57, "Si": 1.11, "P": 1.07,
    "S": 1.05, "Cl": 1.02, "As": 1.19, "Se": 1.20, "Br": 1.20, "Te": 1.38, "I": 1.39,
    # Li, Na, Ge are neither centres nor in the fitted ligand tables; listed so they do not
    #   take the default radius of an unlisted element.
    "Li": 1.28, "Na": 1.66, "Ge": 1.20,
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
_LP_DEG = {"O": 2, "S": 2, "Se": 2, "N": 3, "P": 3}  # degree at which R2 makes the atom a donor

EHT_CUTOFF = -10.0  # extended-Hückel occupied-orbital cut, in eV
_EHT_VE = {  # valence electrons counted by the EHT fragment charge (unlisted elements: 4)
    "H": 1, "B": 3, "C": 4, "N": 5, "O": 6, "F": 7, "Si": 4, "P": 5,
    "S": 6, "Cl": 7, "As": 5, "Se": 6, "Br": 7, "Te": 6, "I": 7,
}  # fmt: skip

TAU_P = 0.05  # ring planarity tolerance (Å, out-of-plane rms) — Rule A and R4
THETA_HAPTIC = 81.02  # M–X–Y angle (deg) below which an M–L bond is side-on. Fitted.

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


# ═══ ①② the conjugation set ═══════════════════════════════════════════════════════════════════
# The rules are stated in `docs/PIPELINE.md` §T3 ①②. All are structural — no fitted number
# apart from the planarity tolerance `TAU_P`.

RULEA = os.environ.get("RULEA", "ge5")  # Rule A ring size: `ge5` (>= 5, default) · `eq6` · `off`
if RULEA not in ("ge5", "eq6", "off"):
    raise SystemExit(f"RULEA={RULEA!r} must be one of ge5|eq6|off")
# `R2CONJ` — R2: a lone-pair donor heteroatom keeps single bonds. Default on.
R2CONJ = os.environ.get("R2CONJ", "1") == "1"
# `R3RING` — R3: a 5-ring holding an R2 donor is Kekulé as a whole. Default on.
R3RING = os.environ.get("R3RING", "1") == "1"
# `R3MODE` — which donors trigger R3: `all` (default) every R2 donor · `N` nitrogen only ·
#   `mono` exactly one donor in an otherwise all-carbon ring · `nomix` all, except a ring whose
#   donors are of more than one element. Any other value acts as `all`.
R3MODE = os.environ.get("R3MODE", "all")
# `R4RING` — R4: a non-planar 4n all-carbon ring is not delocalized. Default on.
R4RING = os.environ.get("R4RING", "1") == "1"
# `R5SOLO` — R5: a lone `Conj` bond is not delocalized. Default on.
R5SOLO = os.environ.get("R5SOLO", "1") == "1"
# `R6SWAP` — R6: among one atom's bonds to neighbours of the same element (`Conj` excluded), swap
#   classes so the shorter bond never has the lower order, when the cap allows. Default off.
R6SWAP = os.environ.get("R6SWAP", "0") == "1"
# `R7RING` — R7: return an R2 donor inside a haptic ring to π. Default on.
R7RING = os.environ.get("R7RING", "1") == "1"
# `R7MIN` — R7 fires only when at least this many ring atoms are already haptic to that metal.
#   Integer, default 2.
R7MIN = int(os.environ.get("R7MIN", "2"))

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
# `USE_DINT` — not read by the package: T1 thresholds always come from `data/d_int.csv`.
USE_DINT = os.environ.get("USE_DINT", "0") == "1"

# ═══ ④ valence budget ═════════════════════════════════════════════════════════════════════════

# `CAPDUP_MAX` — the capacity reduction copies an atom with headroom `r` into `r` replicas, and a
#   matching can then take two replica edges of the **same** physical bond. The repair re-runs the
#   matching with such bonds pinned to one unit; this bounds the rounds. Integer, default 6.
#   An algorithmic safety limit, not a chemical rule.
CAPDUP_MAX = int(os.environ.get("CAPDUP_MAX", "6"))
# `TAUD` — add `TAUD` to ④'s internal `Double` weight, which moves its candidate gate from `g > 0`
#   to `g > -TAUD`. Float, default 0 (off).
TAUD = float(os.environ.get("TAUD", "0"))
# `QCOST` — weight of the formal-charge cost in ④'s matching objective. Raising a bond by one unit
#   adds `QCOST * (dq(a) + dq(b))`, `dq(x) = |q(b)| - |q(b+1)|`, with `q` from
#   `charge.formal.q_atom` (`solvers._qcost_step`); coordinating atoms are exempt.
#   Float, default 1.0; 0 = off; >= 50 makes it effectively lexicographic (charge first, then
#   distance). A cost, not a constraint: CO ligands and hypervalent `S=O`/`P=O` need a charge.
QCOST = float(os.environ.get("QCOST", "1"))
# `CAPQ` — quantise ④'s matching weights to a grid of this size, per element pair around the
#   pair's median (M–L edges form one group), with remaining ties broken by atom index — the same
#   construction `KEKQ` applies to ⑥. Float, default 0 (off).
#   ⚠️ Unlike `KEKQ` it can change the 4-class assignment, not only the integer orders.
CAPQ = float(os.environ.get("CAPQ", "0"))

# `CAPMILP` — solve ④ exactly as a MILP instead of through the matching reduction, which confirms
#   `Triple` greedily first. Default off. `scipy` is imported lazily, so with the flag off it is not
#   a dependency. Above `CAPMILP_MAX` variables (integer, default 1200) it falls back to matching.
CAPMILP = os.environ.get("CAPMILP", "0") == "1"
CAPMILP_MAX = int(os.environ.get("CAPMILP_MAX", "1200"))
# `BML3C_COST` — the ④·⑥ budget an atom in a 3c2e bridge spends **in total**, however many M–L
#   bonds it has: one electron pair over three centres is one bond of valence. Float, default 1.0.
BML3C_COST = float(os.environ.get("BML3C_COST", "1"))

# ═══ ⑤ EHT fragment charge ════════════════════════════════════════════════════════════════════

# `EHTSKIP` — fragments whose extended-Hückel charge target ⑤ does not chase, by composition: the
#   fragment's element symbols sorted and concatenated, comma-separated. Default `NO,SS,CCHH`
#   (NO, S₂, η²-acetylene); empty = none.
EHTSKIP = {v for v in os.environ.get("EHTSKIP", "NO,SS,CCHH").split(",") if v}
# `EHTMINFRAG` — ⑤ ignores the target of fragments smaller than this. Integer, default 0 (off).
EHTMINFRAG = int(os.environ.get("EHTMINFRAG", "0"))
# `EHTCOST` — total likelihood cost ⑤ may spend on one fragment; past it the fragment's orders are
#   reverted. Float, default -1; any negative value = no limit.
EHTCOST = float(os.environ.get("EHTCOST", "-1"))

# ═══ ⑥ Output converter ═══════════════════════════════════════════════════════════════════════

# `ETAEXO` — in ⑥, an atom of an η-coordinated ring pairs only inside that ring (weight −1e6 on a
#   `Conj` bond leaving the ring). Default off.
ETAEXO = os.environ.get("ETAEXO", "0") == "1"

# `KEKQ` — grid size for ⑥'s Kekulé matching weights (`score[Double] − score[Single]`). Weights in
#   one bin are ordered by atom index, so the two equivalent Kekulé structures of a ring do not swap
#   on a few thousandths of an Å. Float, default 8.0; 0 = off. Only the integer orders change;
#   `bonds_4class` is unaffected.
KEKQ = float(os.environ.get("KEKQ", "8"))
# `KEKQMODE` — grid origin for `KEKQ`: `pair` (default) per element pair, anchored on that pair's
#   median weight in the fragment · `abs` absolute grid from 0. Any other value acts as `pair`.
#   Per-pair keeps the real cross-pair preferences (`C=O` vs `C–C`) that one global grid erases.
KEKQMODE = os.environ.get("KEKQMODE", "pair")

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
# `ETA1SIG` — a ligand fragment that gives a metal exactly one haptic atom is η¹, i.e. a σ bond, so
#   the tag is dropped and the bond takes an M–L order like any other. Counted per ligand fragment,
#   not per connected run of haptic atoms. Default on.
ETA1SIG = os.environ.get("ETA1SIG", "1") == "1"
# `NOCTET` — nitrogen never carries five bonds in the output: one `N=O` of a nitro group to a
#   terminal O is demoted, giving `[N+](=O)[O-]`. Period-3 atoms (sulfone S, perchlorate Cl) keep
#   the hypervalent form. Default on. See `charge.formal.octet_fix_period2`.
NOCTET = os.environ.get("NOCTET", "1") == "1"
# `HALW` — Mayer floor for an M–halogen bond **whose halogen already carries an internal covalent
#   bond** (`CF₃`, triflate, `BF₄⁻`, `PF₆⁻`); a terminal halide is untouched. Float, default 0.30;
#   0 = off; needs `wbo`. The fitted `d_bond.csv` has almost no Mayer veto for M–halogen pairs.
HALW = float(os.environ.get("HALW", "0.30"))
HALOGENS = {"F", "Cl", "Br", "I", "At"}  # elements `HALW` applies to

# `AGOC` — for an agostic `C–H···M` contact, also drop the M–C candidate when the carbon carries an
#   H with `d(M,H) < d(M,C)` and `d(M,H) < AGOC` (Å). Float, default 2.0; 0 = off.
#   ⚠️ A (metal, fragment) pair is left untouched if the drop would leave it no M–L bond.
AGOC = float(os.environ.get("AGOC", "2.0"))

# ═══ Haptic bonds (T5, ⑥) ═════════════════════════════════════════════════════════════════════

# `ETA2NEAR` — when forming η² pairs, widen the T4 distance threshold by this much (Å) for the
#   partner atom only; at least one end must pass T4 unwidened. Float, default 0 (off).
ETA2NEAR = float(os.environ.get("ETA2NEAR", "0"))

# `ETAPI` — weight added in ⑥'s Kekulé matching to a bond whose two ends are a metal's only two
#   haptic atoms (η²), so its `Double` sits under the η² when the alternation allows. A weight, not
#   a constraint. Float, default 1e5; 0 = off. Keep it below `ETAEXO`'s 1e6 so `ETAEXO` wins.
ETAPI = float(os.environ.get("ETAPI", "100000"))

# ═══ Joint solve (`rules.joint`) ══════════════════════════════════════════════════════════════

# `JOINT` — solve every ligand-internal bond order in one MILP instead of ①② → ④ → ⑤ in sequence.
#   The haptic set, M–L orders and T7 tags still come from the sequential path. Default off.
#   Needs scipy (imported inside `rules.joint` only).
JOINT = os.environ.get("JOINT", "0") == "1"
# `JOINT_MAX` — above this many MILP variables the joint solve gives up and the sequential path
#   is used. Integer, default 5000.
JOINT_MAX = int(os.environ.get("JOINT_MAX", "5000"))
# `JOINTQ` — weight of the charged-atom penalty (sum of |FC| over non-coordinating C·N·O·F)
#   against the distance score. Float, default 2.0; chosen on the train sample (1 · 2 compared).
JOINTQ = float(os.environ.get("JOINTQ", "2.0"))
# `JOINTCAT` — extra penalty (in units of `JOINTQ`) on a carbenium carbon (sextet, +1) in the joint
#   solve. Only ring carbons of the sequential `Conj` set get that level; it is what lets a Hückel
#   cation (tropylium, cyclopropenium) be written at all. Float, default 1.0.
JOINTCAT = float(os.environ.get("JOINTCAT", "1.0"))
# `JOINTSYM` — penalty per unit of oxidation-state difference between two metals of the same
#   element, so a tie goes to the even split (Re2Cl8 -> III/III, Co2(CO)8 -> 0/0). Float,
#   default 0.5 — mixed valence is rare, and it must outweigh the `JOINTOSW` prior.
JOINTSYM = float(os.environ.get("JOINTSYM", "0.5"))
# `JOINTOSW` — weight of the oxidation-state prior in the joint solve: each candidate costs
#   `JOINTOSW · -ln(p / p_max)`, with `p` the element's state frequency in the train-split CSD names
#   (`data/os_prior.json`, add-one smoothed over the hard range). Float, default 0.1 — a
#   tie-breaker: at 1.0 it outweighed the distance evidence (a mu-CO bent to C=O to buy Fe(II)).
#   0 = off.
JOINTOSW = float(os.environ.get("JOINTOSW", "0.1"))
# `JOINTDON` — the share of `JOINTQ` a sigma donor pays for its first unit of negative charge
#   (an X-type donor: halide, alkoxide, thiolate, amide). 1.0 = the same as any atom. Float,
#   default 0.25; chosen on the train sample (1 · 0.5 · 0.25): at 1.0 dithiolates went neutral.
JOINTDON = float(os.environ.get("JOINTDON", "0.25"))
# `JOINTRAD` — cost (in units of `JOINTQ`) of putting an unpaired electron on a ligand atom rather
#   than on a metal, when `n_unpaired > 0`. Float, default 0.5.
JOINTRAD = float(os.environ.get("JOINTRAD", "0.5"))
# ── JOINT v2 (`rules.joint2`, dev/docs/plans/2026-09-27-joint-v2.md) ──
# `JOINTADJ` — penalty (in units of `JOINTQ`) per pair of adjacent same-sign charges. Float, 1.0.
JOINTADJ = float(os.environ.get("JOINTADJ", "1.0"))
# `JOINTK` — how many candidates (distinct signatures) the K-best search draws. Integer, 5.
JOINTK = int(os.environ.get("JOINTK", "5"))
# `JOINTTIE` — candidates within this many `JOINTQ` of the best MILP score are ranked by Mayer
#   consistency. Float, 1.0.
JOINTTIE = float(os.environ.get("JOINTTIE", "1.0"))
# `JOINTMODE` — `kbest` (candidates, validated, ranked) or `cut` (add a cut per failed check).
JOINTMODE = os.environ.get("JOINTMODE", "kbest")
# `JOINTCONJEPS` — a bond reads `Conj` when flipping its S/D alternation changes the score by at
#   most this much. Float, 0.5.
JOINTCONJEPS = float(os.environ.get("JOINTCONJEPS", "2.0"))
# `JOINTRAWSC` — experimental: the joint MILP reads each bond's Single · Double · Triple scores
#   as they are, instead of letting a `Conj` score support orders 1 and 2 alike (`order_scores`).
#   `Conj` is still read after the solve. Default off.
JOINTRAWSC = os.environ.get("JOINTRAWSC", "0") == "1"
# `JOINTSC` — which ③ the joint MILP reads: `orig` (as the default path) · `noprior` (no
#   class-frequency prior) · `pooled` / `pooledmin` (no prior, one width per element pair: the
#   widest / narrowest of Single · Double · Conj). Default `pooledmin` (train1200). The Conj reading
#   after the solve keeps the default path's ③.
JOINTSC = os.environ.get("JOINTSC", "pooledmin")
# `JOINTCHAINQ` — a haptic chain of k atoms carries a fixed number of charged atoms (odd k one,
#   even k none, a 4n ring none or two). Default on; 0 turns it off (analysis).
JOINTCHAINQ = os.environ.get("JOINTCHAINQ", "1") == "1"
# `JOINTCHAINSOFT` — the chain-charge count as a penalty (`JOINTCHAINW`·JOINTQ per charged atom off
#   the count) instead of a hard row. Default on (holdout: failed 26 -> 4, scores within 0.1 point).
JOINTCHAINSOFT = os.environ.get("JOINTCHAINSOFT", "1") == "1"
JOINTCHAINW = float(os.environ.get("JOINTCHAINW", "1.0"))
# `JOINTADJDON` — two atoms bound to the same metal are exempt from the adjacent same-sign penalty
#   (metallacyclopropane C(-)–C(-)). Default on; 0 penalises them like any pair.
JOINTADJDON = os.environ.get("JOINTADJDON", "1") == "1"
# `JOINTLIGSYM` — per unit charge difference between two ligands with the same element graph
#   (the ligand counterpart of `JOINTSYM`). 0 turns it off.
JOINTLIGSYM = float(os.environ.get("JOINTLIGSYM", "0.5"))
# `JOINTFAR` — a contact M···X whose neighbour Y on the same metal is this many times nearer is a
#   "far contact" and is dropped under JOINT (train1200: no CSD bond is that lopsided).
JOINTFAR = float(os.environ.get("JOINTFAR", "1.3"))

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


# ═══ Post-⑥ repairs (`docs/PIPELINE.md` §T3) ══════════════════════════════════════════════════
# Each repair is kept only if the summed |formal charge| falls.

# `QSHIFT` — after ⑥, when two charged atoms sit at the ends of an alternating path, shift the π
#   along it without exceeding `CAP` (`charge.formal.shift_pi_to_cancel`). On a single bond only a
#   same-sign pair is raised (never peroxide `[O⁻]–[O⁻]`); a pair whose two ends both coordinate a
#   metal is skipped (a dianionic chelate). Default on.
QSHIFT = os.environ.get("QSHIFT", "1") == "1"

# `SIGCUT` — when raising the bond between two adjacent anions is blocked only by a σ M–L's share
#   of the cap, drop that σ M–L and solve ③–⑥ again without it
#   (`charge.formal.sigma_ml_blocking_cancel`, `api.predict`). Skipped when the partner coordinates
#   the same metal, for peroxide, for an `Al` centre, and for a bond with no Mayer value; limited by
#   `SIGCUTW`. Default on. ⚠️ A cut may leave the fragment with no bond to the metal.
SIGCUT = os.environ.get("SIGCUT", "1") == "1"

# `SIGCUTW` — `SIGCUT` only cuts a σ M–L whose Mayer bond index is below this. Float, default 0.40.
#   `SIGCUTFIT` can override it for an adjacent pair.
SIGCUTW = float(os.environ.get("SIGCUTW", "0.40"))

# `SIGETA` — for a σ M–L that `SIGCUTW` protects from `SIGCUT`, make that bond and the partner atom
#   haptic (η²) instead of cutting, so the order can be raised. Default off.
SIGETA = os.environ.get("SIGETA", "0") == "1"

# `SIGCUTFIT` — exception to `SIGCUTW` for an adjacent pair: cut even above the Mayer limit when
#   the raised order is closer to the pair's class median length (`scores4` `med`) **and** the
#   current order lies outside its own distribution (`|d − med[cur]| > scl[cur]`). Default on.
SIGCUTFIT = os.environ.get("SIGCUTFIT", "1") == "1"

# `QGEM` — geminal branch of `QSHIFT` (and `SIGCUT`): two like-signed charges on one common
#   neighbour shift both bonds together (`[O⁻]–C–[O⁻]` → `O=C=O`) — raised for anions, lowered for
#   cations — only when both bond lengths fit the new orders. Default on.
QGEM = os.environ.get("QGEM", "1") == "1"

# `QCHFIT` — exception to `QSHIFT`'s chelate gate: on a single bond whose two ends both coordinate
#   a metal, shift anyway when the donor–donor bond length fits the raised order better.
#   Default off.
QCHFIT = os.environ.get("QCHFIT", "0") == "1"

# ═══ σ M–L bonds against the valence cap ══════════════════════════════════════════════════════

# `SATML` — Rule A's π-headroom test counts the **σ M–L bonds** as well as the internal degree:
#   `deg(X) + b_ML(X) >= CAP(X)` keeps X out of the ring that Rule A pins `Conj`. Default on.
SATML = os.environ.get("SATML", "1") == "1"

# `SIGCAP` — a **σ** M–L bond that would push its coordinating atom past its valence cap
#   (`b_int + n_σML > CAP`) is re-read as **haptic** when that atom already has a
#   `Double`/`Triple`/`Conj` bond; 3c2e atoms and B are skipped. Default on.
#   ⚠️ It can leave an η¹ tag: `drop_eta1` runs before it.
SIGCAP = os.environ.get("SIGCAP", "1") == "1"

# `WMIN` — T4: a global Mayer floor for M–L candidates, on top of the per-element-pair `w_veto`.
#   Float, default 0.0 (off). ⚠️ It also removes haptic M–C bonds, which are weak by construction.
WMIN = float(os.environ.get("WMIN", "0.0"))

# ═══ Oxidation-state parsing from the CSD chemical name (evaluation only) ═════════════════════

NAMEEL = {  # element → its English name in a CSD chemical name
    "Ti": "titanium", "Zr": "zirconium", "Hf": "hafnium", "V": "vanadium", "Nb": "niobium",
    "Ta": "tantalum", "Cr": "chromium", "Mo": "molybdenum", "W": "tungsten", "Mn": "manganese",
    "Re": "rhenium", "Fe": "iron", "Ru": "ruthenium", "Os": "osmium", "Co": "cobalt",
    "Rh": "rhodium", "Ir": "iridium", "Ni": "nickel", "Pd": "palladium", "Pt": "platinum",
    "Cu": "copper", "Ag": "silver", "Au": "gold", "Zn": "zinc", "Sc": "scandium",
    "Y": "yttrium", "La": "lanthanum", "Ce": "cerium", "B": "boron", "Al": "aluminium",
    "Ga": "gallium", "In": "indium", "Sn": "tin", "Pb": "lead", "Mg": "magnesium",
}  # fmt: skip
ALT = {  # alternative name stems (`ferr`ate, `cupr`ate, …)
    "Fe": ["ferr"], "Cu": ["cupr"], "Au": ["aur"], "Ag": ["argent"], "Sn": ["stann"],
    "Pb": ["plumb"], "Ni": ["nickel"], "Pt": ["platin"], "Mn": ["mangan"], "Al": ["alumin"],
}  # fmt: skip
# Roman numeral → oxidation state; `R` matches one numeral (0 to viii); `PAT` matches `name(iii)`
#   and `PATM` a mixed-valence `name(ii,iii)`.
ROMAN = {"0": 0, "i": 1, "ii": 2, "iii": 3, "iv": 4, "v": 5, "vi": 6, "vii": 7, "viii": 8}
R = r"(?:0|i{1,3}|iv|vi{0,3})"
PAT, PATM = re.compile(rf"([a-z]+)\(({R})\)"), re.compile(rf"([a-z]+)\(({R}(?:,{R})+)\)")

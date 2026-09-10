"""Element tables and rule switches — **these values are the pipeline**.

The rules themselves, with their thresholds and the reasoning behind them, are in
`docs/PIPELINE.md`; this file holds the numbers they read and the switches that turn the
optional ones on and off. Every switch is an environment variable, so a run can be reproduced
from the command line without editing the package.

Switches whose default is **off** are experiments that were measured and not adopted; the note
above each one says what it costs. Leave them alone unless you are reproducing that measurement.
"""

# ruff: noqa: E501
from __future__ import annotations

import os
import re
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data"

# ═══ Element tables ═══════════════════════════════════════════════════════════════════════════

CLS = {"Single": 0, "Double": 1, "Triple": 2}
ORD = [1.0, 2.0, 3.0]
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
RCOV = {  # covalent radii (Å) — the fallback distance threshold when a pair has no fitted entry
    "H": 0.31, "B": 0.84, "C": 0.76, "N": 0.71, "O": 0.66, "F": 0.57, "Si": 1.11, "P": 1.07,
    "S": 1.05, "Cl": 1.02, "As": 1.19, "Se": 1.20, "Br": 1.20, "Te": 1.38, "I": 1.39,
}  # fmt: skip

CAP = {  # valence ceiling used by ④ — bonds plus lone pairs, `b_int + b_ML <= CAP`
    "H": 1, "B": 4, "C": 4, "N": 5, "O": 4, "F": 4, "Si": 6, "P": 6,
    "S": 6, "Cl": 7, "As": 6, "Se": 6, "Br": 6, "Te": 6, "I": 6,
}  # fmt: skip
# `CAPSET` — the ceiling above is an **octet** ceiling (`2b + 2lp = 8` with `lp` free down to 0),
#   so every period-2 element lands at 4 and `CAP(N) = 5` allows pentavalent N. Solving octet and
#   formal charge together gives the tighter `b_max = 8 - v + q`: C 4 · N 3 (+1 → 4) · O 2 (+1 → 3)
#   · F 1 (+1 → 2). `tight` is that ceiling with cations allowed, `mid` tightens only O and N.
#   Both were measured and neither pays; `octet` is the default.
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
_EHT_VE = {
    "H": 1, "B": 3, "C": 4, "N": 5, "O": 6, "F": 7, "Si": 4, "P": 5,
    "S": 6, "Cl": 7, "As": 5, "Se": 6, "Br": 7, "Te": 6, "I": 7,
}  # fmt: skip

TAU_P = 0.05  # ring planarity tolerance (Å, out-of-plane rms) — Rule A and R4
THETA_HAPTIC = 81.02  # M–X–Y angle (deg) below which an M–L bond is side-on. Fitted.

# ═══ Centres ══════════════════════════════════════════════════════════════════════════════════

METALS = set(
    "Ti Zr Hf Nb Ta V La Sc Y Ce Cr Mo W Mn Re Fe Ru Os Co Rh Ir Ni Pd Pt "
    "Cu Ag Au Zn Al Ga In Sn Pb Mg B".split()
)
METALS_HARD = METALS - {"B"}


def centers(el):
    """The centre atoms — **`B` is conditional**.

        i is a centre ⟺ el[i] ∈ METALS \\ {B}
                      OR el[i] = B AND the structure holds no METALS \\ {B} atom

    In `B₂H₆` and the boranes B is the centre; inside a transition-metal complex (carborane,
    boryl, `BH₄⁻`) it is a ligand atom. B is the only metal-class element that forms internal
    bonds in the reference at all.
    """
    hard = {i for i, e in enumerate(el) if e in METALS_HARD}
    return hard or {i for i, e in enumerate(el) if e in METALS}


# ═══ ①② the conjugation set ═══════════════════════════════════════════════════════════════════
# The five rules are stated in `docs/PIPELINE.md` §T3 ①②. All are structural — no fitted number
# apart from the planarity tolerance `TAU_P`.

RULEA = os.environ.get("RULEA", "ge5")  # Rule A ring size: `ge5` (>= 5) · `eq6` · `off`
if RULEA not in ("ge5", "eq6", "off"):
    raise SystemExit(f"RULEA={RULEA!r} must be one of ge5|eq6|off")
R2CONJ = os.environ.get("R2CONJ", "1") == "1"  # R2 — lone-pair donor heteroatom keeps single bonds
R3RING = os.environ.get("R3RING", "1") == "1"  # R3 — a 5-ring holding an R2 donor is Kekulé whole
# R3 scope. `all` = every R2 donor · `N` = nitrogen only · `mono` = exactly one donor ·
#   `nomix` = all, except a ring mixing N with O/S.
#   `all` is the default because R3's argument — R2 leaves the ring's C–C bonds behind — holds for
#   furan O and thiophene S exactly as for pyrrole N, and the O-only donor five-rings carry an
#   average of 0.10 aromatic bonds out of 5 in the reference. `nomix` was measured and is not
#   better.
R3MODE = os.environ.get("R3MODE", "all")
R4RING = os.environ.get("R4RING", "1") == "1"  # R4 — a non-planar 4n all-carbon ring is not delocalized
R5SOLO = os.environ.get("R5SOLO", "1") == "1"  # R5 — a lone `Conj` bond is not delocalized
# R6 — for same-element bonds on one centre, distance order and bond-order order must agree.
#   Measured and not adopted: it costs more than it fixes.
R6SWAP = os.environ.get("R6SWAP", "0") == "1"
R7RING = os.environ.get("R7RING", "1") == "1"  # R7 — return an R2 donor inside a haptic ring to π
R7MIN = int(os.environ.get("R7MIN", "2"))  # how many other ring atoms must already be haptic

# ═══ ③ distance likelihood ════════════════════════════════════════════════════════════════════

# `LPCOND` — condition the class prior on the endpoints' internal-degree cell instead of on the
#   element pair alone. The same pair is a different bond at different degrees: a `C–O` at
#   `deg(C)=3, deg(O)=1` is a carbonyl (P(Double) .322) while the global prior says .111, which
#   is enough to flip a real `C=O` to `Single`.
LPCOND = os.environ.get("LPCOND", "1") == "1"
# `LPCOND_NOCONJ` — keep `Conj` on the **global** prior. The `C–C` deg-3/3 cell has P(Conj) = .908,
#   which drags `Double` into `Conj`.
LPCOND_NOCONJ = os.environ.get("LPCOND_NOCONJ", "1") == "1"
LPCOND_NMIN = int(os.environ.get("LPCOND_NMIN", "300"))  # samples a cell needs for its own prior
# `LPA` — prior temperature: `score = distance term + LPA·ln P(c)`. **The one fitted parameter in
#   the rules.** The prior is right about the base rate (`Double` is 2.2% of internal bonds) but
#   in a cell such as `C(3)–N(3)` the amines and amides dominate so heavily that it overrides the
#   distance for a genuine imine. Chosen on a one-decimal grid; dropping the prior entirely
#   (`LPA=0`) is by far the worst setting, so the prior itself is necessary.
LPA = float(os.environ.get("LPA", "0.8"))
# `LNORM` — add the Laplace normalisation term `-ln(2·scl)` to the likelihood. 0 off · 1 on ·
#   2 on except for `Conj`. Measured and not adopted: it favours narrow classes and makes the
#   `C–C` `Conj` over-prediction worse.
LNORM = os.environ.get("LNORM", "0")
LNORM_ON = LNORM in ("1", "2")
LNORM_SKIP_CONJ = LNORM == "2"
# `USE_ROP` — a second likelihood dimension from the extended-Hückel overlap population. On its
#   own it separates `C–C` `Double` from `Conj` far better than distance, but on top of R3+R4 the
#   gain is inside fold noise. Needs a precomputed cache.
USE_ROP = os.environ.get("USE_ROP", "0") == "1"
ROPW = float(os.environ.get("ROPW", "1.0"))
USE_DINT = os.environ.get("USE_DINT", "0") == "1"  # T1 threshold source: fitted table vs radii

# ═══ ④ valence budget ═════════════════════════════════════════════════════════════════════════

# `CAPDUP_MAX` — the capacity reduction copies an atom with headroom `r` into `r` replicas, and a
#   matching can then take two replica edges of the **same** physical bond, spending two units for
#   one bond. The repair re-runs the matching with such bonds pinned to one unit; this bounds the
#   loop. An algorithmic safety limit, not a chemical rule.
CAPDUP_MAX = int(os.environ.get("CAPDUP_MAX", "6"))
# `TAUD` — move the ④ candidate gate for an internal `Double` from `g > 0` to `g > -TAUD`.
#   Measured and rejected: `Double` peaks +0.0006 at τ = 0.5 (inside fold noise) and falls after,
#   because τ lifts every `Double` candidate at once, so the true ones gain no ground on the false.
TAUD = float(os.environ.get("TAUD", "0"))
# `QCOST` — the **charge-cost term in ④'s objective**, and with it the two-stage form of ④:
#     stage 1  minimise the formal charge the assignment has to pay for
#     stage 2  among the assignments that pay the same, take the one the distances like
#   Set it large (>= 50) and the objective is lexicographic — a true two-stage solve. Set it
#   small and the two blend. 0 = off (the old one-sided, distance-primary objective).
#
#   🔴 Why this is the right outer objective, and why "satisfy the valence" is **not**.
#   ④'s constraint is one-sided (`b_int + b_ML <= CAP`), so it can only say *"no room to raise
#   this bond"*, never *"this atom is paying for a charge it should not have to"*. Charge is then
#   free: `q = v + b - 8` is applied unconditionally, so **one bond-order error becomes a
#   2-electron ligand-charge error and a 2-unit oxidation-state error**.
#   But the fix cannot be a hard valence constraint. Measured on CSD: of 968 reference `Triple`
#   bonds, **876 (90.5%) are impossible under neutral valence** — 863 of them `C-O`, i.e. **CO
#   ligands**, where free CO gives C `deg 1` (room 3) and O `deg 1` (room 1) so neutral counting
#   stops at `C=O`; the bond is triple only because the species is **charge-separated**
#   `[O+]#[C-]`. Hypervalent `S=O`/`P=O` and nitro `N+` are the same story.
#   So the outer objective has to make a formal charge **expensive, not impossible**.
#
#   Form: raising a bond by one unit moves `b_int` at both ends by +1, so the matching weight gets
#     `g += QCOST * (dq(a) + dq(b))`,  dq(x) = |q(b)| - |q(b+1)|   — see `solvers._qcost_step`
#   `q` is the real `charge.formal.q_atom`, not a `VTGT` proxy: that is what makes the sulfone
#   (`S` b4->5->6) and the phosphine oxide (`P` b4->5) come out as **+1 each** (raise) where a
#   neutral-valence proxy says 0 and never raises them. On `CO` the two ends give `+1` and `-1`,
#   so the charge term is exactly neutral and the **distance** decides — which is the two-stage
#   behaviour working as intended.
#   A coordinating atom is **waived**: under the ionic cut an anionic donor (`Cl-`, `RO-`, `Cp-`)
#   is normal, and costing it would push ④ to raise bonds just to neutralise it.
#   **On by default at 1.0.** Holdout: T3 `Double` **+0.0037** (.7701 -> .7738) · `OS` **+0.0011**
#   · T8 `Double` +0.0011 · `Sq_L` +-0 · violations +-0; nothing regresses. This is the first form
#   of the fix that **improves** CSD rather than trading against it — the earlier neutral-valence
#   proxy cost T3 `Double` -0.0067 and `OS` -0.0040, because it scored 0 on exactly the
#   hypervalent sites where the real `|q|` says "raise".
#   ⚠️ A **blend** beats the strictly lexicographic form: `QCOST=50` puts T3 `Double` back to
#   baseline. The charge cost is a good prior, not an absolute one.
#   ⚠️ It does **not** move the Gold-DIGR frame-to-frame charge flips (78 -> 79). Those residuals
#   are ④ headroom competition, not charge: see `CAPQ`.
QCOST = float(os.environ.get("QCOST", "1"))
# `CAPQ` — the same canonicalisation `KEKQ` applies to ⑥, applied to ④'s matching weight.
#   ④ picks which bond gets an atom's remaining headroom by `score[Double] − score[Single]`, a
#   continuous function of the bond lengths, so two bonds competing for the same headroom can
#   swap places on a few thousandths of an Angstrom. Between two frames of one reaction path that
#   is a bond-order change that did not happen. Measured on Gold-DIGR: of the residual S/D/T
#   charge flips on geometrically unchanged atoms, **98.2% have an unchanged `Conj` membership**
#   (so not Rule A) and stage ⑤ is not responsible either (disabling it makes them worse) —
#   what is left is this competition.
#   Quantised **per element pair** for the same reason as `KEKQ`: within a pair the small
#   differences are noise, across pairs they are the whole signal.
#   🔴 **Measured and NOT adopted (default 0).** `CAPQ=8` takes the Gold-DIGR charge flips
#   79 -> 68, but unlike `KEKQ` it changes the **class assignment** itself, not just how a `Conj`
#   bond is turned into an integer, and the holdout pays: T3 `Double` **-0.0046** ·
#   T8 `Double` **-0.0064** · T8 `Triple` **-0.0075**. `CAPQ=32` is far worse (T3 `Double`
#   -0.0295). Turning it on is a domain choice for reaction-path geometry, like the old `VLAM`.
CAPQ = float(os.environ.get("CAPQ", "0"))

# `CAPMILP` — solve ④ exactly (a MILP) instead of through the matching reduction. The matching
#   confirms `Triple` greedily first, so 0.92% of ④ calls are not optimal (median objective loss
#   4.58, about one bond decision). Measured and left off: the ④ metrics rise but the deployment
#   output gets worse, because the stages do not share an objective. `scipy` is imported lazily,
#   so with the flag off it is not a dependency. Above `CAPMILP_MAX` variables it falls back.
CAPMILP = os.environ.get("CAPMILP", "0") == "1"
CAPMILP_MAX = int(os.environ.get("CAPMILP_MAX", "1200"))
# `BML3C_COST` — the ④·⑥ budget an atom in a 3c2e bridge spends **in total**, however many M–L
#   bonds it has: one electron pair over three centres is one bond of valence.
BML3C_COST = float(os.environ.get("BML3C_COST", "1"))

# ═══ ⑤ EHT fragment charge ════════════════════════════════════════════════════════════════════

# The trust gate — ⑤ only chases a target it has reason to trust. Overall the extended-Hückel
# target is wrong 3.8% of the time and the error concentrates in a few fragment classes.
EHTSKIP = {v for v in os.environ.get("EHTSKIP", "NO,SS,CCHH").split(",") if v}
#   `NO` 99.8% wrong · `SS` 94.9% · `CCHH` (η²-acetylene) 66.9%. `CCHH` is the weakest member and
#   the one most likely to be fitting the sample rather than chemistry.
#   ⚠️ The list is by **composition, not size**: a size cut would also disable the 22,298 `CO`
#      fragments, whose target is wrong only 1.7% of the time.
EHTMINFRAG = int(os.environ.get("EHTMINFRAG", "0"))  # skip fragments below this atom count (0 = off)
EHTCOST = float(os.environ.get("EHTCOST", "-1"))  # likelihood cost ⑤ may spend per ±1 move
# `ETAEXO` — in ⑥, let an atom of an η-coordinated ring pair only inside that ring. Measured and
#   not adopted.
ETAEXO = os.environ.get("ETAEXO", "0") == "1"

# `KEKQ` — ⑥'s Kekule matching is weighted by `score[Double] − score[Single]`, which is a
#   **continuous function of the bond lengths**. Inside a symmetric aromatic ring the two
#   alternating Kekule structures are chemically identical and their weights differ only by ring
#   distortion, so a few thousandths of an Angstrom flips which one is emitted. Between two frames
#   of one reaction path that reads as a bond-order change that did not happen: measured on
#   Gold-DIGR IRC endpoints, **94.6% of all order flips on geometrically unchanged bonds
#   (1,540 / 1,628) are `Conj`↔`Conj`** — the same 4-class answer, a different arbitrary integer,
#   1,443 of them aromatic `C–C`.
#   `KEKQ > 0` rounds the weight to that grid and breaks what is left canonically by atom index,
#   so a sub-threshold length difference can no longer decide. Genuine preferences (`EMAXAR`:
#   a 1.234 Å `C=O` against a 1.440 Å `C–C`) are orders of magnitude above any sane grid and are
#   unaffected. 0 = off.
#   ⚠️ Consumers that do not need integers should read `bonds_4class` instead — it already says
#      `Conj` in both frames, so this degeneracy is invisible there.
#   🔴 The grid is applied **per element pair, around that pair's own median in the fragment**,
#   not globally. Measured on CSD: resolving the aromatic labels by distance gives
#   `Conj→Single` median **1.394 Å** against `Conj→Double` **1.386 Å` — **0.008 Å apart**, so
#   distance carries essentially no information about which `C–C` of a ring is the double one.
#   Across element pairs it carries a great deal (`EMAXAR`: `C=O` 1.234 Å vs `C–C` 1.440 Å).
#   A single global grid big enough to flatten a ring also erases that (`KEKQ=16` global:
#   `Σq_L` −0.0034); scoping to the element pair flattens the first and leaves the second alone.
#   **On by default at 8.0.** Holdout: T3 `Double` +0.0002 · `OS` +0.0004 · `Σq_L` ±0.0000 ·
#   violations +0.0001 (2 structures); every other task byte-identical, since it only touches how
#   a `Conj` bond is turned into an integer. Gold-DIGR: charge flips on geometrically unchanged
#   atoms from this cause **55 → 8** per 400 reactions. `KEKQ=0` restores the old behaviour.
KEKQ = float(os.environ.get("KEKQ", "8"))
# `KEKQMODE` — where the quantisation grid is anchored.
#   `abs`  absolute grid (origin 0). Geometry-independent, so the bin a bond falls in cannot
#          move when the geometry does.
#   `pair` per element pair, anchored on **that pair's median in this fragment** (the default).
#   Measured on the **full 10,598-reaction** Gold-DIGR set (charge flips on unchanged geometry):
#          原본 3,382 | `pair` **2,440** | `abs` 2,925
#          resonance bucket   1,469 | **487** | 972       <- `pair` is much the better anchor
#          `Conj`-boundary      345 |   514   | 513
#   ⚠️ Both modes raise the `Conj`-boundary bucket by the same ~170, so that regression is **not**
#      anchor drift (the hypothesis this switch was added to test — it was wrong). It comes from
#      `KEKQ` itself. The likeliest reading is that those `Conj`-boundary errors were always
#      there and the old Kekule placement happened to cancel them in the charge; canonicalising
#      stops the cancellation and they become visible. **Not confirmed** — it is the open item.
KEKQMODE = os.environ.get("KEKQMODE", "pair")

# ═══ Charge and output ════════════════════════════════════════════════════════════════════════

# `QHV` — the hypervalent branch of the formal charge: above `b = 4` the octet form would need a
#   negative lone-pair count and reads nitro `-N(=O)=O` as +2, sulfone S as +4 and perchlorate Cl
#   as +6. The hypervalent branch gives 0 for all three and no `b <= 4` site changes.
QHV = os.environ.get("QHV", "1") == "1"
T8FORM = os.environ.get("T8FORM", "thr")  # M–L order form: `thr` monotone thresholds · `lik`

# ═══ T4 — which M–L contacts are bonds ════════════════════════════════════════════════════════

# `SATVETO` — no M–X bond to a non-metal whose internal neighbours already fill its valence
#   (`deg_int(X) >= CAP(X)`), with H, B/Al, and any atom bonded to B/Al exempt. On geometries far
#   from the fitted domain — DFT reaction-path endpoints, say — a metal often sits 2.4 Å from an
#   already-saturated carbon, and those contacts used to be reported as bonds. See
#   `rules.pipeline.drop_saturated` for why the test counts neighbours rather than bond orders.
SATVETO = os.environ.get("SATVETO", "1") == "1"
# `ETA1SIG` — a ligand that gives a metal exactly one haptic atom is eta-1, which is another name
#   for a sigma bond, so the tag is dropped and the bond takes an M-L order like any other.
#   The CSD reference calls **none** of those 17 holdout bonds `Pi`, while every higher k is
#   95-100% right. On the holdout: T5 .9778 -> .9783 and valence violations .0205 -> .0185,
#   with `Sq_L`, T1, T4 and T6 unchanged and `OS` down by one structure.
#   ⚠️ Counted **per ligand fragment**, not per connected run of haptic atoms. Both were measured:
#      a connected-component count drops 27 bonds of which 5 really are `Pi`, i.e. it buys 4 more
#      false positives at the price of 5 true ones.
ETA1SIG = os.environ.get("ETA1SIG", "1") == "1"
# `NOCTET` — nitrogen never carries five bonds in the output. The reference writes a nitro group
#   as `-N(=O)=O`, but N is period 2 and cannot exceed an octet, so one `N=O` to a terminal O is
#   demoted and the charge separates into `N+` / `O-` by the ordinary octet rule. Period-3 atoms
#   (sulfone S, perchlorate Cl) keep the hypervalent form, which is legitimate for them.
#   See `charge.formal.octet_fix_period2`.
NOCTET = os.environ.get("NOCTET", "1") == "1"
# `HALW` — Mayer floor for an M–halogen bond **whose halogen already carries an internal
#   covalent bond**. A terminal halide (`M–Cl`, `M–F`) has no internal bond and is untouched.
#   Why it is needed: `d_bond.csv` has **no `w_veto` worth the name for halogens** — 92 of the 96
#   fitted M–halogen rows carry `w_veto = 0.000`, and pairs such as `Pd–F` have **no row at all**,
#   so they fall back to `c1g·(RCOV[X] + RCOV[M])` with `RCOV` holding no metal radius (default
#   1.6 Å) — a 2.82 Å window with no Mayer floor whatever. That is not sloppy fitting: in the CSD
#   reference **the negative examples do not exist** (no CSD structure puts a metal 2.6 Å from a
#   CF₃ fluorine), so the fit had nothing to learn a floor from. Reaction-path endpoints are full
#   of them — triflate, `CF₃`, `BF₄⁻`, `PF₆⁻`.
#   Measured on Gold-DIGR (accepted M–halogen bonds, split by whether the halogen is already
#   bonded to a non-metal):
#         F  bound (CF₃·OTf)   n=152  median w 0.157   96.7% below 0.30   ← the spurious ones
#         F  terminal          n= 26  median w 0.859    3.8% below 0.30
#         Cl bound (R–Cl, OA)  n= 97  median w 0.354    0.0% below 0.30   ← must survive
#         I  bound (R–I, OA)   n= 26  median w 0.556    0.0% below 0.30   ← must survive
#   So 0.30 removes ~97% of the false M–F and keeps **every** genuine oxidative-addition halide.
#   ⚠️ This is not the rejected global `WMIN`. `WMIN` failed because a **haptic** M–C is weak by
#      construction (the π electrons are shared over five or six carbons) and a global floor
#      cannot tell that from a weak contact — T5 .9778 → .7241. A halogen is never part of a π
#      system, so it can never be haptic, and this floor cannot reach that failure.
#   0 = off.
HALW = float(os.environ.get("HALW", "0.30"))
HALOGENS = {"F", "Cl", "Br", "I", "At"}

# `SIGCAP` — a **sigma** M-L bond that would push its coordinating atom past its valence cap is
#   re-read as **haptic**, when that atom belongs to a pi fragment.
#   The violation is itself the evidence: a carbon cannot hold five bonds, so if `b_int + n_ML`
#   exceeds `CAP` the sigma reading is wrong, and an atom that already carries a `Double`/
#   `Triple`/`Conj` bond has the one alternative that costs no budget — eta-2. This is the same
#   argument rule 5* makes ("eta-2 is a property of the bond, not of each atom"), applied where
#   the angle test alone got it wrong.
#   Measured on Gold-DIGR: **every** valence violation is this one shape — 101 of 101 on carbon,
#   every one over by exactly 1, every one caused by a sigma M-L landing on an atom whose
#   internal bonds already fill it, and 69 of 73 on a `deg 3 / b_int 4` sp2 carbon (an alkene,
#   aryl or carbonyl carbon). Those contacts are also weak for an M-C: Mayer median **0.254**
#   against 0.460 for M-C at large, so they are exactly the marginal side-on approaches the
#   81.02 deg angle test is least reliable on.
#   ⚠️ T4's `SATVETO` cannot catch these: it tests `deg_int >= CAP`, and an sp2 carbon has
#      `deg 3 < CAP 4`. It counts neighbours rather than bond orders on purpose - a `b_int` form
#      would veto every alkene, arene and Cp coordination (`rules.pipeline.drop_saturated`).
#   **On by default.** CSD holdout: violations **.0187 -> .0125** (a third of them; the excess
#   over the reference baseline of 0.4% nearly halves) at a cost of haptic F1 .9787 -> .9738.
#   **Every other task is unchanged to four decimals** - T1, all four T3 classes, T4, T6, all
#   three T8 classes, `Sq_L` and `OS`. Gold-DIGR: valence violations **10.7% -> 0.0%** of
#   reactions, which takes the fraction passing every self-check from 75.9% to **84.5%**.
#   ⚠️ The haptic cost is real: some of these reclassifications disagree with the CSD `Pi` label,
#      so the rule does over-apply. It is kept in this unconditioned form because the argument
#      it encodes has no free parameter - the violation *is* the evidence - and because a
#      chemically impossible output is a different kind of error from a mislabelled one.
SIGCAP = os.environ.get("SIGCAP", "1") == "1"

# `WMIN` — a global Mayer floor for M–L candidates, on top of the per-element-pair `w_veto`.
#   Measured and rejected. A **haptic** M–C is weak by construction — the π electrons are shared
#   over five or six carbons, so each individual M–C is small — and a floor cannot tell that from
#   a weak contact. `WMIN=0.3` costs T5 haptic .9778 → .7241 to buy 1.5%p of valence violations.
WMIN = float(os.environ.get("WMIN", "0.0"))

# ═══ Oxidation-state parsing from the CSD chemical name (evaluation only) ═════════════════════

NAMEEL = {
    "Ti": "titanium", "Zr": "zirconium", "Hf": "hafnium", "V": "vanadium", "Nb": "niobium",
    "Ta": "tantalum", "Cr": "chromium", "Mo": "molybdenum", "W": "tungsten", "Mn": "manganese",
    "Re": "rhenium", "Fe": "iron", "Ru": "ruthenium", "Os": "osmium", "Co": "cobalt",
    "Rh": "rhodium", "Ir": "iridium", "Ni": "nickel", "Pd": "palladium", "Pt": "platinum",
    "Cu": "copper", "Ag": "silver", "Au": "gold", "Zn": "zinc", "Sc": "scandium",
    "Y": "yttrium", "La": "lanthanum", "Ce": "cerium", "B": "boron", "Al": "aluminium",
    "Ga": "gallium", "In": "indium", "Sn": "tin", "Pb": "lead", "Mg": "magnesium",
}  # fmt: skip
ALT = {
    "Fe": ["ferr"], "Cu": ["cupr"], "Au": ["aur"], "Ag": ["argent"], "Sn": ["stann"],
    "Pb": ["plumb"], "Ni": ["nickel"], "Pt": ["platin"], "Mn": ["mangan"], "Al": ["alumin"],
}  # fmt: skip
ROMAN = {"0": 0, "i": 1, "ii": 2, "iii": 3, "iv": 4, "v": 5, "vi": 6, "vii": 7, "viii": 8}
R = r"(?:0|i{1,3}|iv|vi{0,3})"
PAT, PATM = re.compile(rf"([a-z]+)\(({R})\)"), re.compile(rf"([a-z]+)\(({R}(?:,{R})+)\)")

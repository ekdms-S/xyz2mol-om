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
# `LPCOND_NMIN` — samples a cell needs for its own prior; below it the bond falls back to the
#   **element-pair global** prior. ⚠️ The shipped `scores4.json` was fitted at 300
#   (`_meta.n_min_cell`), so lowering this at runtime does nothing — the cells are not in the
#   file. Changing it means refitting (`dev/analysis/scratch/260903_export_scores4.py`).
#   🔴 **The fallback is the weakness, not the threshold.** Falling back to the global prior
#   hands the bond the *most common* cell's answer, which is exactly the one a rare cell needs
#   to be told apart from. `C–O` is the worked case (`260910_lpcond_cell_gap.py`, train):
#       cell (1,1)  n 22,294   98% Triple     a terminal / metal carbonyl
#       cell (2,1)  n    238   93% Double     a carbon with two neighbours — CO₂, an acyl
#   `(2,1)` is **62 samples short of 300**, so it is dropped and CO₂ inherits the carbonyl prior:
#   at 1.145 Å that scores Triple −1.01 against Double −6.81, the cap forces the other C–O to
#   `Single`, and free CO₂ comes out **`[O+]#C[O-]`** (owner: "CO2는 아주 간단한 화학종인데
#   이걸 왜 triple로 표기하게 되는거지?").
#   **Refitting at 150 was measured and is not adopted.** Cells 57 → 75; `(2,1)` then reads
#   `Double` at both 1.145 and 1.198 Å, as it should. Holdout 6,793:
#       T3 `Double` .7748 → .7753 · T6 .9865 → .9867   (better)
#       T8 `Triple` .7228 → **.7183** · T5 .9795 → .9793 · T3 `Triple` .9772 → .9771  (worse)
#       `Sq_L` .8587 · `OS` .8938 · violations .0125   (**unchanged**)
#   ⇒ it buys nothing on the two metrics this was meant to fix, because `QSHIFT` already
#   repairs the CO₂ shape after ⑥, and it regresses T8 `Triple`. 60 gives the same numbers as
#   150. Revisit only with a fallback that respects the degree instead of the global prior.
LPCOND_NMIN = int(os.environ.get("LPCOND_NMIN", "300"))
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

# `AGOC` — `C–H···M` 아고스틱 접촉의 **탄소 쪽** M–L 후보도 지운다 (Å 절대 문턱; 0 = off).
#   `drop_agostic` 은 M–**H** 만 지우는데, 아고스틱에서 금속은 탄소의 네 번째 결합손을 차지한 것이
#   아니라 **이미 있는 C–H 결합의 전자쌍을 빌려 쓴다.** 남은 M–C 를 `sigma` 로 세면 그 1 이
#   탄소를 질식시켜, `SATML` 이 (규칙대로) π 를 막고 방향족이 깨지며 카바니온이 생기고 금속
#   산화수가 2 밀린다. 판정은 기하만 쓴다 — `d(M,H) < d(M,C)` **및** `d(M,H) < AGOC`.
#   ⚠️ 떼면 그 조각이 금속에서 완전히 떨어지는 (금속, 조각) 짝은 손대지 않는다 — 실측 30.9% 가
#   그 경우이고, 거기서 떼면 «해리했다» 는 더 나쁜 오류가 된다.
#   **On by default (2.0 Å).** 오너 판단으로 채택 — *"CSD 에서 이 결합을 Single 로 부르더라도
#   agostic 에 해당된다면 해당 label 을 지우는 게 맞다. agostic interaction 은 지금 우리가 맞출
#   수 있는 대상도 아님."* 그런데 실측해보니 **대가가 거의 없다**:
#     holdout 6,793 — 출력이 바뀐 구조 **6 개**, `Σq_L` 틀→맞 0 · 맞→틀 0
#       T4 .9918 → .9917 · T5 .9795 → **.9796** · T3 `Double` .7749 → .7747
#       T1 · T3 나머지 세 클래스 · T6 · T8 세 클래스 · `Σq_L` .8587 · `OS` .8938 · 위반 .0125
#       — 전부 **불변**
#   앞서 잰 "93.8%(30/32) 가 `Single`" 은 더 **넓은** 조건(H 가 더 가깝 + 포화)이었다. 절대 문턱
#   2.0 Å 을 걸면 CSD 결정 구조에서는 거의 걸리지 않는다 — 활성화 직전 기하가 없기 때문이다.
#   Gold-DIGR (800 반응): 프레임 **1.12%** 의 답이 바뀌고, **분자가 쪼개진 프레임 0 건**(가드가
#   의도대로 동작), ΔOS ≥ 2 반응 **264 → 255** (사라짐 13 · 새로 생김 4).
#   손판정 표: `A_dOS2_no_topo_change__01` Ru **+3 → +1** · `__03` Pd **+4 → +2**, 둘 다 반대
#   프레임과 일치하게 됐다.
AGOC = float(os.environ.get("AGOC", "2.0"))

# `ETA2NEAR` — η² 짝을 만들 때, **짝 원자에만** T4 거리 문턱을 이만큼(Å) 넓힌다.
#   `_eta2_pair` 의 docstring 은 "η² 는 **결합**의 성질이라 양 끝이 다 haptic 이다" 라고 하지만,
#   구현은 `b in xs` 로 **양 끝이 각각 T4 를 통과할 것**을 요구한다 — 원자 단위 조건이 다시 들어와
#   있는 것이다. 옆으로 붙은 알카인은 이 조건에 구조적으로 걸린다: 먼 쪽 탄소는 정의상 금속에서
#   더 멀다.
#   🔴 `10.1021_acs.organomet.8b00684__45_Int9-2` R 이 그 케이스다. Ni–C0 2.089 Å 은 받아들여
#   지는데 Ni–C1 은 **2.651 Å 대 문턱 2.594 Å — 0.057 Å 초과** 라 짝이 안 만들어지고,
#   `drop_eta1` 이 홀로 남은 태그를 σ 로 되돌리고, 그 σ 가 C0 의 원자가 4 중 1 을 먹어
#   `C0≡C1`(1.218 Å) 이 `Double` 로 내려가며 **탄소마다 −1** ⇒ Ni **+4**. 같은 접촉이 haptic 이
#   되는 P 프레임은 `C≡C` 에 Ni **+2** 로 옳게 나온다.
#   ⚠️ 반대 방향 — 이미 받아들인 Ni–C0 를 **버려서** σ 비용을 없애는 것 — 은 위험하다. 그 결합은
#   실재하고(Mayer 0.274), T4 는 파이프라인에서 가장 정확한 단계(F1 .9918)이며, 뒤 단계가 그것을
#   취소하면 0.01 Å 노이즈로 답이 뒤집힌다. 짝을 인정하는 쪽은 아무것도 취소하지 않는다.
#   ⚠️ 완화로 들어온 원자끼리만의 짝은 만들지 않는다 — 최소 한쪽은 T4 를 정식으로 통과해야 한다.
#   🔴 **측정 후 기각 (2026-09-10). 기본값 off.** 이 케이스는 고쳐지지만(R 이 Ni +4 → **+2** 로
#   P 와 일치) CSD 가 비싸게 받는다 — 정답지가 "그 탄소는 `Pi` 가 아니다" 라고 말한다.
#     holdout 6,793            기준      0.05 Å     0.10 Å
#       T4 결합 유무           .9918     **.9883**  **.9859**
#       T5 haptic              .9795     **.9512**  **.9358**
#       T6 η^k                 .9865     **.9687**  **.9609**
#       T8 `Double`            .7469     .7501      .7530
#       `Σq_L` · `OS`   .8587/.8938  .8579/.8946  .8570/.8942
#   T5 가 **−2.8%p**, T6 가 −1.8%p 다. 얻는 `OS` +.0008 로는 그 값을 못 낸다.
#   ⇒ **T4 쪽 레버는 양방향 다 막혔다.** 받아들인 결합을 버리는 것은 불안정해서 위험하고, 짝을
#   들이는 것은 CSD 규약과 정면으로 어긋난다. 이 케이스를 풀려면 T4 문턱이 아니라 **σ M–L 이
#   원자가를 먹는다는 사실 자체**를 다루는 규칙이 필요하다 — 상한은 하드 제약이고 전하는 소프트
#   비용이라, "삼중+중성" 과 "이중+전하 둘" 의 비교가 애초에 일어나지 않는 것이 근본이다.
ETA2NEAR = float(os.environ.get("ETA2NEAR", "0"))

# `ETAPI` — ⑥ 의 Kekule 매칭에서, **양 끝이 같은 금속에 haptic 인 결합에 π 를 얹는 가중치**.
#   haptic M–L 결합은 금속이 π 결합을 옆에서 잡은 것이므로, `Single` 위에 걸린 η² 는 그 자체로
#   모순이다. 그리고 이 출력을 학습 데이터로 쓰면 단순히 지저분한 정도가 아니라 **모델이
#   "haptic 은 single 에서도 나온다" 고 배운다.**
#   🔴 **제약이 아니라 가중치다.** `w` 는 ⑥ 의 최대가중 매칭 타이브레이크이므로, 이 보너스는
#   그것 없이도 완전 매칭이 존재할 때만 반영되고(= "순서를 바꾸는 게 문제가 없으면"), 교대
#   패턴이 감당 못 하면 조용히 포기된다. 4클래스 라벨 · haptic 집합 · 원자가 예산은 이미 다
#   정해진 뒤이므로 **다른 것은 아무것도 움직이지 않는다.**
#   ⚠️ **η² 에만 건다.** 더 높은 η 는 교대 자체가 요청을 금지하고, 출력은 이미 그 하한에 붙어
#   있다. 원자 `k` 개 고리의 최대 매칭은 `⌊k/2⌋` 이므로 `⌈k/2⌉` 개는 **반드시** `Single` 이다.
#   실측 (Gold-DIGR 600 반응 · `260910_haptic_on_single.py`) — 관측이 하한과 정확히 일치한다:
#       η⁵ Cp    고리 5 · 매칭 2 ⇒ 하한 **60%**  · 관측 60%   (659 결합)
#       η⁶ 아렌  고리 6 · 매칭 3 ⇒ 하한 **50%**  · 관측 50%   (106 결합)
#       η⁴ 33% · 관측 33%   ·   η³ 50% · 관측 49%
#       **η²      결합 1 · 하한 0%      · 관측 17%**  ← 유일하게 어긋나는 것 (59/350)
#   `ETAPI=1e5` 로 그 59 건이 **17 건(5%)** 이 된다. 남는 17 건은 4클래스가 하드 `Single` 이라
#   ⑥ 이 손댈 수 없다. holdout 6,793: **T1·T3 네 클래스·T4·T5·T6·T8 세 클래스·Σq_L·OS·위반
#   전부 소수점 넷째 자리까지 동일** — 공짜다.
#   📌 소비자에게: π 소속은 `bonds_kekule` 가 아니라 **`bonds_4class`(`Conj`)** 로 읽어야 한다.
#   정수 차수는 이 교대를 손실 압축한 것이라, 거기서 "haptic 은 single 에서도 나온다" 를 배우는
#   것은 표기의 한계를 화학으로 오해하는 것이다. `Single` 로 나온 628 건 중 570 건이 `Conj` 다.
#   `ETAEXO` 의 1e6 보다 작게 두어, 둘이 부딪히면 `ETAEXO` 가 이긴다. 0 = off.
ETAPI = float(os.environ.get("ETAPI", "100000"))

# `QSHIFT` — ⑥ 뒤에서, 같은 부호 전하 2개가 교대 경로의 양 끝에 있으면 π 를 그 경로로 옮겨
#   전하를 상쇄한다 (`charge.formal.shift_pi_to_cancel`).
#   같은 결합 집합 위에 |전하| 가 더 작은 **유효한** 배치가 존재하는데 솔버가 그것을 고르지 못한
#   경우만 건드린다 — 캡을 넘기지 않고, 배위하는 음이온 자리가 둘이 아니고, 전하 합이 실제로 줄 때만.
#   오너 지목 (`10.1021_jo802516k__05`): `C0⁻–C2=C3–O4⁻` (조각 −2 · Au **+3**) 가 R 프레임에서는
#   `C0=C2–C3=O4` (조각 0 · Au **+1**) 로 나온다. **골격도 M–L 도 하나도 안 변했는데 산화수만 두
#   단계 뛴다** — 산화적 부가 없이 Au(I)→Au(III) 는 화학이 아니고, 반응 예측 모델에는 "아무 일도
#   없었는데 그 방향으로 반응이 일어난다" 고 가르치는 것이라 가장 유해한 부류다.
#   🔴 `QCOST` 를 16 까지 올려도 안 움직인다 — 이 선택은 ② 의 Double 매칭이 아니라 ③ 에서 하드
#   클래스로 굳기 때문이다. 그래서 ⑥ **뒤**에서 되돌린다. 오너가 지시한 "1단 전하 비용 최소
#   해집합 → 2단 그 안에서 거리로 선택" 을 출력 단계에서 강제하는 형태다.
#   ★ 같은 부호 쌍만이 아니라 **반대 부호 쌍**도 본다. 경로 결합 수가 홀수면 양 끝이 같은 방향,
#   짝수면 반대 방향으로 움직인다 — 그래서 `C⁻–C=C–O⁻ → C=C–C=O` (홀수) 와
#   **`O⁺≡C–O⁻ → O=C=O`** (짝수, CO₂) 가 같은 규칙으로 처리된다.
#   🔴 **`k = 1` 은 절대 건드리지 않는다.** 두 원자 사이 결합이 하나뿐인 양쪽성 이온은 그 표기가
#   맞는 것이다 — 일산화탄소 `[C⁻]≡[O⁺]` · 아민 옥사이드 `R₃N⁺–O⁻` · 인 일리드 `R₃P⁺–C⁻`.
#   여기서 차수를 옮기면 금속 카보닐 전체가 무너진다.
#   **On by default.** holdout 6,793 (`38087e4` 대비):
#     `Sq_L` .8553 → **.8587** · `OS` .8899 → **.8938** ·
#     T1/T3 네 클래스/T4/T5/T6/T8/위반 **소수점 넷째 자리까지 전부 불변**
#     (같은 부호만 볼 때는 .8570 / .8913 이었다 — 반대 부호 확장이 그만큼 더 얹는다)
#   Gold-DIGR (표본 800 반응 · 1,600 프레임): 12 프레임(0.75%)에서 발동하고 **전부** 금속
#     산화수를 옳은 방향으로 2 내린다 — `Pd +5 → +3` (존재하지 않는 산화수) · `Pt +4 → +2` ·
#     `Mo +6 → +4` · `Os +2 → 0` · `Ru +4 → +2` · `V +4 → +2` · `Ni +3 → +1`.
#   ⚠️ **배위 게이트가 없으면 반대로 망가진다.** 두 음이온 자리가 둘 다 금속에 배위하면 그것은
#      잘못 놓인 π 가 아니라 진짜 이음이온 킬레이트다 (다이싸이올렌 `[S⁻]C(R)=C(R)[S⁻]` ·
#      벤젠-1,2-다이싸이올레이트 · 카테콜레이트 · 아미디네이트). 게이트 없이 재면 holdout 145
#      구조가 바뀌어 `Σq_L` 맞→틀 **15** 대 틀→맞 4, `Sq_L` .8458 · `OS` .8802 로 떨어진다.
QSHIFT = os.environ.get("QSHIFT", "1") == "1"

# `SATML` — rule A's pi-headroom test counts the **sigma M-L bonds** as well as the internal
#   degree: `deg(X) + b_ML(X) >= CAP(X)` keeps X out of the ring that rule A pins `Conj`.
#   Why. A ring carbon carrying an H *and* a sigma bond to the metal already has four sigma
#   bonds and is sp3, but the ring is still planar enough to pass `TAU_P`, so rule A pinned it
#   anyway -- and a **pinned** bond is not the ④ matching's to move, so nothing downstream
#   could take the pi back. That is where `b_int(X) + b_ML(X) > CAP(X)` was coming from.
#   Measured (CSD holdout 6,793): 42 structures carried exactly this violation, and in **every
#   one** the reference calls both ring bonds at that atom `Single` -- KOZFIS C13/C39, ZEJNUZ
#   C8/C16, AGOJOW C0, all `C(H)(N)(N)->M` aminal carbons (`260910_sigcap_decline_audit.py`).
#   **On by default.** Holdout, over `QCOST=1 KEKQ=8 HALW=.3` + the boron sextet:
#     T3 `Single` .9905 -> .9906 · `Double` .7747 -> .7748 · `Triple` .9772 (=) ·
#     `Conj` .9615 -> **.9618** · T5 .9784 -> **.9795** · `OS` .8910 -> .8899 ·
#     `Sq_L` .8553 (=) · violations .0125 (=)
#   ⚠️ **The cost is on Gold-DIGR: step2 reactions 7,435 -> 7,113 (-322).** `pi_suppressed`
#      rises 1,832 -> 2,030 structures. Those are **aryl C-H sigma-complexes**: an approaching
#      metal gets a sigma M-C to a ring carbon that still carries its H, `SATML` then correctly
#      refuses it the pi, and the ring dearomatizes (`[C-][C-]`, fragment 2 too negative).
#      The M-C bond is the thing that is wrong there -- it is an eta-2 (C,H) interaction, not a
#      sigma bond -- and `drop_agostic` cannot see it because it only removes M-**H**.
#      Sampled 300 reactions: 13 frames, **13 of 13** an aromatic C-C at 1.37-1.45 A with both
#      atoms `deg 3` (`260910_satml_pi_suppressed.py`). ⇒ the eta-2 (C,H) rule is the follow-up.
SATML = os.environ.get("SATML", "1") == "1"

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
#   🔴 **Half of that argument is now known to be wrong** (2026-09-10). The violation says
#   the *atom* has too much; `SIGCAP` relabels the *bond*, so the tally stops counting the atom
#   while the atom keeps its bond orders. Two things follow.
#     (a) With no second atom to make a face out of it writes **eta-1** -- a sigma bond under
#         another name, which `drop_eta1` exists to forbid. `drop_eta1` runs *before* this
#         block, so those are never re-checked: on Gold-DIGR (400 reactions) **25 of 25** eta-1
#         fragments came from here, none from R7 or the angle test
#         (`260910_boron_and_eta1.py`). 7.8% of reactions carry one.
#     (b) Of the violations it removed on the CSD holdout, **42 were our own ③ error** -- rule A
#         pinning an sp3 aminal carbon `Conj` -- and the reference calls the M-L bond `Single`
#         in all 42. `SATML` now fixes those at the root, and holdout violations stay .0125
#         without `SIGCAP` doing the work.
#   What keeps it on: the *other* class it catches is real -- an aryl C-H sigma-complex, where
#   the ring genuinely is aromatic and the M-C genuinely is not a plain sigma bond.
#   🔴 **And `SATML` shrank the eta-1 problem to almost nothing on its own** (re-measured
#   2026-09-10 on the shipped path): eta-1 is **0.42% of Gold-DIGR reactions** (5 of 1,200), not
#   the 7.8% measured *before* `SATML`. Turning `SIGCAP` off takes eta-1 to 0 but costs **418
#   reactions and 1,002 valence-violating rows** (7,113 -> 6,695). And the 5 that remain have
#   **no H on the carbon** (0 of 3 audited), so an eta-2 (C,H) rule would not reach them either
#   -- that follow-up is **dropped, not deferred**. ⇒ **left as is.**
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

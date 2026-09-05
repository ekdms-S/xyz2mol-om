"""Constants and adoption flags used in the decisions — **these values are the whole pipeline**.

⚠️ **Ported from `ognm-bh-workspace/code/analysis/scratch/260830_fit_t10_charge.py`**
(2026-09-03). Function bodies were moved **verbatim** — the decision rules are unchanged.
"""

# ruff: noqa: E501
from __future__ import annotations

import os
import re
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data"


CONJ = {"Aromatic", "Delocalised"}
CLS = {"Single": 0, "Double": 1, "Triple": 2}
ORD = [1.0, 2.0, 3.0]
VAL = {
    "H": 1,
    "B": 3,
    "C": 4,
    "N": 5,
    "O": 6,
    "F": 7,
    "Si": 4,
    "P": 5,
    "S": 6,
    "Cl": 7,
    "As": 5,
    "Se": 6,
    "Br": 7,
    "Te": 6,
    "I": 7,
}
FULL = {"H": 2}  # filled-shell quota. 8 by default, 2 only for H
METALS = set(
    "Ti Zr Hf Nb Ta V La Sc Y Ce Cr Mo W Mn Re Fe Ru Os Co Rh Ir Ni Pd Pt "
    "Cu Ag Au Zn Al Ga In Sn Pb Mg B".split()
)
# ★★ center-atom decision (2026-09-03) — **`B` is conditional.**
#   rule  i is a center ⟺ el[i] ∈ METALS \ {B}
#                       OR el[i] = B AND the structure contains **no** METALS \ {B} atom
#   In `B₂H₆` and pure boranes `B` is a center; inside a transition-metal complex (carborane,
#   boryl, `BH₄⁻`) it is a **ligand atom**. Evidence (measured on CSD train reference labels,
#   2026-09-03): the only metal-class element with internal bonds is `B` (30,628 bonds ·
#   1,583 structures) — `Al`, `Zn`, `Sn`, `In`, `Pb`, `Ga`, `Mg` have 0 internal bonds.
#   ⚠️ The body must **stay identical to** workspace `260830_fit_t10_charge.centers`.
METALS_HARD = METALS - {"B"}


def centers(el):
    """The **set of center-atom indices** ([design doc] §3.0 0). Exactly the rule above."""
    hard = {i for i, e in enumerate(el) if e in METALS_HARD}
    return hard or {i for i, e in enumerate(el) if e in METALS}


HUCKEL = [2, 6, 10, 14, 18]

USE_DINT = os.environ.get("USE_DINT", "0") == "1"
# ★ adopted option D ([design doc] §5.0.11 ⑭) — on by default. USE_D3=0 falls back to the old
#   T2 gate path.
USE_D3 = os.environ.get("USE_D3", "1") == "1"
# ★ adopted option D_eht (owner confirmed 2026-09-01) — on by default. USE_EHT=0 = old D path.
USE_EHT = os.environ.get("USE_EHT", "1") == "1"
EHT_CACHE = os.environ.get("EHT_CACHE", "")  # fragment-charge cache CSV (recomputed each time
#                                             if absent)

T8FORM = os.environ.get("T8FORM", "thr")

TAU_P, TAU_E, LAM, MAX_ITER = 0.05, 0.02, 10.0, 50  # ⑩ values fixed by CV
SP2_EL = {"C", "N", "O", "S", "B", "P", "Se"}
RCOV = {
    "H": 0.31,
    "B": 0.84,
    "C": 0.76,
    "N": 0.71,
    "O": 0.66,
    "F": 0.57,
    "Si": 1.11,
    "P": 1.07,
    "S": 1.05,
    "Cl": 1.02,
    "As": 1.19,
    "Se": 1.20,
    "Br": 1.20,
    "Te": 1.38,
    "I": 1.39,
}
VMAX = {
    "H": 1,
    "B": 4,
    "C": 4,
    "N": 4,
    "O": 3,
    "F": 1,
    "Si": 4,
    "P": 5,
    "S": 6,
    "Cl": 1,
    "As": 5,
    "Se": 6,
    "Br": 1,
    "Te": 6,
    "I": 1,
}

ORD4 = [1.0, 2.0, 3.0, 1.5]  # 0 Single · 1 Double · 2 Triple · 3 Conj
VTGT = {
    "H": 1,
    "B": 3,
    "C": 4,
    "N": 3,
    "O": 2,
    "F": 1,
    "Si": 4,
    "P": 3,
    "S": 2,
    "Cl": 1,
    "Br": 1,
    "I": 1,
    "Se": 2,
    "As": 3,
    "Te": 2,
}
LAM_LO = 10.0  # under-valence penalty (not applied to coordinating atoms — M–L absorbs it)

CAP = {  # lone-pair capacity — `2·b_int + 2·lp = 8` ⇒ `b_int + b_ML <= 4` (1 for H)
    "H": 1,
    "B": 4,
    "C": 4,
    "N": 5,
    "O": 4,
    "F": 4,
    "Si": 6,
    "P": 6,
    "S": 6,
    "Cl": 7,
    "As": 6,
    "Se": 6,
    "Br": 6,
    "Te": 6,
    "I": 6,
}

# ★ `CAPSET` — variants of the `CAP` ceiling (owner remark 2026-09-02, [design doc] §5.4.1 C-2).
#   Default stays at the current `octet`.
#   The present table is **an octet ceiling, not a valence cap** (`2b + 2lp = 8 ⇒ b ≤ 4`) — with
#   `lp` free down to 0, every period-2 element lands at 4 (N 5). `CAP(O)=4` allows O²⁺ and
#   `CAP(N)=5` allows pentavalent N. Solving octet and formal charge together, the correct
#   ceiling is **`b_max = 8 − v + q`**:
#       C 4 · N 3(+1 → 4) · O 2(+1 → 3) · F 1(+1 → 2)
#   `tight` = the ceiling with cations allowed · `mid` = tightens only O and N, leaves F alone.
_CAPSET = os.environ.get("CAPSET", "octet")
if _CAPSET == "tight":
    CAP = dict(CAP, O=3, N=4, F=2)
elif _CAPSET == "mid":
    CAP = dict(CAP, O=3, N=4)
elif _CAPSET != "octet":
    raise SystemExit(f"CAPSET={_CAPSET!r} must be one of octet|mid|tight")

EHT_CUTOFF = -10.0
_EHT_VE = {
    "H": 1,
    "B": 3,
    "C": 4,
    "N": 5,
    "O": 6,
    "F": 7,
    "Si": 4,
    "P": 5,
    "S": 6,
    "Cl": 7,
    "As": 5,
    "Se": 6,
    "Br": 7,
    "Te": 6,
    "I": 7,
}

RULEA = os.environ.get("RULEA", "ge5")
if RULEA not in ("ge5", "eq6", "off"):
    raise SystemExit(f"RULEA={RULEA!r} must be one of ge5|eq6|off")

R2CONJ = os.environ.get("R2CONJ", "1") == "1"
# 🔴 R3 (2026-09-02, trial) — R2 blocks **only bonds attached to a heteroatom**. The `C=C` of a
#    pyrrole or imidazole 5-ring is carbon-carbon, so it is not caught and leaks into `Conj`;
#    that is what the 6,171 `Double` errors against the [design doc] §8 reference labels are.
#    R3 keeps **the whole 5-ring containing an R2-flagged atom** as Kekule.
#    ⚠️ For 5-membered heterocycles that CSD records as `Aromatic` this can go the other way and
#       produce errors — only a measurement will tell.
R3RING = os.environ.get("R3RING", "1") == "1"  # ★ adopted 2026-09-02
# R3 scope — all (any donor) · **N (nitrogen donors only · adopted)** · mono (1 donor + all the
#   rest carbon)
#   measured (CV · CVPOOL 26,075 · [design doc] §8 scoring):
#                                        Double  all .5586 · N .5297 · mono .4226
#                                        Sq_L    all .7796 · N .7932 · mono .7914
#   ⇒ N buys 82% of all's gain for 37% of its cost.
R3MODE = os.environ.get("R3MODE", "N")  # ★ adopted scope = nitrogen donors only
# 🔴 ROP — the **second dimension** of the T3 likelihood (2026-09-02). Distance cannot separate
#    Double from Conj in `C–C` (best 1-D threshold F1 0.4473 vs ROP 0.6171 · n = 1,500 ·
#    in-sample upper bound). All three angle variants (dihedral, bond angle, out-of-plane
#    deviation) matched the trivial baseline, i.e. zero separating power.
# 🔴 R4 (2026-09-02, trial) — a **4n all-carbon ring (4- or 8-membered)** is antiaromatic and
#    therefore not delocalized (neutral COT is the tub-shaped D2d form). 1,269 of the bonds we
#    wrongly call `Conj` are here.
#    ⚠️ η⁸-COT²⁻ is a **planar 10π aromatic** and must be excluded ⇒ apply only when non-planar.
R4RING = os.environ.get("R4RING", "1") == "1"  # ★ adopted 2026-09-02
# 🔴 R5 (2026-09-02, trial) — **a `Conj` fragment of a single bond is not delocalized.**
#   `Conj` requires at least 2 bonds to hold. In the error breakdown, "acyclic isolated double
#   bond marked `Conj`" accounted for 3,411 bonds (22% of the errors whose truth is `Double`).
#   Demoting them lets `_solve_cap` pick an integer (S/D/T) from likelihood and constraints.
#   0 parameters.
#   measured (CV · `CVPOOL` · original CSD reference labels): `Double` .6682 → **.6913** ·
#   `Conj` .9593 → .9612 · leakage 10,274 → 8,752 · `Sq_L` .7916 → .7932 · `OS` .8044 → .8054
#   (charge even rises slightly).
#   ⚠️ **Putting the rule **outside** `conj_forbidden` means the CV path never sees it** —
#      `260831_propagation_prior_cv` does not call `predict_T3_EHT`; it runs its own path
#      (`conj_lik` -> `solve_cap`). R5 was added to both. The first version fell into this trap
#      and **measured zero effect.**
R5SOLO = os.environ.get("R5SOLO", "1") == "1"  # ★ adopted 2026-09-03
# ★ `QHV` — generalization of the hypervalent charge formula (2026-09-03 · default off · for the
#   rule see the `q_atom` comment)
QHV = os.environ.get("QHV", "1") == "1"  # ★ adopted 2026-09-03
# ★ `R6SWAP` — **for same-element bonds on one center, distance order and bond-order order must
#   agree** (2026-09-03).
#   Sites where **two or more atoms of the same element** hang off one center — nitro
#   `N(=O)=O`, carboxylate `C(=O)O` — are currently scored **independently** per bond ⇒ the
#   longer bond can come out `Double` and the shorter one `Single`.
#   rule  swap ⟺ e1=(X,Y1) · e2=(X,Y2) · el[Y1] == el[Y2]
#               AND d(e1) < d(e2)  AND  ord(e1) < ord(e2)
#               AND after the swap neither Y1 nor Y2 exceeds `CAP`
#   ⚠️ **Being a swap, neither X's valence nor the fragment's total bond order changes** — it
#      does not break the step-⑤ EHT charge target.
#   `Conj` (1.5) is excluded. 0 parameters.
R6SWAP = os.environ.get("R6SWAP", "0") == "1"
# 🔴 `R7RING` — **restore an R2 donor inside a haptic ring as a π candidate** (adopted
#   2026-09-03 · on by default in the release).
#   rule  add (M,X) ⟺ X is an R2 donor (`_LP_DEG`: O·S·Se deg ≥ 2 · N·P deg ≥ 3, where deg is
#                     the number of ligand-**internal** neighbors · H included · M–L excluded)
#                  AND X belongs to a ring r with |r| = 5 (r from `nx.cycle_basis`)
#                  AND at least R7MIN = 2 of the **other atoms** of r passed T5 to the **same
#                      metal M**
#                  AND (M,X) is a T4 bond (d < d_bond AND w > w_veto)
#                  AND ∠(M–X–Y) < THETA_HAPTIC = 81.02°
#                      (Y = the internal neighbor of X whose bond midpoint is closest to M)
#   ⇒ **T3 bond orders (the 4 classes) are not changed.** It only waives T5's condition that
#     "X belongs to a π fragment".
#   Why: once R2/R3 make a 5-ring Kekule there are at most 2 double bonds, so **one atom drops
#       out of the π candidates** (η⁵ → η⁴). R2 is an element rule, so the same failure hits not
#       only pyrrole-type N but also **furan O · thiophene S · selenophene Se · phosphole P**.
#   It is fixed only in a stage **after** T3, so no DAG cycle appears in [design doc] §3.0.
#   0 new fitted parameters (R7MIN is on an integer grid).
R7RING = os.environ.get("R7RING", "1") == "1"  # ★ adopted 2026-09-03
# lower bound on the number of same-ring atoms that passed T5 to the same metal
R7MIN = int(os.environ.get("R7MIN", "2"))
# ★ `BMLSKIP3C` — exclude atoms taking part in a 3c2e bond from the `b_ML` budget of the ④
#   valence cap.
#   🔴 **default 0 (off). Rejected after measurement on 2026-09-03.**
#   [design doc] §3.1 ④ had long said *"exclude 3c2e-participating atoms and B"*, but **the code
#   never did.** So it was implemented and A/B-measured by CV over all of train, and it was
#   **equal or worse in every class**:
#       Single .9893→.9892 · **Double .7157→.7137** · Triple .9814→.9806 · Conj unchanged
#       Σq_L · OS · T6 identical to 4 decimals · structures with valence violations 715 → **713**
#       (2 fewer)
#   ⇒ **you pay `Double` −0.0020 to buy 2 violations.** Rejected, and the doc was made to match
#     the code.
#   **Why it loses** — the argument "a 3c2e bond puts one electron pair across 3 centers, so
#   counting it as 2 two-center bonds makes the constraint unsatisfiable" **only holds for μ-H,
#   and there the constraint is idle anyway** (H has 0 internal bonds, so `CAP(H)` binds no
#   internal order at all). What the budget was actually binding is the **3c2e carbon**
#   (a bridging C with 5 or more internal neighbors), and releasing the budget there raises the
#   internal order — over the full CV, 264 bonds (156 structures) changed, **209 of them
#   `Single`→`Double`**, and `Double` F1 went down (= most of those raises were wrong).
#   ⚠️ To turn it on, `BMLSKIP3C=1`. The rule is `pipeline.bridge_tags` · [design doc] §3.0 5c.
BMLSKIP3C = os.environ.get("BMLSKIP3C", "0") == "1"  # ⛔ rejected 2026-09-03 (measured)
# ★ `BML3C_COST` — the ④·⑥ budget an atom taking part in a 3c2e bond spends **in total,
#   regardless of how many M–L bonds it has** (2026-09-06).
#   One electron pair spanning 3 centers is worth **one** bond of valence — not two, not zero.
#   `BMLSKIP3C` above is the `0.0` end of this same knob, and it lost for exactly that reason:
#   releasing the whole budget gave a bridging methyl carbon headroom 1 and 209 `Single`→`Double`
#   raises followed. The midpoint is the chemically correct value.
#       μ-CO   internal 1 + cost 1 = 2  ⇒ headroom 2, `C≡O` stays  (per-bond ⇒ headroom 1, `C=O`)
#       μ-CH₃  internal 3 + cost 1 = 4  ⇒ headroom 0 = CAP(C)       (per-bond ⇒ 5, a violation)
#   ⚠️ measured here, not assumed: `0.0` and `1.0` give the **same** answer on Co₂(CO)₈ and
#   Al₂Me₆ — a μ-CH₃ cannot be raised under either, because its internal neighbours are H and
#   `CAP(H)=1` leaves the H side no headroom. The two part on bridging carbons with **carbon**
#   neighbours, which is where the 2026-09-03 CV found 209 wrong `Single`→`Double` raises for
#   `0.0`. `1.0` states the chemistry (one pair across three centers is one bond of valence)
#   instead of removing the constraint, so it should not buy those raises.
#   values  1.0 = one pair (default) · 0.0 = the old `BMLSKIP3C` · <0 = one per M–L bond
#           (the behaviour before this branch — keep it for A/B measurement)
#   ⚠️ **Not measured against the CSD reference labels yet.** Do that before merging to master.
BML3C_COST = 0.0 if BMLSKIP3C else float(os.environ.get("BML3C_COST", "1"))
# ★ T7 ([design doc] §3.0 5c) — the **normal valence** of a bridging atom. A `deg` above this
#   value is taken as 3c2e. H 1 · C·Si 4 · B 3. An element not in this table is not a 3c2e
#   candidate (= if it bridges, it is `dative`).
VALENCE_3C = {"H": 1, "C": 4, "Si": 4, "B": 3}
# ★ `GNEG` — lift the `g ≤ 0` rejection in the ④ exact solution **only when the fragment charge
#   demands a raise** (2026-09-03).
#   Since `q_frag = C0 + 2B`, a fragment with `q_EHT > q(all Single)` **must** have its bond
#   orders raised. But ④ only admits bonds with `score(Double) − score(Single) > 0` ⇒ when the
#   likelihood prefers `Single`, ④ cannot raise it even though the charge demands it, and the
#   later ⑤ greedily raises something arbitrary instead.
#   measured (stage dump): of 2,913 targets, **1,890 (64.9%) were caught by this filter and never
#   even became candidates.**
GNEG = os.environ.get("GNEG", "0") == "1"
# ★ `EHTMINFRAG` — **do not apply the ⑤ EHT fragment-charge target to fragments with fewer atoms
#   than this value** (2026-09-03 · default 0 = current behavior, applied to every fragment).
#   Evidence (measured · 2,916 targets · against the reference assignment): the EHT target error
#   rate by fragment size is **99% for 2 atoms (496/502)** · 39% for 3-9 atoms · 29% for 10+,
#   and the **fix rate on 2-atom fragment targets is 0.0%**. 2-atom fragments are nitrosyl
#   `M–N=O` and `N₂`-type, and in real cases (`GOFYOQ`, `MENKAR`) **both the likelihood and the
#   ④ cap solution get `Double` right and only ⑤ pushes it down to `Single`** (EHT target −3,
#   reference fragment charge −1). The error is systematic: 92% of it is −2.
#   ⚠️ This is a **structural gate, not a fitted threshold** — no new parameter is fitted.
EHTMINFRAG = int(os.environ.get("EHTMINFRAG", "0"))
# ★ `EHTSKIP` — do not apply the ⑤ EHT fragment-charge target **only to fragments of these
#   compositions** (2026-09-03).
#   composition key = the fragment's elements sorted and concatenated (`NO` · `SS` · `CCHH`).
#   Comma-separated.
#   Why composition rather than size (`EHTMINFRAG`) — measured (against the reference assignment ·
#   train · **all fragments, not conditioned on being a target**):
#       `CO`  22,298 fragments → error  1.7%   ← 2 atoms, yet almost never wrong
#       `NO`     493 fragments → error 99.8%   (error −2 in 489/492)
#       `SS`     118 fragments → error 94.9%   (error +2 in 112/112)
#       `CCHH`   142 fragments → error 66.9%   (error +2 in 91/95)
#   ⇒ turning off every 2-atom fragment (`EHTMINFRAG=3`) would also turn off the 22,298 `CO` —
#     and that lost in CV (`Double` .6913 → .6861 · `Σq_L` .7932 → .7908). So narrow it by
#     composition instead.
#   ⚠️ This is **a list of compositions chosen by measurement, not a fitted parameter**. Since
#      the error is fixed in one direction per composition, "correct the target by a constant"
#      is an alternative, but that adds one constant per composition.
# ★ adopted 2026-09-03
EHTSKIP = {v for v in os.environ.get("EHTSKIP", "NO,SS,CCHH").split(",") if v}
# ★ `LPCOND` — **condition the prior of the 4-class likelihood on the endpoint internal degrees**
#   (2026-09-03).
#   `lp[c] = ln P(c | element pair)`  →  `ln P(c | element pair, (deg_x, deg_y))`. `med` and
#   `scl` are untouched. If a cell has fewer than `LPCOND_NMIN` samples it **falls back to the
#   element-pair global**. The degrees come from T1 (internal bonds, CV F1 1.0000), so this is
#   not reference-label leakage.
#   Why — the margin of a `Double`→`Single` error was a distance term of +0.79 cancelled by a
#   prior of −1.13. Globally `C–O` is `Single .417 / Double .111`, but in the cell
#   `deg(C)=3, deg(O)=1` (carbonyl) it is `.344 / .322` and the penalty disappears.
#   ⚠️ `LPCOND_NOCONJ` — return only `Conj` (class 3) to the global prior. Conditioning every
#      class makes `P(Conj) = .908` in the `C–C` deg3-deg3 cell drive **`Double`→`Conj` leakage
#      up by +505** (CV measured 8,752 → 9,257). Excluding `Conj` instead brings it down to
#      **8,549**.
LPCOND = os.environ.get("LPCOND", "1") == "1"  # ★ adopted 2026-09-03
LPCOND_NOCONJ = os.environ.get("LPCOND_NOCONJ", "1") == "1"  # ★ adopted 2026-09-03
LPCOND_NMIN = int(os.environ.get("LPCOND_NMIN", "300"))
# prior temperature — `score = distance term + LPA·ln P(c)`. 1.0 = current · 0.0 = `D_flat`
# (rejected).
LPA = float(os.environ.get("LPA", "1.0"))
# 🔴 `EHTCOST` — **do not treat the EHT fragment-charge target as an absolute command**
#   (2026-09-03).
#   measured: **5.8% (5,434)** of fragments have an EHT target that disagrees with the reference
#   assignment, and those fragments have a bond error rate of **8.6% vs 3.2%** (2.7x) and
#   **59.7% carry at least one error** (11.0% for agreeing fragments). Excess errors ≈ **6,500
#   bonds**. To hit its target (−2, truth 0), `AFOKAH` changed **6** bonds including a 1.295 A
#   `C=N`.
#   ⇒ if the **likelihood cost** of meeting the target exceeds the threshold, that fragment gives
#     up the target and reverts. −1 = unlimited (old behavior).
EHTCOST = float(os.environ.get("EHTCOST", "-1"))
# 🔴 `LNORM=1` — include the **normalization term `−log(2·scl)`** of the Laplace log posterior
#   (2026-09-03).
#   The current formula omits that term, so **a class with narrow spread gets no reward.** `C=O`
#   (whose bond-length distribution is narrow) is the beneficiary. 0 parameters · the
#   statistically correct form.
#   ⚠️ The direction can be opposite per task — `Conj` also has a narrow scl (`C–C` 0.0089) and
#      gains just as much.
#   measured (2026-09-03): applied everywhere, **the direction flips pair by pair** — the intended
#   effect works (`C=O`→`Single` errors 851→625, −26.6%) but `Conj` over-calling grows by
#   `C–C` +352 · `C–N` +285 · `C–O` +203, a net worsening of +783. `Conj`'s narrow `scl` is not a
#   real bond-length distribution but **an artifact of aromatic ring lengths being uniform**, so
#   rewarding it through Laplace was wrong to begin with.
#   ⇒ `LNORM=2` — apply the normalization term **to S/D/T only** and exclude `Conj`.
LNORM = os.environ.get("LNORM", "0")
LNORM_ON = LNORM in ("1", "2")
LNORM_SKIP_CONJ = LNORM == "2"
# ⛔ not adopted — on top of R3+R4 the gain is within fold variance (Double +0.0014 · x2m variant
#   +0.0018). Alone it is +0.059, but it **targets the same errors as R3 and R4**. There is no
#   case for spending one more parameter.
USE_ROP = os.environ.get("USE_ROP", "0") == "1"
ROPW = float(os.environ.get("ROPW", "1.0"))
_LP_DEG = {"O": 2, "S": 2, "Se": 2, "N": 3, "P": 3}

NAMEEL = {
    "Ti": "titanium",
    "Zr": "zirconium",
    "Hf": "hafnium",
    "V": "vanadium",
    "Nb": "niobium",
    "Ta": "tantalum",
    "Cr": "chromium",
    "Mo": "molybdenum",
    "W": "tungsten",
    "Mn": "manganese",
    "Re": "rhenium",
    "Fe": "iron",
    "Ru": "ruthenium",
    "Os": "osmium",
    "Co": "cobalt",
    "Rh": "rhodium",
    "Ir": "iridium",
    "Ni": "nickel",
    "Pd": "palladium",
    "Pt": "platinum",
    "Cu": "copper",
    "Ag": "silver",
    "Au": "gold",
    "Zn": "zinc",
    "Sc": "scandium",
    "Y": "yttrium",
    "La": "lanthanum",
    "Ce": "cerium",
    "B": "boron",
    "Al": "aluminium",
    "Ga": "gallium",
    "In": "indium",
    "Sn": "tin",
    "Pb": "lead",
    "Mg": "magnesium",
}
ALT = {
    "Fe": ["ferr"],
    "Cu": ["cupr"],
    "Au": ["aur"],
    "Ag": ["argent"],
    "Sn": ["stann"],
    "Pb": ["plumb"],
    "Ni": ["nickel"],
    "Pt": ["platin"],
    "Mn": ["mangan"],
    "Al": ["alumin"],
}
ROMAN = {"0": 0, "i": 1, "ii": 2, "iii": 3, "iv": 4, "v": 5, "vi": 6, "vii": 7, "viii": 8}
R = r"(?:0|i{1,3}|iv|vi{0,3})"
PAT, PATM = re.compile(rf"([a-z]+)\(({R})\)"), re.compile(rf"([a-z]+)\(({R}(?:,{R})+)\)")

# ★ T5 — angle threshold for the haptic decision (1 global value · [design doc] §3 4a).
#   This constant originally lived in the scoring script (`260831_propagation_prior_cv.py`).
THETA_HAPTIC = 81.02

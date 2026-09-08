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
# R3 scope — **all (any R2 donor · adopted 2026-09-08)** · N (nitrogen only) · mono (1 donor,
#   the rest carbon)
#   Why `all`: R3's argument is that R2 leaves the ring's C–C bonds behind, and that is true for
#   furan O and thiophene S exactly as it is for pyrrole N. Restricting it to N was never derived
#   — it came from one CV table.
#   Measured on holdout 6,793 (48 shards · `260907_deploy_full_score.py`). The `Sq_L`/`OS` columns
#   are given **both ways**, because the ligand charge moved onto the Kekule integers on the same
#   day and that is what decided this:
#                                    N (old)        all (adopted)
#     harmful `Double` errors          305            **302**
#     T3 `Double`                     .7420          **.7682**   (+223 bonds)
#     T3 `Conj` · `Single`            .9583 · .9901  .9611 · .9904
#     valence violations              .0302          .0303
#     -- charge counted on the 4-class values (the old way) --
#     `Sq_L` · `OS`                   .8398 · .8715  .8269 · .8669
#     reported != emitted charge        298            356
#     -- charge counted on the emitted Kekule integers (current) --
#     `Sq_L` · `OS`                   .8536 · .8823  **.8519 · .8831**
#     reported != emitted charge        214            **214**
#   🔴 **`all` looked expensive only under the old count.** Counting on the Kekule integers, the
#   `Sq_L` difference shrinks to 2 structures, `OS` comes out *better*, and the reported-vs-emitted
#   mismatch is identical — while `all` keeps its 223 extra correct `Double` bonds.
#   Chemistry, not just the metric: the O-only donor five-rings, 458 of them on holdout, carry an
#   average of **0.10 aromatic bonds out of 5** in the reference — they really are Kekule.
#   (census: `dev/analysis/scratch/260907_r3_scope.py`)
#   ⚠️ `nomix` (like `all` but leaving a ring that mixes N with O/S alone) was measured and is
#      **not** better: harmful `Double` 302, `Sq_L` .8303, reported != emitted 349 under the old
#      count -- it recovered only 7 of the 58. The earlier finding that the loss sat in `C3NO`
#      rings was made before the 09-07/08 rules and no longer holds.
#   ⛔ The older CV numbers that favoured `N` (CVPOOL 26,075, 2026-09-02 scoring: `Double`
#      all .5586 / N .5297 · `Sq_L` all .7796 / N .7932) predate every rule adopted on 09-07/08.
R3MODE = os.environ.get("R3MODE", "all")  # ★ adopted scope = every R2 donor (2026-09-08)
#   `nomix` — like `all`, except a five-ring whose donors are **not all the same element**
#   (N together with O or S) keeps its `Conj`: an oxazole/thiazole really is aromatic, and
#   those are exactly the rings `all` gets wrong.
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
# ★ The reported ligand charge is counted on the **emitted Kekule integers** plus the
#   residual `kekulize` returns, not on `ORD4[cls]` where a `Conj` bond is 1.5 (adopted
#   2026-09-08, owner's proposal).
#   Why: `_qfrag` summed `ORD4[cls]`, so an atom with three `Conj` bonds read `b = 3.5` and picked
#   up a formal charge of `-0.5` that **no emitted bond accounts for**. The per-atom charges
#   stamped into the SMILES were already counted from the Kekule integers (`api.predict`), so the
#   only thing on the 4-class basis was the ligand `charge` field -- which is why
#   `complex_smiles_ok` kept reporting "charge sum differs".
#   Worked example `MBTZRE01` (benzothiazole-2-thiolate, the correct charge is -1):
#       reported -3 · 4-class per-atom sum -2 · emitted Kekule **-1**
#   Two leaks add up there: atoms 11 and 20 read `b = 3.5` (`-0.5` each), and `frag_charge`'s
#   matching adds another `-1` for the conjugated component.
#   `frag_q` is added on top, so the charge a Kekule skeleton genuinely cannot express (an
#   even-ring dianion has a perfect matching and a neutral skeleton) is still reported.
#   Measured (holdout 6,793 · 48 shards) -- **no bond decision changes**, T1/T3/T4/T5/T6/T8 and
#   the valence-violation rate are all identical:
#       `Sq_L`                     .8398 → **.8536**
#       `OS`                       .8715 → **.8823**
#       reported != emitted charge   298 → **214**
#   For reference, feeding the CSD reference bond orders to the same charge rule gives `Sq_L`
#   .8528 — this metric was being held down by **how it was counted**, not by our bond orders.
#   No flag: counting a charge on half-integer valences is not a policy anyone would pick.
#   ⚠️ ⑤ still calls `_qfrag` (the 4-class count) to ask how far the fragment is from its
#      EHT target, because at that point no Kekule structure exists yet. That count has
#      the same -0.5 leak, so ⑤ can chase a delta that is off. **Not yet measured.**
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
#   Deprecated — it is the `0.0` end of `BML3C_COST` below; set `BML3C_COST=0` instead.
BMLSKIP3C = os.environ.get("BMLSKIP3C", "0") == "1"  # off by default
# ★ `BML3C_COST` — the ④·⑥ budget an atom taking part in a 3c2e bond spends **in total,
#   regardless of how many M–L bonds it has.** One electron pair spanning 3 centers is worth
#   **one** bond of valence.
#       μ-CO   internal 1 + cost 1 = 2  ⇒ headroom 2, `C≡O` stays  (per-bond ⇒ headroom 1, `C=O`)
#       μ-CH₃  internal 3 + cost 1 = 4  ⇒ headroom 0 = CAP(C)       (per-bond ⇒ 5, a violation)
#   values  1.0 = one pair (default) · 0.0 = release the budget (`BMLSKIP3C`) · <0 = one per M–L bond
#   ⚠️ **Not measured against the CSD reference labels yet.** Do that before merging to master.
BML3C_COST = 0.0 if BMLSKIP3C else float(os.environ.get("BML3C_COST", "1"))
# ★ T7 ([design doc] §3.0 5c) — the closed-shell budget (bonds + lone pairs) of a bridging atom.
#   A `b_use` above this value is taken as 3c2e. H 1 · C·Si 4 · B 3. An element not in this
#   table is not a 3c2e candidate (= if it bridges, it is `dative`).
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
# ══ ⑤ EHT target trust gate — `pipeline._eht_untrusted` ═══════════════════════════════════
#   One rule with three conditions, all saying the same thing: *for this class of fragment the
#   bare-fragment extended-Hückel charge is not a target worth chasing.* ⑤ then leaves the
#   fragment to the likelihood and the ④ solution. `EHTMINFRAG` (size) is above; the two
#   below are composition and local motif. Each is a **list chosen by measurement, not a
#   fitted parameter** — set either to empty to turn that condition off.
#
# ★ `EHTSKIP` — skip these whole-fragment compositions (2026-09-03).
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
# ★ `EHTNITRO` — skip a fragment holding a **nitro or nitrite group**: an N carrying **exactly
#   two terminal O** (2026-09-08). Same action as `EHTSKIP`, expressed as a local motif because
#   the group appears inside fragments of any composition.
#   Why: `EHTCOST` (capping what ⑤ pays) blocked only 15 of the 77 `02_eht_target` failures --
#   ⑤ pays a median of 0.82, so the likelihood barely resists. The fault is the target, not its
#   price: the EHT target disagreed with the reference in 72 of those 75.
#   Measured by motif (holdout · 31,412 fragments · reference = CSD labels through
#   `charge.kekulize`): the target is wrong 3.8% of the time overall, but for the 160 fragments
#   the detector fires on -- split by what the N actually is (`260908_nitro_motif_split.py`):
#       `R–NO2`  N deg 3, 2 terminal O, third neighbour C   64 fragments · **84.4% wrong**
#       free `NO2-`  N deg 2, 2 terminal O                  13 fragments · **100% wrong**
#       `X–NO2`  third neighbour not C                       8 fragments ·   37.5% wrong
#       `NO3-`   **three** terminal O                       75 fragments ·  **0.0% wrong**
#   🔴 Every error is exactly `-2` per group (67 of 70), so this is a systematic bias, not noise.
#   ⚠️ **Exactly two, not at least two.** The first version read `>= 2` and threw away the 75
#      nitrate fragments whose target is always right. Narrowing to `== 2` left the harmful-
#      `Double` count and its whole decomposition **unchanged at 305** (48 shards) while halving
#      the collateral -- so the wider detector was buying nothing.
#   ⚠️ Refitting the EHT constants is **not** the alternative: fitted on train (126,362
#      fragments) they move accuracy .9602 → .9631, holdout .9618 → .9632, nitro .5625 → .5875.
#      They come verbatim from xyz2mol_tm `get_proposed_ligand_charge` and are already near
#      optimal; the residual is the method (extended Hückel on a bare fragment). Correcting the
#      target by `+2` per group instead of dropping it is untested.
#   Measured (48 shards · deployment path · holdout 6,793):
#     harmful `Double` errors **424 → 307** (121 fixed, 4 broken) · `02_eht_target` **77 → 17**
#     `Double` .7272 → **.7317** · `Sq_L` .8372 → **.8398** · `OS` .8672 → **.8712**
#     violations .0302 and T1/T4/T5/T6/T8 unchanged · charge conservation 297 → 298
#     train 27,294  `Double` .7177 → **.7223** · `Sq_L` .8290 → **.8344** · `OS` .8583 → **.8640**
#     ablation from the shipped default: **+119** harmful `Double` — the single largest of the
#     adopted rules. Interaction with `ADJQVETO` is **-3** (both off 497 vs 500 additive).
# ★ adopted 2026-09-08
# ══════════════════════════════════════════════════════════════════════════════════════════════
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
# ★ ⑤ may not create a new pair of **adjacent same-sign formal charges** — a move with
#   `Δ > 0` is dropped from the candidate pool, so ⑤ takes the next-best move and, if there
#   is none, **gives up the target for that fragment** (④'s assignment stands).
#   **0 fitted parameters.** (proposal 1, 2026-09-07 · owner's proposal)
#   Why: of the 135 harmful `Double` errors that ④ got right and ⑤ demoted, **63 (47%) end
#   up with both ends of the bond negative** (`(−1,−1)` 52 · `(−2,−1)` 10). ⑤ moves a bond
#   ±1 to reach the EHT fragment-charge target, and pushing a `C=C` down makes `C⁻ C⁻`
#   (`IKOZUE` C6–C8: d 1.343 Å, likelihood margin `D−S` +8.18, ④ said `Double`).
#   `Δ` = (bonds incident to `e`'s endpoints whose two atoms both carry a nonzero formal
#   charge of the same sign) **after** the move minus **before**, floored at 0. Charges come
#   from the same Kekule matching the output converter uses (`atom_bond_sums`).
#   ⛔ **The soft form was built and rejected.** A weight only *reorders* candidates, so where
#      the offending move is the **only** candidate it is still taken. `IKOZUE` C6–C8 is
#      exactly that case — even a weight of `1e9` left it demoted, and on holdout weights
#      1/3/10 moved `Double` by −0.0003/−0.0005/−0.0005 (i.e. nothing).
#   Measured (deployment path · `260907_deploy_full_score.py`; the rule was **chosen on train**
#   and holdout only confirms it):
#     train 27,294   `Double` .6997 → **.7071** · `Σq_L` .8212 → **.8288** · `OS` .8488 → **.8532**
#                    · `Single` .9884 → .9886 · `Triple` .9758 → .9767 · `Conj` .9558 → .9558
#                    · correct internal bonds **+494**, no diagonal cell got worse
#     holdout 6,793  `Double` .7117 → **.7198** · `Σq_L` .8295 → **.8312** · `OS` .8586 → **.8618**
#                    · correct internal bonds **+126**
#     harmful `Double` errors (the target of the investigation) **630 → 563**, and the cause this
#       rule addresses, `02_eht_target`, **135 → 74** (74 fixed · 7 newly broken)
#     CRW deployment domain 22/22 unchanged · 23 package tests pass
#   ⚠️ The one cost: valence violations **+0.06%p** (train .0301 → .0307 · holdout .0299 → .0303)
#      — ⑤ demotions were sometimes relieving a cap that ④ had spent.
# ★ adopted 2026-09-07 (proposal 1)
# ══ Two defect fixes, not options — kept here for the evidence only ═══════════════════════
#   Both restore an invariant the implementation was breaking. There is no flag: the old
#   behaviour is not a policy anyone would choose. Reproduce it from git history if needed.
#
# ★ `is_cluster_frag` decides on the **Kekule integer** bond orders, not the 4-class ones
#   (2026-09-08).
#   Why: the cluster test is `b_int(x) > CAP(el[x])`, and a `Conj` bond counts **1.5**, so a
#   carbon with three of them reads 4.5 > 4 and an ordinary substituted arene is taken for a
#   multicentre cage -- its ligand charge then comes from EHT instead of the formal-charge sum.
#   The EHT number is often right by luck, so `Sq_L`/`OS` looked fine **while the emitted
#   structure disagreed with them**: `EJUJUP` reports `q_L = 0` / `OS(Cr) = 0` and draws two
#   carbanions, i.e. a complex whose formal charges sum to -2 against an input total of 0
#   (`complex_smiles_ok` already said so: "charge sum differs from total charge -2 vs 0").
#   Measured (holdout 6,793 - `260908_charge_conservation.py`):
#     cluster calls, fragment level        906 -> 159   (747 were artifacts of the 1.5)
#     structures whose emitted charges do not sum to the input total
#                                          383 (5.64%) -> **297 (4.37%)**   (93 fixed, 7 broken)
#     reported ligand charge != Kekule     318 -> 230
#     cost:  `OS` .8618 -> .8611 (2 structures) · `Sq_L` .8312 unchanged · T1/T3/T4/T5/T6/T8 all
#            unchanged
#   ⚠️ The remaining 298 have other causes and are **not** measured yet.
#   Ablation from the shipped default: harmful `Double` unchanged at 305, so this earns its
#   place on charge conservation (298 vs 380) and `OS` (.8712 vs .8694), not on the target.
# ★ The ⑥ Kekule matching is weighted by the **`Double` − `Single` distance likelihood**
#   (2026-09-08). No flag — `w` is the same dict that carries the `CAPINESS` promise, so a
#   flag here could not separate the two (see below).
#   Why: `frag_charge` calls `max_weight_matching(..., maxcardinality=True)` with **no weights**,
#   so among the several Kekule structures of the same cardinality the one that comes out is
#   unrelated to the bond lengths. `EMAXAR`'s conjugated fragment is two bonds -- the acyl `C=O`
#   (1.234 A) and the exocyclic `C-C` (1.440 A) -- and the unweighted matching took the `C-C`,
#   producing `O-` plus a ring carbanion.
#   The cardinality is unchanged (`maxcardinality=True` still holds), so this only breaks ties.
#   Measured (holdout 6,793 · 48 shards). 🔴 **Read the right metric**: T3 F1 scores
#   `bonds_4class`, and this changes only the stage *after* it, so T3 cannot see the gain. The
#   metric that moves is the harmful-`Double` target, which is defined on `bonds_kekule`:
#     harmful `Double` errors      563 -> **505**  (71 fixed · 13 newly broken)
#       of which `03_overconj`     112 -> **51**   · `02_eht_target` 74 -> 77
#     `PUDLEG` N3-C29 and `LEKJOB` C18-C20 now come out `Double`; `EMAXAR` does not (its
#       fragment scores the ring bond higher, +1.431 vs +0.949) and neither does `EJUJUP`
#       (blocked at ④, not ⑥).
#     cost: 4-class `Double` .7198 -> .7195 (3 bonds) · `OS` .8611 -> .8607 (1 structure) ·
#           `Sq_L` .8312 and charge conservation (297) unchanged · violations .0303 -> .0302
#     train 27,294: `Double` .7071 -> .7072 · `Sq_L` .8237 -> .8235 · `OS` .8500 -> .8498
#   🔴 **The first ablation of this was invalid** (2026-09-08, caught by codex review). The
#   ⑥ gate read `if (CONJW or CAPINESS) and w:`, so with `CAPINESS=1` setting `CONJW=0`
#   left the weighted matching running and every metric came out identical — that was
#   evidence the switch did nothing, not that the weighting did nothing. Zeroing the
#   likelihood term while keeping the `-1e6` promise gives the real number:
#     harmful `Double` errors **305 → 333 (+28)** on holdout 6,793.
# `CAPINESS=1` — in the ④ cap budget, charge an atom `k` instead of `k+1` for its `Conj` bonds
#   **when the fragment still has a maximum matching that leaves that atom unmatched**
#   (2026-09-08, measuring only, off by default).
#   The `k+1` rule reads "k conjugated bonds, one of which is the pi bond". That is right for an
#   atom the Kekule matching *pairs up*, and one too many for an atom it leaves over -- and an
#   odd fragment always leaves one over. The over-charge is what blocks a legitimate double bond
#   somewhere else on that atom:
#     `EJUJUP` C7: `Conj` 1 -> 2.0 + O 1.0 + C15 1.0 = 4.0 = CAP  ⇒ headroom 0, and `C7=C15`
#       (1.330 A, `D−S` margin +8.38) never becomes a candidate edge. C7 sits on a 7-atom
#       fragment with max matching 3, so a maximum matching leaving C7 unmatched exists.
#     `EMAXAR` C13: same shape, blocking `C13=C14` (1.388 A).
#   Test: atom x is inessential in fragment F  ⟺  |M(F − x)| == |M(F)|. Fragments with a perfect
#   matching have no such atom and are skipped, so the extra matchings are only run where the
#   deficiency is non-zero.
#   ⚠️ This is the **feasibility** form Codex asked for, not a blanket relaxation: headroom is
#      granted only where an unmatched assignment actually exists. It is still not a guarantee --
#      ⑥ picks its own matching and may pair x up anyway. Watch the valence-violation rate.
#   Three guards, all necessary (each was measured after it was missing):
#     ① **at most `deficiency` grants per fragment** — a fragment leaves exactly
#        `|F| − 2·|M(F)|` atoms over, so granting more is a promise it cannot keep. Without this,
#        42 atoms newly broke the cap, **every one** in a fragment of deficiency 1 that had
#        granted 2-9 atoms.
#     ② **demand covers M–L too** — `r[x]` gates an internal `Double` *and* an M–L order raise.
#        Ranking on internal bonds alone sent the whole T8 `Triple` gain (.639 → .734) back to
#        baseline.
#     ③ **the promise is carried into ⑥** as `−1e6` on the granted atom's `Conj` edges, so the
#        Kekule matching leaves it unmatched (`maxcardinality=True` keeps the cardinality, and
#        `_inessential` already proved such a matching exists). Without this, 33 atoms broke the
#        cap — all of them in fragments of deficiency 1 with exactly 1 grant, i.e. keepable.
#   Measured (48 shards · deployment path):
#     holdout 6,793  `Double` .7195 → **.7272** · `Single` .9896 → .9898 · `Triple` .9765 → .9770
#                    T8 `Double` .7400 → **.7479** · T8 `Triple` .6391 → **.7343**
#                    `Sq_L` .8312 → **.8372** · `OS` .8607 → **.8672**
#                    valence violations .0302 → **.0302** (flat) · charge conservation 297 → 298
#                    harmful `Double` errors 505 → **424** (83 fixed, 2 broken);
#                    `01_likelihood` 320 → 249
#     train 27,294   `Double` .7072 → **.7177** · T8 `Triple` .6998 → **.7784**
#                    `Sq_L` .8235 → **.8290** · `OS` .8498 → **.8583** · violations .0309 → .0307
#   🔴 The T8 `Triple` jump is **one narrow systematic error**: all 20 recovered bonds are
#      carbynes `M≡C` (Mo 5 · W 6 · Re 4 · Os 3 · Ru 2) whose carbon holds exactly one `Conj`
#      bond. `AYEVAB` W0≡C1: `use` = 2 (`Conj` 1) + 1.0 (M–L) = 3.0 ⇒ headroom 1 ⇒ `Double` only;
#      charging 1 gives headroom 2 and the `Triple` goes in.
# ★ adopted 2026-09-08
# `ETAEXO=1` — in the ⑥ Kekule matching, an atom of an **η-coordinated ring** may only pair
#   **inside that ring** (2026-09-08, measuring only, off by default).
#   Why (owner): an η⁵-Cp coordinates because its π stays in the ring. A ring carbon that takes
#   an exocyclic double becomes a fulvene-type sp² centre and the ring can no longer bind η.
#   `FEDDID`: the chain alternation is shifted one bond, so `C12=C14` puts Cp carbon C14's π
#   outside the ring; the reference keeps `C10=C12` and leaves C14 as the ring carbanion.
#   Implemented like the `CAPINESS` promise -- `−1e6` on the offending edges, `maxcardinality`
#   untouched, so the count of double bonds cannot change.
#   Scope measured first: only **10 of the 424** harmful-`Double` targets sit in such a fragment
#   and the rule fixes **1**, but across holdout **147 of 1,749** η structures (8.4%) emit one.
#   ⛔ **Measured and rejected as a ⑥ rule (2026-09-08).** It is very nearly non-binding: the
#      scoring CSVs are byte-identical across all 48 shards, every metric matches to four
#      decimals, and the η census moves **147 → 145** — two structures.
#      The reason is where the offending bond is decided: this constraint can only touch edges
#      the ⑥ matching owns, i.e. `Conj` edges, and 145 of the 147 keep their exocyclic double
#      because it was already fixed at ③/④. A rule that actually reaches them has to live
#      there, not in the matching. The flag is left in place so that experiment starts from a
#      measured baseline rather than from scratch.
ETAEXO = os.environ.get("ETAEXO", "0") == "1"
# ★ In ④ the matching must not spend **two** capacity units on **one** bond
#   (2026-09-08, measuring only, off by default. Found by Codex, then measured).
#   The capacity trick copies an atom with headroom `r` into `r` replicas, and an internal bond
#   `e=(a,b)` gets a replica edge for **every** `(ia, ib)` pair. When `r[a] = r[b] = 2` the
#   matching can take `(a,0)-(b,0)` **and** `(a,1)-(b,1)`: two disjoint pairs, one physical bond.
#   Then `g` is counted twice in the objective (so `e` can crowd out another bond) and two units
#   of headroom are spent at each end — while the output records `out[e] = 1`, a single `Double`.
#   The docstring's "exact maximum-likelihood assignment" does not hold there.
#   Measured (holdout 6,793): **96** such double-selections out of 15,538 matched internal bonds
#   (0.6%), in 5,822 ④ matchings (`260908_solvecap_dup.py`).
#   The repair re-runs the matching: any bond the previous round selected is fixed at **one**
#   unit, its ends' headroom is charged once, and the freed unit is offered to the other bonds.
#   It stops when a round produces no double-selection (or after `CAPDUP_MAX` rounds).
#   Measured (48/12 shards · deployment path):
#     holdout 6,793  T3 `Double` .7317 → **.7417** · `Single` .9899 → .9901 · `Triple`/`Conj`
#                    unchanged · correct internal bonds **+131** (`Double`→`Double` +139)
#                    `Sq_L` .8398 · `OS` .8712 · violations .0302 — **all three unchanged**
#                    charge conservation 297 → 298 · harmful `Double` errors 308 → 305
#     train 27,294   `Double` .7223 → **.7327** · everything else unchanged
#   ⚠️ T8 pays a little: holdout `Double` .7479 → .7461 · `Triple` .7343 → .7317 (−8 correct M–L
#      bonds). Freeing the wasted unit changes what the M–L order optimization competes for.
#      Not decomposed.
#   ⚠️ The 4-class gain is 10x the harmful-error gain (+131 bonds vs −3 targets) because the two
#      are scored on different outputs — T3 F1 on `bonds_4class`, the target set on
#      `bonds_kekule` after the ambiguity, μ-CO and resonance filters.
# ★ adopted 2026-09-08
#   `CAPDUP_MAX` bounds the repair loop — an algorithmic safety limit, not a chemical rule.
CAPDUP_MAX = int(os.environ.get("CAPDUP_MAX", "6"))
# ══════════════════════════════════════════════════════════════════════════════════════════════
# `TAUD` — the ④ candidate gate for an internal `Double`. An edge enters the matching when
#   `g = s[Double] − s[Single] > −TAUD` (default 0.0 = the current `g > 0`).
#   Why a threshold rather than `LPA`: `LPA` scales the prior of **every** class at once, so
#   flattening it to help `Double` also removes the `Conj` and `Triple` priors -- the reason
#   `LPCOND_NOCONJ` exists at all. `TAUD` moves only the `Double`↔`Single` decision boundary,
#   which is the standard cost-sensitive form `g > log(C_FP/C_FN)`.
#   The block it targets: 116 of the 308 remaining harmful `Double` errors are bonds where
#   `Single` outscores `Double`, so ④ never creates the candidate edge at all -- no cap, matching
#   or EHT rule can reach them. In 73 of 112 the distance alone is nearer the `Double` median.
#   🔴 This is a **fitted** parameter, unlike every rule adopted so far. Fit on train, confirm on
#   holdout, and read `Single`/`Conj`/`Triple` F1, `Sq_L`, `OS` and valence violations with it --
#   raising `Double` recall necessarily costs `Single` precision.
#   ⛔ **Measured and rejected (2026-09-08).** Swept on train 27,294 over `CAPDUP=1`:
#     τ    0     0.5    1      2      3
#     D  .7327  .7333 .7311  .7285  .7260      ← peaks at +0.0006, inside fold noise, then falls
#     OS .8640  .8655 .8664  .8675  .8674      ← rises, but bought with `Double`
#     `Sq_L` and violations flat throughout.
#   The gate was never the obstacle. A first attempt that only widened the gate changed
#   **nothing at all** (τ = 0.5/1/2 byte-identical) because `max_weight_matching` never takes a
#   negative edge; the weight has to move too, and this is the corrected form. It still does not
#   work: τ lifts every `Double` candidate at once, so the true ones gain no ground on the false
#   ones inside the matching. The 116-case block is not a threshold problem.
TAUD = float(os.environ.get("TAUD", "0"))
# `CAPMILP=1` — solve ④ **exactly** instead of through the matching reduction (2026-09-08,
#   measuring only, off by default).
#   Why: the present ④ confirms `Triple` first -- local argmax with headroom 2 -- spends the
#   capacity, and only then matches `Double`/M–L, so it never weighs one `Triple` against
#   several `Double`s. Measured by solving the **same objective under the same constraints**
#   exactly and comparing: **113 of 12,245** ④ calls on holdout (0.92%) are not optimal, and the
#   objective lost is median **4.58** · mean 5.90 · max 23.07. A typical `D−S` margin is 0.8-8,
#   so a median loss of 4.58 is a whole bond decision.
#   The formulation is the current one written out, nothing added:
#     y[e,1], y[e,2] ∈ {0,1}   e is `Double` / `Triple` (both 0 = `Single`), y[e,1]+y[e,2] ≤ 1
#     z[key,u] ∈ {0,1}         M–L increment u, z[key,1] ≤ z[key,0]
#     max Σ (s[D]−s[S])·y1 + (s[T]−s[S])·y2 + Σ (sm1−sm0)·z0 + (sm2−sm1)·z1
#     s.t. use_base[a] + Σ_{e∋a}(y[e,1] + 2·y[e,2]) + Σ_{x=a}(z0+z1) ≤ CAP[a]
#   `CAPDUP` becomes unnecessary here -- the capacity is a constraint, not replicated vertices,
#   so one bond cannot take two units by construction.
#   ⚠️ `scipy` is imported **lazily**, only when this flag is on, so the package keeps declaring
#      just numpy · networkx · rdkit.
#   ⚠️ Above `CAPMILP_MAX` variables it falls back to the matching: measured on 600 holdout
#      structures, **89.75% solved · 9.00% too large · 1.25% solver failure**.
#   ⛔ **Measured and left off (2026-09-08).** It does what it claims -- the ④ metrics rise, and
#   `Triple` rises by more than any other change so far -- but the **deployment output gets
#   worse**:
#     holdout 6,793  T3 `Double` .7417 → **.7490** · `Triple` .9770 → **.9805** · `Single` .9901
#                    → .9902 · T8 `Double` .7461 → .7471
#                    `Sq_L` .8398 → **.8372** · `OS` .8712 → .8708 · violations .0302 unchanged
#                    harmful `Double` errors **305 → 313** (1 fixed, 9 newly broken)
#     train 27,294   `Double` .7327 → **.7414** · `Triple` .9777 → **.9818**
#                    `Sq_L` .8344 → **.8330** · `OS` .8640 unchanged · violations .0307 → .0306
#   🔴 This is the sharpest evidence for Codex's structural point: the stages do not share an
#      objective, so solving ④ **better** hands ⑤ a different starting point and the pipeline
#      ends **worse**. Stage-local exactness is not the lever. Adopting it would need ⑤/⑥ to be
#      re-tuned against it, which is the joint-inference redesign, not this flag.
CAPMILP = os.environ.get("CAPMILP", "0") == "1"
CAPMILP_MAX = int(os.environ.get("CAPMILP_MAX", "1200"))

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

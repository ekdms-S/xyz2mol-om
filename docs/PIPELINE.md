# Pipeline — what is decided in what order, by what formula

**Every decision rule** from one `xyz` coming in to bonds, orders, charges and oxidation states
coming out. This document is the pipeline **as it is**: the rules, the formulas, the thresholds and
the output contracts. The arguments behind them, the measured evidence and the rejected
alternatives are kept in the project workspace, not here.

Notation. `d(X,Y)` distance (Å) · `w(M,X)` xtb GFN2 **Mayer** bond order · `q_frag` fragment charge ·
`deg(X)` number of **internal** neighbors within the ligand (H included · M–L excluded) · `b_int(X)` sum of internal bond orders ·
`b_ML(X)` sum of M–L bond orders · `n_ML(X)` **number** of M–L bonds of X · `n_lp(X)` lone pairs X still has to give ·
`v` number of valence electrons.

## Current performance — shipped defaults

| task | holdout 6,793 | train 27,294 |
|---|---|---|
| T1 internal bond existence | **.9998** | .9998 |
| T3 `Single`/`Double`/`Triple`/`Conj` | **.9901 / .7667 / .9775 / .9617** | .9895 / .7601 / .9792 / .9592 |
| T4 M–L·M–M existence | **.9915** | .9927 |
| T5 haptic | **.9800** | .9807 |
| T6 η^k | **.9865** | .9832 |
| T8 M–L `Single`/`Double`/`Triple` | **.9935 / .7556 / .7254** | .9937 / .7621 / .7735 |
| T10 `Σq_L` · `OS` | **.8648 · .8967** | .8580 · .8909 |
| valence-violating structures | **.0035** | .0037 |
| harmful `Double` errors | 283 bonds · 158 structures (2.33%) | — |
| reported fragment charge ≠ the emitted structure's | 129 (1.90%) | — |

⚠️ The last two rows come from a separate scorer and an earlier run; every other figure is one run
of `260907_deploy_full_score` over both splits.

The **holdout split is 6,793 structures never used in any fit**, opened blind once. `Σq_L` and `OS`
are scored against tmQMg-L ligand charges and the roman numeral in the CSD chemical name, which
cover 23% and 41% of structures; every other task is scored against CSD `bond_type`, which covers
all of them. Feeding the **reference** bond orders to the same charge rule gives `Σq_L` .8528 ·
`OS` .8698 — on the common pool the pipeline is above that line on both, so what is left of the
`OS` gap is Lewis notation rather than order prediction.

**Fitted parameters in the rules: one** — the prior temperature `LPA = 0.8` (§T3 ③). Every other
rule is a structural condition, apart from two tolerances set by measurement: the planarity
tolerance `τ_plane = 0.05 Å` (§T3 ①②) and the Mayer ceiling `SIGCUTW = 0.40` (§T3 post-⑥). The
per-element-pair tables (`d_int`, `d_bond`, `b_ml_t8forms`, `scores4`) are fits and are listed at
the end of this document.

Class codes: `0 Single · 1 Double · 2 Triple · 3 Conj` (delocalized, formal order 1.5).

---

## Terms

Short names used throughout. Task codes `T1`·`T3`… are the benchmark's, stage numbers `①`…`⑥` are
T3's internal order.

| term | meaning |
|---|---|
| `Conj` | a bond carrying part of a **delocalized π system**. Not an integer order — ⑥ resolves it to 1 or 2. Internally it costs `k+1` valence for `k` such bonds on an atom, never 1.5 |
| `CAP(X)` | X's **valence ceiling** — bonds plus lone pairs it can hold (C 4 · N 5 · S 6 …). Not the neutral-atom valence: this is an ionic cut, so a deficient atom is an anion, not an error |
| `b_int` · `b_ML` | X's **internal bond-order sum** · **number of M–L bonds** it makes. Different units on purpose (see ④) |
| `w` | **Mayer bond order** from xtb GFN2 `--sp --wbo`. Used for M–L existence and order, never for internal bonds |
| haptic · `η^k` | an M–L bond to an atom of a π system (`η⁵`-Cp) rather than to a lone pair. `k` = how many atoms of that π fragment bind the same metal |
| 3c2e · dative | a bridging atom's tag: **3c2e** = one electron pair shared over three centres (μ-H, μ-CH₃, μ-CO, B–H–B); **dative** = two genuine 2-centre donations (μ-Cl) |
| `q_L` · `OS(M)` | **ligand charge** — the formal charges of every ligand atom, summed **on the emitted Kekulé integers** · **metal oxidation state**, what is left of the complex charge after the ligands |
| the veto (⑤) | ⑤ may not create a **new pair of adjacent same-sign formal charges** (no `C⁻ C⁻`). Hard rejection, not a penalty |
| trust gate (⑤) | fragment classes whose extended-Hückel charge target is not trusted (parity · composition · nitro motif) |
| harmful `Double` | the deployment error metric: a reference `Double` that the **emitted Kekulé structure** does not call 2, excluding ambiguous Kekulé positions, μ-CO π-acceptors, and charge-transfer resonance |
| Rule A · R2…R7 | the named chemistry rules. A · R2 · R3 · R4 · R5 build the conjugation set (§T3 ①②); R7 patches its side effect on haptic detection (DAG 5′) |

---

## Order (DAG)

🔴 **Read the picture first.** The numbered list under it is a *reference per step*, **not** an
execution order. `T3` runs **twice**, and the M–L **type** decisions (5 · 5″) sit **between** the
two passes — they are what pass 2's budget is made of.

```text
                    0. metal / non-metal split  (METALS · `centers()`)
                                       │
             ┌─────────────────────────┴─────────────────────────┐
             ↓                                                   ↓
   1. [T1] internal bond exists                    4. [T4] M–L · M–M bond **exists**
   2.      rings = SSSR                                    existence only — no type, no order
        ── metal-independent ──                                  │
             │                                                   │ coordinating-atom set
             │                                                   ↓  = `coord`
             └─────────────────────────┬─────────────────────────┘
                                       ↓
                   3. [T3] pass 1     bml = {} · ml_sc = None
                                       │   no metal budget at all
                                       ↓
                    π fragments  ·  provisional internal orders
                                       │
             ┌─────────────────────────┴─────────────────────────┐
             ↓                                                   ↓
   5. [T5] provisional haptic                        5″. [T7] `bridge_tags`
          angle only (θ < 81.02°)                          3c2e / dative
          spends 0 budget                                  reads pass-1 **orders**
             │                                             3c2e spends `BML3C_COST` in total
             └─────────────────────────┬─────────────────────────┘
                                       ↓
                        `bml` budget fixed  (`rules.pipeline.bml_budget`)
                                       ↓
                   3. [T3] pass 2     bml · ml_sc
                                       │   7. [T8] M–L orders are solved **inside** this
                                       ↓      same ④ matching, not in a later step
    final internal orders · final M–L orders · 5. [T5] final haptic (pass-2 π) + 5′. [R7]
                                       ↓
             6. [T6] η^k  →  8. [T10] q_L · OS(M)  →  ⑥ output converter
```

Why pass 1 exists: a haptic M–L bond spends no budget, but whether a bond is haptic can only be
told once the π fragments are known — so T3 is solved **once with an empty budget** to get them.
5″ rides along on the same pass because its rule needs internal orders not yet contaminated by the
metal budget.

```
0.  metal / non-metal split                             METALS list · `config.centers`

━━ metal-independent ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1.  [T1] internal bond exists ⟺ d(X,Y) < d_int(X,Y)     56 element pairs + fallback (data/d_int.csv)
2.       rings = `nx.cycle_basis`                       no parameters
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

3.  [T3] 4-class assignment ①→②→③→④→⑤→⑥                see §T3 below
         🔴 **runs twice** (`pipeline.predict_T3_T5`)
              pass 1   bml = {} · ml_sc = None   → provisional orders · π fragments
              pass 2   bml = the budget · ml_sc  → the output orders, M–L included

4.  [T4] M–X bond **exists**  ⟺   d(M,X) < d_bond(M,X)  AND  w(M,X) > w_veto(M,X)
         existence only — no type and no order.       element pairs 315 (M–L) · 37 (M–M)
         X is any atom, metals included — (M,M) pairs are decided here too.
         the coordinating-atom set it produces is passed to **pass 1** as `coord`
           (it waives the conjugation under-valence penalty)
         agostic excluded: `C–H···M` is not a bond
           ⟺ that H has exactly one metal-like neighbour AND an internal neighbour that is not
             metal-like        (μ-H and `B–H···M` are kept — those are real 3c2e)
         saturated atoms excluded: no M–X bond to an atom its own bonds already fill up
           ⟺ el(X) ∉ {H, B, Al}  AND  no internal neighbour of X is B or Al  AND  deg_int(X) ≥ CAP(X)
             `deg_int` counts **neighbours, not bond orders** — T4 runs before ③, so no order exists yet
             the B·Al exception covers cage carbons (a dicarbollide C has deg 5 > CAP 4)

5.  [T5] that bond is haptic
           ⟺  ∠(M–X–Y) < θ = 81.02°   AND  X belongs to a π fragment
           **or** the bond-level test 5* fires for a π bond X is an end of
           Y = the neighbour of X within its fragment whose **bond midpoint is closest to M**
           if false, σ-dative (the bond itself was already settled in 4, so it stays)
           π fragment = connected component of {Conj ∪ Double ∪ Triple} bonds
           ⇒ **X belongs to a π fragment ⟺ X touches a `Double`, `Triple`, or `Conj` bond.**
             🔴 **No fragment-size condition.** A lone isolated double bond is a π fragment too.

5†. **η¹ is a σ bond, so it is not haptic** (`rules.pipeline.drop_eta1` · 0 fitted parameters)

           drop(M, X) ⟺ X's **ligand fragment** gives M exactly one haptic atom

         🔴 Applied to the **pass-1** haptic set too, so the atom that stops being haptic also
            starts paying its ④ valence unit.
         ⚠️ Counted **per ligand fragment**, not per connected run of haptic atoms.

5*. **η² is a property of the bond, not of each atom** (`rules.pipeline._eta2_pair` · 0 fitted parameters)

           both(M, X–Y) ⟺ X–Y is an internal bond with π character (`Double`/`Triple`/`Conj`)
                       AND X and Y both have a T4 bond to the **same** M
                       AND neither X nor Y is H
                       AND ∠(M–X–Y) < θ  **or**  ∠(M–Y–X) < θ

         🔴 applied in **pass 1 as well** — what it buys is the ④ budget: a haptic M–L bond costs
            0 valence, which is what lets pass 2 raise the π bond at all.

5′. [R7] **Return an R2 donor inside a haptic ring to the π candidates**   (on by default)
         Turn it off with `R7RING=0`. 0 fitted parameters (`R7MIN` is an integer lattice).

           add(M,X) ⟺ X is an R2 donor (O·S·Se: deg ≥ 2 · N·P: deg ≥ 3)
                    AND X belongs to a ring r with |r| = 5  (r from `cycle_basis`)
                    AND at least R7MIN = 2 of the **other atoms** of r passed 5 for the **same metal M**
                    AND (M,X) is a T4 bond  (d < d_bond AND w > w_veto)
                    AND ∠(M–X–Y) < θ = 81.02°
           Y candidates: X is not in a π fragment, so 5's "same-fragment neighbour" cannot be used ⇒
                    pick among the **neighbours that do belong to a π fragment**. η^k is added to that Y's fragment.

         ⇒ **T3 bond orders are not changed.** Only 5's "X belongs to a π fragment" condition is
           waived. It patches a stage **after** T3, so no cycle appears in the DAG above.

5″. [T7] bridge tag — the **type** of an existing M–L bond, `rules.pipeline.bridge_tags`

           n_center(X) = n_ML(X) + (internal neighbours of X whose element is B·Al)
           b_use(X)    = b_int(X) + n_ML(X)        b_int from the **pass-1** Kekule orders
           bridge(X) ⟺ n_center >= 2
           3c2e(X)   ⟺ bridge AND el ∈ {H,C,Si,B} AND b_use > VALENCE_3C[el] (H 1 · C·Si 4 · B 3)
           dative(X) ⟺ bridge AND the above is false

           μ-H       M–H–M         b_use 0+2 = 2 > 1  →  3c2e
           μ-CH₃     M–CH₃–M       b_use 3+2 = 5 > 4  →  3c2e
           μ-CO      M–CO–M        b_use 3+2 = 5 > 4  →  3c2e    (C≡O · a C=O would give 4)
           B–H···M   borohydride   n_center = M 1 + neighbour B 1 · b_use 2 > 1  →  3c2e
           B–H–B     diborane      n_center = neighbour B 2 (**no metal**) · b_use 2 > 1 → 3c2e
           μ-CR₂     bridging carbene   b_use 2+2 = 4  →  dative (genuinely two 2c2e)
           μ-Cl      M–Cl–M        Cl ∉ VALENCE_3C    →  dative (3c4e)
           terminal  M–L           n_center 1         →  no tag

         🔴 **This tag is not a label on the output — it changes the answer.** A `3c2e` atom spends
            `BML3C_COST` (1.0) of the ④ budget *in total* instead of one unit per M–L bond, which
            raises the internal orders ④ assigns it; ⑥ and the fragment charge use the same budget.
         `cls` (the pass-1 classes) is a required argument — pass 2 needs this tag to build its budget.
         output      `ml_bonds[(m,x)]["bridge"]` = None|"3c2e"|"dative" · `["type"]` is
                     haptic > bridge > sigma · all-internal legs in `fragment["bonds_3c2e"]`
         ⚠️ **No bond disappears** — both M–L bonds stay and 7 assigns their orders too.

6.  [T6] η^k     k = number of atoms in that π fragment that passed 5     no parameters
         counted **per ligand**, which is what the reference (tmQMg-L `n_haptic_bound`) counts.
         Ferrocene's two rings are two ligands, so it comes out as two η⁵.
         an atom entering via 5′ is counted in that Y's fragment

         ⚠️ **A bridged (ansa) metallocene comes out as one η¹⁰, not η⁵:η⁵** — the two rings are
            joined, so they are one ligand. This is the counting rule, not a detection error:
            `ml_bonds` says which ten atoms. Group them by ring to recover the per-ring numbers.

7.  [T8] M–L order (non-haptic bonds only)
           Single ⟺ w < t₁(M,X)    Double ⟺ t₁ ≤ w < t₂    Triple ⟺ w ≥ t₂
           421 element pairs × 2 parameters (data/b_ml_t8forms.csv) · **distance is not used**
           t₂ = ∞ means that pair has no Triple — a fit result, not a rule

8.  [T10] ligand charge · oxidation state                no parameters → §Charge below
```

---

## §T3 — internal bond orders within a ligand

This is the core. Every other task is one threshold; T3 answers **six questions in order**, each
constraining the next.

| | question | how |
|---|---|---|
| **①②** | which bonds share a **delocalized π system**? | chemistry, 0 fitted parameters |
| **③** | what order does the **geometry** want? | per-element-pair distance likelihood, prior damped by `LPA = 0.8` |
| **④** | what can the **valence budget** afford? | hard constraint, maximum-weight matching |
| **⑤** | does the fragment's **electron count** agree? | extended-Hückel fragment charge as a target |
| **⑥** | emit **integers** | Kekulé matching |
| **post-⑥** | is there a **valid assignment with less charge** on the same bonds? | `QSHIFT` · `QGEM` · `SIGCUT`, accepted only if `Σ|q|` falls |

Stages ④–⑥ can each overrule the one before it, so a bond that ③ wants as `Double` may still come
out `Single`. That is the usual reason for a wrong `Double`.

### ①② The conjugation set — which bonds are delocalized

A `Conj` bond is one carrying part of a delocalized π system; its order is not an integer until ⑥
resolves it. One rule puts bonds in, four take them out. **All five are chemistry — 0 fitted
parameters** apart from the planarity tolerance `τ_plane`.

```
IN   Rule A   a ring is delocalized if it is planar and its atoms are unsaturated
              pin(e) ⟺ e lies in a ring r (`nx.cycle_basis`) · |r| ≥ 5
                       · out-of-plane rms(r) ≤ τ_plane = 0.05 Å
                       · both ends unsaturated (deg(X) < CAP(X))
              a pinned bond is **forced** to `Conj`, overriding ③ — but R2·R3·R4 are
              subtracted from the pinned set first, so an exclusion always wins

OUT  R2       a **lone-pair donor** heteroatom is part of π but its own bonds are order 1
              forbid(X) ⟺ X ∈ {O,S,Se: deg ≥ 2} ∪ {N,P: deg ≥ 3}  ⇒ no Conj on any bond of X
              exception ⟺ X = N and its fragment's EHT charge > 0   (pyridinium N⁺)

     R3       a 5-ring holding an R2 donor is Kekulé **as a whole**
              forbid(ring r) ⟺ |r| = 5 AND r holds an R2 donor

     R4       an antiaromatic 4n carbocycle is not delocalized
              forbid(ring r) ⟺ |r| ∈ {4, 8} AND all carbon AND out-of-plane rms > τ_plane

     R5       a **lone** Conj bond is not delocalized — delocalization needs a neighbour
              forbid(e) ⟺ neither end of e touches another Conj bond
```

⚠️ **Known cost of R3:** five-rings **mixing N with O or S** (isoxazole · oxazoline · thiazoline)
are genuinely aromatic azoles, and R3 flattens them to Kekulé.

### ③ Distance likelihood — what the geometry wants

```
score(e, c) = − |d(e) − med[k, c]| / scl[k, c]  +  LPA · lp[k, cell(e), c]

  k       = element pair (sorted)             med = per-class distance median
  scl     = 1.4826 × MAD                      lp  = ln P(c | condition)
  cell(e) = (min(deg(X), 4), min(deg(Y), 4))  endpoint internal-degree pair, ordered with k
  LPA     = 0.8                               prior temperature — the one fitted parameter

lp[k, cell, c] = ln P(c | k, cell)   when the cell holds ≥ LPCOND_NMIN = 300 samples
               = ln P(c | k)         below that
exception: c = Conj always uses the global value
```

🔴 **The prior is conditioned on the degree cell, not globally per element pair** — the same
element pair is a different bond at different degrees (`C–O` is .111 `Double` globally but .322 in
the carbonyl cell and .983 `Triple` in the CO-ligand cell), and the global form flips carbonyls to
`Single` against the distance.

⚠️ The 300-sample cut-off is a **statistical floor, not a chemical claim** — a bond whose cell sits
just under it is scored by a different formula than its neighbour.

Values in `data/scores4.json` — 18 element pairs · 57 conditioned cells, fitted on 26,075
train structures.

### ④ Valence budget — what can be afforded

```
maximize    Σ_e  score(e, c(e))
subject to  b_int(X) + b_ML(X)  ≤  CAP(X)         for every non-metal X

  b_int  k conjugated bonds cost **k + 1** — k σ bonds plus the one π that the Kekulé matching
         will place on this atom. **Exception:** if the fragment has a maximum matching that
         leaves X unmatched, X gets no π and costs **k**; ⑥ is then held to that promise.
  b_ML   **count**, not order sum — under the ionic cut an M–L bond is one donated lone pair.
         haptic costs 0. An atom in a **3c2e** bridge costs 1.0 *in total* however many M–L bonds
         it has: one electron pair over three centres is one bond of valence.
  CAP    H 1 · B 4 · C 4 · N 5 · O 4 · F 4 · Si 6 · P 6 · S 6 · Cl 7 · Br/Se/As/Te/I 6
```

🔴 **There is no lower bound.** This is an ionic cut, so an anionic ligand is normal (`Cl⁻`·`RO⁻`·
`Cp⁻`). A deficient valence is not an error — it is a **negative charge**.

How it is solved: take the slack `r(X) = ⌊CAP(X) − use(X)⌋` as a capacity, replicate each atom that
many times, and run a **maximum-weight matching** (Blossom). `Triple` is confirmed first, greedily,
for bonds whose likelihood argmax is `Triple` and whose ends both have slack ≥ 2. M–L orders are
solved inside the same matching.

⚠️ **Not the exact maximum** — the greedy `Triple` pass can spend slack the matching would have
used better (113 of 12,245 holdout calls). The exact MILP (`CAPMILP`) makes the deployment output
worse and is off by default.
⚠️ Only bonds with `score(Double) > score(Single)` are offered to the matching. A bond the
likelihood scores as `Single` is never a candidate for promotion.

### ⑤ Fragment electron count — does the charge agree

Extended Hückel (RDKit `rdEHTTools`) gives a ligand fragment's charge directly, and that fixes the
fragment's **total** bond order to one integer:

```
q_frag(EHT) = Σ v_i − 2 × #{Hückel orbitals with E < −10 eV}   (+ HOMO/LUMO correction)
identity     q_frag = C0(composition) + 2B,   C0 = Σ v_i − 8n   (2 for H)
         ⇒   B* = (q_EHT − C0) / 2

assignment ⟺ move bonds ±1 to reach B = B*, respecting ④'s ceiling, ≤ 12 rounds
             short → +1 from the largest likelihood gain · over → −1 from the smallest loss
```

**The veto.** A move that creates a **new pair of adjacent same-sign formal charges** is dropped
from the candidate list; if nothing else is available ⑤ gives up the target and ④'s assignment
stands. It is a hard rejection — a soft penalty only reorders the candidate list.

**The trust gate.** ⑤ only chases a target it has reason to trust:

| condition | | target error rate |
|---|---|---|
| parity | `q_EHT − q_current` is odd — unreachable by ±1 moves | — |
| composition | fragment is `NO` · `SS` · `CCHH` | 99.8% · 94.9% · 66.9% |
| motif | an N carrying **exactly two** terminal O (nitro / nitrite) | 84.4% (`R–NO₂`) · 100% (free `NO₂⁻`) |

Two details are load-bearing: **exactly two** terminal O, not at least two (three is nitrate, whose
target is right); and **composition**, not size (a size cut would also disable `CO`).

⚠️ `CCHH` (η²-acetylene) is the weakest member at 66.9% — the item most likely to be fitting the
CSD sample rather than chemistry.

### ⑥ Output converter — integers

Turns the 4 classes into integer S/D/T with a maximum-cardinality matching over the `Conj` bonds,
the same one the charge calculation uses, so the two cannot disagree. The matching is **weighted**
by `score(Double) − score(Single)`, so the Kekulé structure that comes out is the one the bond
lengths want; an atom ④ granted headroom on condition of staying unmatched gets `−10⁶` on every
incident edge, which cannot shrink the matching but keeps that promise.

```
kekulize(G, el, cls, b_ML) → (orders, frag_q)
  orders  {(i,j): 1.0 | 2.0 | 3.0}
  frag_q  charge the skeleton cannot express (tropylium +1 · even-ring dianion −2)
          measured 202 / 26,074 fragments = 0.8%
```

**What ⑥ reports about itself — `pi_suppressed`.** When ④ could not afford a π bond, ⑥ writes it
`Single` and the charge step puts a lone pair on each end, which leaves the fragment charge 2 too
negative and the metal's oxidation state 2 too high. Every fragment carries the bonds where that
happened:

```
suspect(i,j) ⟺ orders[(i,j)] == 1  AND  q(i) < 0  AND  q(j) < 0
                AND  score(Double) − score(Single) > 0        ← raw likelihood, before ④'s −10⁶
```

Element pairs with no `Double` class fitted (`As–C`, `B–B`) are excluded. ⚠️ It is a **flag, not a
correction**: the orders and charges are returned unchanged. Rate and what it catches: README
`## ⚠️ Limits`.

### Post-⑥ repairs — a π in the wrong place, and a σ M–L that pays for it

④ treats the valence ceiling as hard and formal charge as soft, so "raise the bond and both ends go
neutral" is never in its feasible set. Three rules undo that on the emitted integers. All three
fire only where a **valid** assignment with a smaller `Σ|q|` exists on the same bond set, and the
result is kept only if `Σ|q|` actually falls.

```
QSHIFT  two like- or opposite-signed charges at the ends of an **alternating path** — flip the path
        odd path length  → both ends move the same way   `C⁻–C=C–O⁻ → C=C–C=O`
        even path length → they move oppositely          `O⁺≡C–O⁻ → O=C=O`  (CO₂)
        k = 1 splits on sign: an **opposite**-sign pair is left alone (CO `[C⁻]≡[O⁺]`, amine
        oxide `R₃N⁺–O⁻`, phosphorus ylide `R₃P⁺–C⁻`); a **like**-sign pair is raised, which
        neutralizes both ends at once (`[C⁻](H)(H)[O⁻]` → formaldehyde)

QGEM    two like-signed anions sharing **one common neighbour** — raise **both** bonds together
        an alternating ±1 shift cannot reach this shape. `[O⁻]–C–[O⁻]` → `O=C=O` (metal-bound CO₂),
        nitromethane `C–N([O⁻])[O⁻]` → `C–N⁺(=O)O⁻` with no metal present
        moves only when the length fits both raised orders

SIGCUT  the pair could cancel, and **only the σ M–L valence budget** blocks it — drop that σ M–L,
        then re-solve ③④⑤⑥ from scratch
        ⚠️ the orders cannot be patched in place: without the M–L the haptic set, η, the budget
           and the fragment split all change, so the whole solve is repeated
```

Exclusions, all structural:

- **`QSHIFT`·`SIGCUT`: a pair whose two anionic sites both coordinate the same metal is left
  alone** — that is a genuine dianionic ligand (dithiolene, catecholate, amidinate; for `SIGCUT`,
  benzyne or a metallacyclopropene), not a misplaced π.
- **peroxide `[O⁻]–[O⁻]` is never raised** — it is a real species and a common ligand. `[C⁻]–[C⁻]`
  (→ ethene) is raised, so only O–O is blocked.

What `SIGCUT` may cut:

```
SIGCUTW = 0.40    Mayer ceiling. A σ M–L at or above it is a real covalent bond and is kept;
                  with no Mayer available nothing is cut
B · Al centres    never cut. `MLIKE_EXTRA` makes them centres, but their C bonds are the
                  cage/skeleton covalent bonds of a carborane or a boryl, not donations
SIGCUTFIT         ceiling **exception**: a Mayer above `SIGCUTW` is still cut when the raised
                  order fits the bond **length** better — `|d − med[raised]| < |d − med[cur]|`
                  and the current order is outside its own distribution (`|d − med[cur]| > scl[cur]`),
                  both read from `scores4`. A comparison of two fitted medians, so no new constant
```

🔴 **(a″) is applied twice** — once right after ⑥ and once after the repairs, because `QGEM`
raising both `N–O` of a nitro group re-creates the five-bonded `N(=O)=O` that (a″) removes.

---

## §Charge — `q_L` and `OS(M)` (0 fitted parameters)

```
(a) per-atom formal charge
        lp(X) = max(0, 4 − b)                     lone pair count (max(0, 1 − b) for H)
        q(X)  = v − b − 2·lp(X)

    ⇒ b ≤ 4 :  q = v + b − 8      (octet)         v + b − 2 for H
      b > 4 :  q = v − b          (hypervalent)

    the two branches: above `b = 4` the octet form would need a negative lone-pair count and
    reads nitro N (b 5) as +2, sulfone S (b 6) as +4, perchlorate Cl (b 7) as +6; the
    hypervalent branch gives 0 for all three and no `b ≤ 4` site changes.

(a″) **nitrogen never carries five bonds.** After ⑥, one `N=O` to a **terminal** O of a
     five-bonded N is demoted to `N–O` and the charge follows from (a): N `+1`, that O `−1`.
     The fragment total is unchanged. Period-3 atoms keep the hypervalent form, which is
     legitimate for them (sulfone S, perchlorate Cl).

(a′) the remaining sites where the octet breaks — an (element, deg, b, neighbor element) table
        heteroatom-stabilized carbene `("C", 2, 2, N or O among neighbors)`  → 0     (octet formula −2)
        sulfoxide      `("S", 3, 4, O among neighbors)`                      → 0     (octet formula +2)
        nitrite        `("N", 2, 4, two O neighbors)`                        → −1    (octet formula +1)
    ⚠️ The neighbor-element condition is essential — keyed on `(element, deg, b)` alone, azide
       and isocyanide get caught too and the ligand charge is off by −2.

(b) conjugated fragment charge
        monocyclic all-carbon `CmHm`  →  Hückel:  z = m − (4n+2) minimizing |m − h| (larger h on a tie)
        otherwise                     →  sum of (a) over the ⑥ Kekulé integers

(c) 3c2e — the decision itself is **T7 at stage 5″** of the DAG. A bridging atom on a metal gets
    the same `q_atom(element, b_int)` as any other; the tag reaches the charge only indirectly,
    through the internal orders the ④·⑥ budget allows:  budget → `b_int` → `q_atom` → `q_L` → `OS(M)`.
    So a bridging CO comes out with the same ligand SMILES as a terminal one, `[O+]#[C-:1]`.

    A leg that lies **inside the ligand** (the `B–H` of `B–H–B` or of a κ²-BH₄) is different: it is
    drawn as an ordinary bond but holds no pair of its own, so `q_atom` takes `b_3c` — the order sum
    of those legs, reported as `fragment["bonds_3c2e"]` — and subtracts it from `b` before applying
    (a). B₂H₆ is then `B(+1)` · bridging `H(−1)` · terminal `H(0)`.
    Only legs that carry no pair are listed: a `C–B` on a bridging carbon is an ordinary 2c2e bond
    and stays in `bonds_kekule` alone.

(d) q_L = sum of the formal charges of **all atoms** of the ligand fragment   ← not only the coordinating atoms
          🔴 counted on the **emitted Kekulé integers** (⑥), plus the residual ⑥ returns for the
          charge a skeleton cannot express — the same count the per-atom charges in the SMILES
          use, so the reported charge and the emitted structure cannot disagree.
    OS(M) = (q_total − Σ_L q_L) / n_M              ← distributed evenly over the metals
```

⚠️ **M–L is not counted in `b`** — it is an ionic cut, so the coordinating atom takes both M–L electron pairs entirely.

---

## What is left

- **`Double` is the only weak class.** The rest are .96–.99. The remaining errors are heteroatom
  double bonds (`C=N` · `C=S` · `C=O` · `N=N` · `N=O` · `C=Se`): `Double` is only 2.2% of internal
  bonds, so in a cell such as `C(3)–N(3)` the prior outweighs the distance and ④ never sees the
  bond as a candidate (§T3 ③).

### 🔴 Valence violation rate — the axis that is not in the performance table

Even with a high F1, the **output can be chemically impossible**, so the valence ceiling violations
are reported alongside (the `valence-violating structures` row at the top of this file):

```
violation(X) ⟺ b_int_kek(X) + b_ML(X) > CAP(X)          X is a non-metal
  b_int_kek : k conjugated bonds count as **k+1** (not a 1.5 conversion) — or **k** where ④ granted
              headroom on the promise that ⑥ leaves the atom unmatched (see ④)
  b_ML      : sum of M–L bond orders (haptic excluded)
  ⚠️ 3c2e-tagged atoms and `B` are dropped from the tally — outside the two-center formalism
```

⚠️ **The baseline is not 0** — feeding the CSD reference labels as they are, 0.4–0.7% of structures
violate (hypervalency · ionic/covalent boundary · CSD notation conventions), so our rate has to be
read against that and not against zero.

🔴 A known residual for **`Al`**: a Mayer cache built for true transition metals has no `Al–X`
entry, and a missing entry is read as "veto passed", so each one becomes an M–L bond that eats the
`CAP` budget of the neighbouring `C` and `H`. Supplying `wbo` for `Al` removes it.

### Trivial baselines (always read the performance next to these)

| Task | Trivial prediction | Trivial performance |
|---|---|---|
| T1 | all bonded | 0.7306 |
| T3 | all `Single` | S .9097 · D 0 · T 0 |
| T4 | all bonded | 0.5276 |
| T5 | all haptic | 0.6766 |
| T6 | all `k=0` | 0.8704 |
| T8 | all `Single` | accuracy .9637 |

---

## Fitted parameter summary

| File | What | Count |
|---|---|---|
| `data/d_int.csv` | T1 per-element-pair distance threshold | 56 + `H–H` default + 1 fallback |
| `data/d_bond.csv` | T4 `d_bond` · `w_veto` | 315 (M–L) + 37 (M–M) |
| `data/b_ml_t8forms.csv` | T8 monotone thresholds `t₁ ≤ t₂` | 421 pairs × 2 |
| `data/b_ml_mayer.csv` | T8 likelihood form (fallback) | 421 pairs |
| `data/scores4.json` | T3 distance likelihood `med`·`scl`·`lp`·`lp_cell` | 18 element pairs · 57 cells |

⚠️ **The `M = B` rows of `d_bond`, `b_ml_dist`, `b_ml_mayer` and `b_ml_t8forms` are never read** —
`B` is not in `METALS`, so `d_int` and `scores4` carry those bonds instead.

These rules carry **no fitted parameter** — each is a structural condition: Rule A · R2 · R3 · R4 ·
R5 · R7 · the ⑤ EHT trust gate (composition list + nitro motif) · the ⑤ adjacent-same-sign veto ·
the ④ unmatched-atom exception · the bond-level η² rule (5*) · the hypervalent charge formula ·
the T4 agostic and saturated-atom exclusions.

Global constants:

| constant | value | what it is |
|---|---|---|
| `LPA` | 0.8 | prior temperature in the ③ likelihood — **the one fitted parameter in the rules**, chosen on a one-decimal grid |
| `θ_haptic` | 81.02° | M–X–Y angle below which an M–L bond is side-on — **fitted**; the two decimals are not meaningful |
| `τ_plane` | 0.05 Å | ring planarity tolerance (Rule A · R4) |
| EHT cutoff | −10 eV | occupied-orbital cut in the fragment charge |
| `LPCOND_NMIN` | 300 | samples a degree cell needs before its own prior is used — a **statistical floor**, not chemistry |
| 3c2e cost | 1.0 | valence a bridging atom spends in total for its M–L bonds — one shared pair is one bond |

⚠️ Two choices in this pipeline are **measured rather than derived**, and are flagged where they
are used: the `CCHH` entry of the ⑤ composition list (66.9% target error rate) and the value of
`LPA`.

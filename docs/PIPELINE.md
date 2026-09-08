# Pipeline — what is decided in what order, by what formula


## ★ Current performance — re-measured 2026-09-08 (shipped defaults)

Three rules on top of the base pipeline, each a single idea with **no fitted parameter** —
`ADJQVETO=1` (⑤ may not create adjacent same-sign formal charges) · `CAPINESS=1` (④ charges
an atom `k`, not `k+1`, when ⑥ can leave it unmatched) · the ⑤ **EHT target trust gate**
(`EHTMINFRAG` · `EHTSKIP=NO,SS,CCHH` · `EHTNITRO=1`). Two defect fixes are unconditional
(one capacity unit per bond in ④; cluster detection on Kekulé integers), and the ⑥ matching
is always weighted by the distance likelihood.
`ADJQW=0 TAUD=0 ETAEXO=0 CAPMILP=0` (built, measured, rejected — see the `config` comments).

| task | holdout 6,793 | train 27,294 |
|---|---|---|
| T1 internal bond existence | .9998 | .9998 |
| T3 `Single`/`Double`/`Triple`/`Conj` | **.9904 / .7682 / .9769 / .9611** | .9895 / .7586 / .9776 / .9587 |
| T4 M–L·M–M existence | .9905 | .9918 |
| T5 haptic | .9777 | .9793 |
| T6 η^k | .9863 | .9819 |
| T8 M–L `Single`/`Double`/`Triple` | **.9932 / .7456 / .7317** | .9934 / .7512 / .7728 |
| T10 `Σq_L` · `OS` | **.8519 · .8831** | .8495 · .8770 |
| valence-violating structures | .0303 | .0306 |
| harmful `Double` errors | **302 bonds · 169 structures (2.49%)** | — |
| reported ligand charge ≠ the emitted structure's | **214 (3.15%)** | — |

Feeding the **reference** bond orders to the same charge rule gives `Σq_L` .8528 · `OS` .8698.
On the **common pool** (1,155 / 2,635 structures — kekulizing the reference fails on a few) the
comparison now reads `Σq_L` **.8563 vs .8528** and `OS` **.8774 vs .8725**: since 2026-09-08 the
pipeline is **above** that line on both. It was never an upper bound — it measures how much of the
remaining gap is our Lewis notation against tmQMg-L's rather than our bond orders — and what is
left of the gap is notation, not order prediction.

🔴 **The numbers further down are from the date they were taken** — they record how a rule was
adopted, not current performance. Quote this table.
**Every decision rule** from one `xyz` coming in to bonds, orders, charges, and oxidation states coming out.
This document is the pipeline **as it is**. Trial and error, rejected alternatives and the tuning history are
kept in the project workspace (`docs/analysis/` · `docs/settings/`), not here.

Notation. `d(X,Y)` distance (Å) · `w(M,X)` xtb GFN2 **Mayer** bond order · `q_frag` fragment charge ·
`deg(X)` number of **internal** neighbors within the ligand (H included · M–L excluded) · `b_int(X)` sum of internal bond orders ·
`b_ML(X)` sum of M–L bond orders · `n_ML(X)` **number** of M–L bonds of X · `n_lp(X)` lone pairs X still has to give ·
`v` number of valence electrons.

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
| 3c2e · dative | a bridging atom's tag: **3c2e** = one electron pair shared over three centres (μ-H, μ-CH₃, μ-CO); **dative** = two genuine 2-centre donations (μ-Cl) |
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
                        `bml` budget fixed  (`pipeline.bml_budget`)
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
5″ rides along on the same pass because the bond-order form of its rule needs internal orders
that are not yet contaminated by the metal budget (see 5″).

```
0.  metal / non-metal split                             METALS list

━━ metal-independent ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1.  [T1] internal bond exists ⟺ d(X,Y) < d_int(X,Y)     45 element pairs (data/d_int.csv)
2.       rings = `nx.cycle_basis`                       no parameters
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

3.  [T3] 4-class assignment ①→②→③→④→⑤→⑥                see §T3 below
         🔴 **runs twice** (`pipeline.predict_T3_T5`) — see the picture above.
              pass 1   bml = {} · ml_sc = None   → provisional orders · π fragments
              pass 2   bml = the budget · ml_sc  → the output orders, M–L included

4.  [T4] M–X bond **exists**  ⟺   d(M,X) < d_bond(M,X)  AND  w(M,X) > w_veto(M,X)
         existence only — no type and no order. The coordinating-atom set it produces is
         passed to **pass 1** as `coord` (it waives the conjugation under-valence penalty).
         X is any atom, metals included — (M,M) pairs are decided here too.  element pairs 314 (M–L) · 37 (M–M)
         agostic excluded: `C–H···M` is not counted as a bond
           ⟺ that H has exactly one metal-like neighbor and has an internal neighbor that is not metal-like
           (μ-H and `B–H···M` are kept — those are real 3c2e)

5.  [T5] that bond is haptic
           ⟺  ∠(M–X–Y) < θ = 81.02°   AND  X belongs to a π fragment
           Y = the neighbor of X within its fragment whose **bond midpoint is closest to M**
           if false, σ-dative (the bond itself was already settled in 4, so it stays)
           π fragment = connected component of {Conj ∪ Double ∪ Triple} bonds
           ⇒ **X belongs to a π fragment ⟺ X touches a `Double`, `Triple`, or `Conj` bond.**
             🔴 **There is no fragment-size condition.** A lone isolated double bond is a π fragment too.

5′. [R7] **Return an R2 donor inside a haptic ring to the π candidates**   (on by default)
         Turn it off with `R7RING=0`. 0 fitted parameters (`R7MIN` is an integer lattice).

           add(M,X) ⟺ X is an R2 donor (O·S·Se: deg ≥ 2 · N·P: deg ≥ 3)
                    AND X belongs to a ring r with |r| = 5  (r from `cycle_basis`)
                    AND at least R7MIN = 2 of the **other atoms** of r passed 5 for the **same metal M**
                    AND (M,X) is a T4 bond  (d < d_bond AND w > w_veto)
                    AND ∠(M–X–Y) < θ = 81.02°

         ⇒ **T3 bond orders are not changed.** Only the "X belongs to a π fragment" condition of 5 is waived.
           It patches a stage **after** T3 only, so no cycle appears in the DAG above.
         why: when R2 · R3 make a 5-membered ring Kekulé, there are at most two double bonds, so
             **one atom drops out of the π candidates** (η⁵ → η⁴). R2 is an **element rule**, so the same
             failure occurs not only for pyrrole-type N but for **furan O · thiophene S · selenophene Se · phosphole P**.
         Y candidates: X is by definition not in a π fragment, so the "same-fragment neighbor" of 5 cannot be used ⇒
             pick among the **neighbors that do belong to a π fragment**. η^k is added to that Y's fragment.

5″. [T7] bridge tag — the **type** of an existing M–L bond, `pipeline.bridge_tags`

           n_center(X) = n_ML(X) + (internal neighbours of X whose element is B·Al)
           b_use(X)    = b_int(X) + n_ML(X)        b_int from the **pass-1** Kekule orders
           bridge(X) ⟺ n_center >= 2
           3c2e(X)   ⟺ bridge AND el ∈ {H,C,Si,B} AND b_use > VALENCE_3C[el] (H 1 · C·Si 4 · B 3)
           dative(X) ⟺ bridge AND the above is false

         The two halves of `b_use` carry different units by design: an internal bond spends X's
         budget by its **order**, an M–L bond by its **count** — under the ionic cut an M–L bond
         has no order, it is one donated lone pair. `VALENCE_3C[el]` is accordingly not a
         neutral-atom valence but the closed-shell budget (bonds + lone pairs), so with
         `VALENCE_3C[el] − b_int(X) = n_lp(X)` the rule reads **`n_ML > n_lp`** — X donates to
         more centers than it has lone pairs for, and one pair has to be shared over three
         centers. The element list is the shorthand for "`n_lp ≥ 2` whenever `n_center = 2`";
         μ₃ and above are out of scope.

           μ-H       M–H–M         b_use 0+2 = 2 > 1  →  3c2e
           μ-CH₃     M–CH₃–M       b_use 3+2 = 5 > 4  →  3c2e
           μ-CO      M–CO–M        b_use 3+2 = 5 > 4  →  3c2e    (C≡O · a C=O would give 4)
           B–H···M   borohydride   n_center = M 1 + neighbour B 1 · b_use 2 > 1  →  3c2e
           μ-CR₂     bridging carbene   b_use 2+2 = 4  →  dative (genuinely two 2c2e)
           μ-Cl      M–Cl–M        Cl ∉ VALENCE_3C    →  dative (3c4e)
           terminal  M–L           n_center 1         →  no tag

         🔴 **This tag is not a label on the output — it changes the answer.** An atom tagged
         `3c2e` spends `BML3C_COST` (1.0) of the ④ valence budget *in total* instead of one unit
         per M–L bond, which raises its headroom and therefore the **internal bond orders** ④
         assigns to it; ⑥ and the fragment charge use the same budget. So μ-CO comes out as
         `C≡O` with one 3c2e bond and a neutral ligand (`[O+]#[C-]`, the same fragment as a
         terminal CO), not as a ketonic `C=O` with two dative bonds.

         `cls` (the pass-1 classes) is a required argument: pass 2 needs this tag to build its
         budget, so the orders have to come from the pass that has no metal budget.
         output      `ml_bonds[(m,x)]["bridge"]` = None|"3c2e"|"dative" · `["type"]` is
                     haptic > bridge > sigma
         ⚠️ **No bond disappears** — both M–L bonds stay and 7 assigns their orders too.

6.  [T6] η^k     k = number of atoms in that π fragment that passed 5     no parameters
         counted per fragment (ferrocene is two η⁵, not one η¹⁰)
         an atom entering via 5′ is counted in that Y's fragment

7.  [T8] M–L order (non-haptic bonds only)
           Single ⟺ w < t₁(M,X)    Double ⟺ t₁ ≤ w < t₂    Triple ⟺ w ≥ t₂
           420 element pairs × 2 parameters (data/b_ml_t8forms.csv) · **distance is not used**
           t₂ = ∞ means that pair has no Triple — a fit result, not a rule

8.  [T10] ligand charge · oxidation state                no parameters → §Charge below
```

---

## §T3 — internal bond orders within a ligand

This is the core. Every other task is one threshold; T3 answers **five questions in order**, each
constraining the next.

| | question | how |
|---|---|---|
| **①②** | which bonds share a **delocalized π system**? | chemistry, 0 fitted parameters |
| **③** | what order does the **geometry** want? | per-element-pair distance likelihood |
| **④** | what can the **valence budget** afford? | hard constraint, maximum-weight matching |
| **⑤** | does the fragment's **electron count** agree? | extended-Hückel fragment charge as a target |
| **⑥** | emit **integers** | Kekulé matching |

Stages ④–⑥ can each overrule the one before it, so a bond that ③ wants as `Double` may still come
out `Single`. That is the usual reason for a wrong `Double` — see `failures/README.md`.

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
              furan O and pyrrole N donate their **own lone pair**, so they contribute 2 electrons
              and keep single bonds. Pyridine-type N (deg 2) donates one p electron and is 1.5.
              Carbon is always a p-electron donor, so it is never covered.

     R3       a 5-ring holding an R2 donor is Kekulé **as a whole**
              forbid(ring r) ⟺ |r| = 5 AND r holds an R2 donor
              R2 only blocks bonds touching the heteroatom; the `C=C` of pyrrole, furan or
              thiophene is carbon–carbon, so it escapes and leaks into `Conj`.

     R4       an antiaromatic 4n carbocycle is not delocalized
              forbid(ring r) ⟺ |r| ∈ {4, 8} AND all carbon AND out-of-plane rms > τ_plane
              neutral COT is the tub-shaped D2d form. The planarity test keeps the planar 10π
              η⁸-COT²⁻ delocalized.

     R5       a **lone** Conj bond is not delocalized — delocalization needs a neighbour
              forbid(e) ⟺ neither end of e touches another Conj bond
```

⚠️ **`R3` covers every R2 donor, and the cost of that is one motif.** Restricting it to nitrogen
(the default until 2026-09-08) was never derived — the argument holds for furan O and thiophene S
just as it does for pyrrole N, and the O-only donor five-rings carry an average of **0.10 aromatic
bonds out of 5** in the reference, i.e. they really are Kekulé. Widening it costs `Σq_L` 0.0129
(15 structures) and `OS` 0.0046 (13), and **10 of the 12 structures that get worse are `C₃NO`
five-rings** (isoxazole · oxazoline) with 3 more `C₃NS` — rings **mixing N with O or S**. Excluding
just those was not tried; that condition would itself be read off the losses.

**Why R7 exists** (it lives at stage 5′ of the DAG, but its cause is here): once R2·R3 make a
5-ring Kekulé, that ring has at most two double bonds, so **one atom drops out of the π set** and
an η⁵ ring reads as η⁴. R7 puts that atom back as a haptic candidate **without touching any bond
order**. It is a patch on this section's side effect, not an independent rule.

### ③ Distance likelihood — what the geometry wants

```
score(e, c) = − |d(e) − med[k, c]| / scl[k, c]  +  lp[k, cell(e), c]

  k       = element pair (sorted)             med = per-class distance median
  scl     = 1.4826 × MAD                      lp  = ln P(c | condition)
  cell(e) = (min(deg(X), 4), min(deg(Y), 4))  endpoint internal-degree pair, ordered with k
```

🔴 **The prior is conditioned on the degree cell, not globally per element pair** — because the
same element pair is a different bond at different degrees:

| `C–O` cell | n | Single | **Double** | Triple | Conj |
|---|---|---|---|---|---|
| global | 59,722 | .417 | **.111** | .367 | .105 |
| `deg(C)=3, deg(O)=1` (carbonyl) | 18,737 | .344 | **.322** | .000 | .334 |
| `deg(C)=1, deg(O)=1` (CO ligand) | 22,298 | .000 | .017 | **.983** | .000 |
| `deg(C)=4, deg(O)=2` (ether) | 10,087 | **1.000** | .000 | .000 | .000 |

The global prior penalises a carbonyl by `ln(.111/.417) = −1.32` and flips it to `Single` even when
the distance says `Double`; the cell prior makes that `−0.066` and it disappears.

```
lp[k, cell, c] = ln P(c | k, cell)   when the cell holds ≥ 300 samples
               = ln P(c | k)         below that
exception: c = Conj always uses the global value
```

⚠️ **Two crude edges here, both deliberate.** The 300-sample cut-off is a *statistical* floor (a
cell with 40 samples estimates a prior badly), **not a chemical claim** — a bond whose cell sits
just under it is scored by a different formula than its neighbour. And `Conj` is excluded from
conditioning because the `C–C` deg-3/3 cell has `P(Conj) = .908`, which drags `Double` into `Conj`
(measured +505 errors).

Values in `data/scores4.json` — 15 element pairs · 54 conditioned cells · 26,075 train structures.

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

⚠️ **This is not the exact maximum** (measured 2026-09-08 — the text here used to claim it was).
The greedy `Triple` pass runs before the matching, so it can spend slack the matching would have
used better: **113 of 12,245** calls on holdout are suboptimal, median loss 4.58 in likelihood
units. Solving ④ exactly with a MILP was built and **rejected** — it raised the ④ metrics but made
the *deployment output* worse (harmful `Double` 305 → 313, measured before the 09-08 R3 change), because the stages do not share an
objective.

⚠️ Only bonds with `score(Double) > score(Single)` are offered to the matching. A bond the
likelihood scores as `Single` is never even a candidate for promotion — **114 of the 302 remaining
harmful `Double` errors die here**, before any budget question is asked.

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
stands. Of the harmful `Double` errors that ④ got right and ⑤ demoted, **47% ended with both ends
of the bond negative** — ⑤ was pushing a `C=C` down into `C⁻ C⁻` to hit its target. A *soft*
penalty does not work: with a single candidate a weight only reorders the list and the bad move is
still taken.

**The trust gate.** ⑤ only chases a target it has reason to trust. Three conditions, all the same
statement — *for this class of fragment the bare-fragment Hückel charge is not worth chasing*:

| condition | | target error rate |
|---|---|---|
| parity | `q_EHT − q_current` is odd — unreachable by ±1 moves | — |
| composition | fragment is `NO` · `SS` · `CCHH` | 99.8% · 94.9% · 66.9% |
| motif | an N carrying **exactly two** terminal O (nitro / nitrite) | 84.4% (`R–NO₂`) · 100% (free `NO₂⁻`) |

Overall the target is wrong 3.8% of the time, so the error is concentrated in these classes. Two
details are load-bearing:

- **Exactly two terminal O, not at least two.** Three is nitrate, and the target is right for
  nitrate in **75 of 75** holdout fragments. The first version read `≥ 2` and discarded them for
  nothing.
- **Composition, not size.** A size cut would also disable the 22,298 `CO` fragments, whose target
  is wrong only 1.7% of the time.

⚠️ **`CCHH` (η²-acetylene) is the weakest member of the list at 66.9%** — one third of the time it
is discarding a correct target. It is the item most likely to be fitting the CSD sample rather than
chemistry.

### ⑥ Output converter — integers

Turns the 4 classes into integer S/D/T with a maximum-cardinality matching over the `Conj` bonds,
the same one the charge calculation uses, so the two cannot disagree.

The matching is **weighted** by `score(Double) − score(Single)`: a fragment usually has several
Kekulé structures of the same size, and without weights the one that came out was unrelated to the
bond lengths (`EMAXAR` put the π on a 1.440 Å `C–C` instead of the 1.234 Å `C=O`). The weight also
carries ④'s promise — an atom granted headroom on condition of staying unmatched gets `−10⁶` on
every incident edge, which cannot shrink the matching but keeps that promise.

```
kekulize(G, el, cls, b_ML) → (orders, frag_q)
  orders  {(i,j): 1.0 | 2.0 | 3.0}
  frag_q  charge the skeleton cannot express (tropylium +1 · even-ring dianion −2)
          measured 202 / 26,074 fragments = 0.8%
```

---

## §Charge — `q_L` and `OS(M)` (0 fitted parameters)

```
(a) per-atom formal charge
        lp(X) = max(0, 4 − b)                     lone pair count (max(0, 1 − b) for H)
        q(X)  = v − b − 2·lp(X)

    ⇒ b ≤ 4 :  q = v + b − 8      (octet)         v + b − 2 for H
      b > 4 :  q = v − b          (hypervalent)

    why hypervalency is needed: the old formula `v + b − 8` uses `lp = 4 − b < 0` (a negative lone pair count).
    It counted nitro `–N(=O)=O` (b 5) as +2, sulfone S (b 6) as +4, and perchlorate Cl (b 7) as +6.
    The new formula gives **0** for all of them. Not a single `b ≤ 4` site changes.

(a′) the remaining sites where the octet breaks — covered by an (element, deg, b, neighbor element) table
        heteroatom-stabilized carbene `("C", 2, 2, N or O among neighbors)`  → 0     (octet formula −2)
        sulfoxide      `("S", 3, 4, O among neighbors)`                      → 0     (octet formula +2)
        nitrite        `("N", 2, 4, two O neighbors)`                        → −1    (octet formula +1)
    ⚠️ The neighbor-element condition is essential — keyed on `(element, deg, b)` alone, azide and
       isocyanide get caught too and the ligand charge is off by −2.

(b) conjugated fragment charge
        monocyclic all-carbon `CmHm`  →  Hückel:  z = m − (4n+2) minimizing |m − h| (larger h on a tie)
        otherwise                     →  sum of (a) over the ⑥ Kekulé integers

(c) 3c2e — the decision itself is **T7 at stage 5″** of the DAG, not here. `charge.py` has no
    3c2e branch: a bridging atom gets exactly the same `q_atom(element, b_int)` as any other.
    The tag reaches the charge only **indirectly**, through the internal orders that the ④·⑥
    budget allows:  budget → `b_int` → `q_atom` → `q_L` → `OS(M)`.
    Worked example (Co₂(CO)₈ · GFN2 geometry)

        use(C) = b_int 1 + cost 1 = 2  → headroom 2 → `C≡O` → q(C) −1 → q_L 0 → Co **0**

    CO is a neutral 2e donor whether it bridges or not, so a bridging CO comes out with the
    **same** ligand SMILES as a terminal one, `[O+]#[C-:1]`.

(d) q_L = sum of the formal charges of **all atoms** of the ligand fragment   ← not only the coordinating atoms
          🔴 counted on the **emitted Kekulé integers** (⑥), plus the residual ⑥ returns for the
          charge a skeleton cannot express. Not on the 4-class values: a `Conj` bond is 1.5 there,
          so an atom with three of them reads `b = 3.5` and picks up `−0.5` that no emitted bond
          accounts for. This is the same count the per-atom charges in the SMILES already used, so
          the reported charge and the emitted structure can no longer disagree.
    OS(M) = (q_total − Σ_L q_L) / n_M              ← distributed evenly over the metals
```

⚠️ **M–L is not counted in `b`** — it is an ionic cut, so the coordinating atom takes both M–L electron pairs entirely.

---

## Performance

**Current numbers are at the top of this file.** The table below is the *historical* one — the
holdout as it stood in 2026-09-04, before `R7`, the `B` ligand switch, and everything adopted on
09-07/08. It is kept because it is the record of the split's single blind opening.

| Metric | holdout 6,456 (2026-09-04 code) | train CV 26,075 (same date) |
|---|---|---|
| T3 `Single` / **`Double`** / `Triple` / `Conj` | .9900 / **.7258** / .9822 / .9631 | .9884 / **.7047** / .9809 / .9558 |
| `Σq_L` · `OS(M)` exact match per structure | 80.9% · 81.3% | 82.5% · 83.9% |
| T5 haptic F1 · T6 η^k | .9748 · .9860 | .9789 · .9812 |
| valence violation (structures) | 2.06% | 2.71% |

⚠️ The two T3 columns score **different pools** — the `B` switch brought 30,628 `B–X` bonds into
scoring. On the common bond set `Double` reads .6888 → .6952.

### What is left

- **`Double` is the only weak class.** The rest are .98–.99.
- Feeding the CSD reference bond orders wholesale to the same charge rule gives `Σq_L` **0.8528** ·
  `OS` **0.8698** (re-measured 2026-09-08; the 83.4% / 85.6% printed here before was stale, and it is
  a reference point, **not a ceiling** — its pool differs, and on the common pool the pair reads
  0.8442 → 0.8528 and 0.8649 → 0.8725).
  differs from our Lewis accounting (haptic charge convention · ionic/covalent boundary). **It is not prediction error.**
- The remaining `Double` errors cluster in carbonyl `C–O` · imine `C–N` · thiocarbonyl `C–S` · azo `N–N`,
  and close to half of them are a question of **how far `Conj` is taken**.

### 🔴 Valence violation rate — the axis that is not in the performance table

Even with a high F1, the **output can be chemically impossible**. So the valence ceiling violations are reported alongside:

```
violation(X) ⟺ b_int_kek(X) + b_ML(X) > CAP(X)          X is a non-metal
  b_int_kek : k conjugated bonds count as **k+1** (not a 1.5 conversion) — or **k** where ④ granted
              headroom on the promise that ⑥ leaves the atom unmatched (see ④)
  b_ML      : sum of M–L bond orders (haptic excluded)
  ⚠️ 3c2e-tagged atoms and `B` are dropped from the tally — they are outside the two-center formalism
```

| Evaluation | Structures | Violating atoms | **Violating structures** | Reference-label baseline (same count) |
|---|---|---|---|---|
| **holdout** | 6,456 | 201 / 385,913 = **0.05%** | **133 = 2.06%** | 28 = **0.43%** |
| train CV | 26,075 | 1,045 / 1,547,807 = **0.07%** | **715 = 2.74%** | 103 = **0.40%** |

⚠️ **The baseline is not 0** — feeding the CSD reference labels as they are, **0.4%** of structures violate
(hypervalency · ionic/covalent boundary · CSD notation conventions). Our value has to be read **against that**:
holdout excess **1.6%p**.

**What the 2.74% (train CV) violations are** — μ-bridge 74 · other 969 · 164 exceed on T3 alone.
🔴 A known residual: `B` and `Al` are central atoms, but the Mayer cache holds only true transition metals, so `B–X` is
always missing, and every one of those missing entries becomes an M–L bond that eats the `CAP` budget of the neighboring `C` and `H`
(measured: **6.0% of M–L candidates are missing, all of them `B`-centered**).
**The ⑥ output converter (kekulization) removes the spurious violations** that arise from converting `Conj` to 1.5 —
most of what this metric would otherwise count.

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
| `data/d_int.csv` | T1 per-element-pair distance threshold | 45 + 1 fallback |
| `data/d_bond.csv` | T4 `d_bond` · `w_veto` | 314 (M–L) + 37 (M–M) |
| `data/b_ml_t8forms.csv` | T8 monotone thresholds `t₁ ≤ t₂` | 420 pairs × 2 |
| `data/b_ml_mayer.csv` | T8 likelihood form (old form, fallback) | 420 pairs |
| `data/scores4.json` | T3 distance likelihood `med`·`scl`·`lp`·`lp_cell` | 15 element pairs · 54 cells |

The rules have **0 fitted parameters**: Rule A · R2 · R3 · R4 · R5 · R7 · the ⑤ trust gate
(composition list + nitro motif) · the ⑤ adjacent-same-sign veto · the ④ unmatched-atom exception ·
the hypervalent charge formula.

Global constants — five, and two of them are quoted to more digits than the data supports:

| constant | value | what it is |
|---|---|---|
| `τ_plane` | 0.05 Å | ring planarity tolerance (Rule A · R4) |
| `θ_haptic` | 81.02° | M–X–Y angle below which an M–L bond is haptic — **fitted**; the two decimals are not meaningful |
| EHT cutoff | −10 eV | occupied-orbital cut in the fragment charge |
| `LPCOND_NMIN` | 300 | samples a degree cell needs before its prior is used — a **statistical floor**, not chemistry |
| 3c2e cost | 1.0 | valence a bridging atom spends in total for its M–L bonds — one shared pair is one bond |

⚠️ The `CCHH` entry of the ⑤ composition list is the one place where the choice is **measured
rather than derived** (66.9% target error rate — the weakest of the three); it is flagged in §T3.
`R3`'s scope was such a place until 2026-09-08 and is no longer: it now covers every R2 donor,
which is what its own argument implies.

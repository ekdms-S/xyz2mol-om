# Pipeline — what is decided in what order, by what formula

**Every decision rule** from one `xyz` coming in to bonds, orders, charges, and oxidation states coming out.
Trial and error and the rejection history are not here → `ognm-bh-workspace/docs/analysis/2026-09-03-t3-tuning-history.md`.

Notation. `d(X,Y)` distance (Å) · `w(M,X)` xtb GFN2 **Mayer** bond order · `q_frag` fragment charge ·
`deg(X)` number of **internal** neighbors within the ligand (H included · M–L excluded) · `b_int(X)` sum of internal bond orders ·
`b_ML(X)` sum of M–L bond orders · `v` number of valence electrons.

Class codes: `0 Single · 1 Double · 2 Triple · 3 Conj` (delocalized, formal order 1.5).

---

## Order (DAG)

```
0.  metal / non-metal split                             METALS list

━━ metal-independent ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1.  [T1] internal bond exists ⟺ d(X,Y) < d_int(X,Y)     45 element pairs (data/d_int.csv)
2.       rings = SSSR                                   no parameters
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

3.  [T3] 4-class assignment ①→②→③→④→⑤→⑥                see §T3 below
         🔴 **T3 runs twice** (`pipeline.predict_T3_T5`). The numbers below are *not* a chain
            that follows it — 5, 5″ and the M–L half of 7 happen **between the two passes**
            and are exactly what pass 2's budget is made of.

              pass 1   bml = {} · no M–L optimization   → provisional orders · π fragments
                  ↓    (5 provisional haptic · 5″ bridge tags are read off this)
              pass 2   bml = the budget · M–L scores     → the output orders, incl. M–L

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
             🔴 **There is no fragment-size condition** (the "size ≥ 2" in the first edition of this
             document is not in the code — corrected 2026-09-03). A lone isolated double bond is a π fragment too.

5′. [R7] **Return an R2 donor inside a haptic ring to the π candidates**   (adopted 2026-09-03 · on by default)
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
         (moved here 2026-09-06; it used to be filed under §Charge, which is not where it runs)

           n_center(X) = (M–L bonds of X) + (internal neighbours of X whose element is B·Al)
           deg(X)      = (internal bond **orders** of X, pass-1 Kekule count) + (M–L bonds of X)
           bridge(X) ⟺ n_center >= 2
           3c2e(X)   ⟺ bridge AND el ∈ {H,C,Si,B} AND deg > VALENCE_3C[el] (H 1 · C·Si 4 · B 3)
           dative(X) ⟺ bridge AND the above is false

         🔴 `deg` counts **orders, from pass 1** (2026-09-06). It used to count *neighbours*,
            which misses every bridging atom whose internal bond is multiple — `μ-CO` is that
            case (`1+2=3 ≤ 4` → dative ✗ · `3+2=5 > 4` → 3c2e ✓). Pass-1 orders are used
            because pass 2 needs this tag to build its budget; pass 1 has no metal budget and
            already calls that C–O `Triple`, so there is no circularity.
         real cases  μ-H → 3c2e · B–H···M → 3c2e · **μ-CO → 3c2e** · μ-Cl → dative (3c4e) ·
                     μ-CR₂ bridging carbene → dative (`2+2=4`, genuinely two 2c2e) ·
                     terminal Cl → no tag
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

## §T3 — internal bond orders within a ligand (six stages)

This is the core. The other tasks are one threshold each, but T3 has **several layers of constraints**.

### ① Rule A — pin planar rings to `Conj`

```
pin(e) ⟺ e lies in a ring r and
         |r| ≥ 5
         AND  out-of-plane rms(r) ≤ τ_plane = 0.05 Å
         AND  both end atoms are unsaturated:  deg(X) < CAP(X)
```

`rms` is the root mean square distance of the ring atom coordinates to their best-fit plane.

### ② `Conj` prohibition rules — R2 · R3 · R4

Bonds that **cannot** be `Conj` are cut out first. All three have **zero fitted parameters** and are derived from chemistry.

```
R2  a lone-pair-donating (pyrrole-type) heteroatom cannot be Conj
    forbid(X) ⟺ X ∈ {O, S, Se: deg ≥ 2} ∪ {N, P: deg ≥ 3}   ⇒ remove Conj from every bond of X
    exception ⟺ X = N and the EHT charge of its fragment > 0  (pyridinium N⁺)
    why: furan O and pyrrole N give their own lone pair to the π system, so they are part of π but
        their **formal bond order is 1**. The pyridine type (deg 2) gives a p electron and is 1.5.
        Carbon is always a p-electron donor and so is not covered.

R3  a 5-membered ring containing an R2-flagged nitrogen is Kekulé as a whole
    forbid(ring r) ⟺ |r| = 5 AND every R2 donor in r is nitrogen   ⇒ remove Conj from every bond of r
    why: R2 blocks only the bonds attached to the heteroatom. The `C=C` of pyrrole or imidazole is
        carbon–carbon, so it is not caught and leaks into Conj.

R4  non-planar all-carbon 4n rings (4- and 8-membered) are Kekulé
    forbid(ring r) ⟺ |r| ∈ {4, 8} AND all carbon AND out-of-plane rms > τ_plane
    why: COT is 4n antiaromatic, so it is a neutral tub-shaped D2d. η⁸-COT²⁻ is a planar 10π aromatic,
        and the planarity condition excludes it.
```

### ③ Distance likelihood — the 4-class score of the remaining bonds

```
score(e, c) = − |d(e) − med[k, c]| / scl[k, c]  +  lp[k, cell(e), c]

  k       = element pair (sorted)             med = per-class distance median
  scl     = 1.4826 × MAD                      lp  = ln P(c | condition)
  cell(e) = (min(deg(X), 4), min(deg(Y), 4))  ← endpoint internal-degree pair, ordered to match the element order
```

🔴 **The prior `lp` is conditioned on the degree cell, not globally per element pair.**

```
lp[k, cell, c] = ln P(c | element pair k, degree pair cell)   when the cell has ≥ 300 samples
               = ln P(c | element pair k)                     below that, fall back to global

exception: c = Conj(3) **always** uses the global value
```

Why it is done this way (measured):

| `C–O` cell | n | Single | **Double** | Triple | Conj |
|---|---|---|---|---|---|
| global | 59,722 | .417 | **.111** | .367 | .105 |
| `deg(C)=3, deg(O)=1` (carbonyl) | 18,737 | .344 | **.322** | .000 | .334 |
| `deg(C)=1, deg(O)=1` (CO ligand) | 22,298 | .000 | .017 | **.983** | .000 |
| `deg(C)=4, deg(O)=2` (ether) | 10,087 | **1.000** | .000 | .000 | .000 |

The global prior gives a carbonyl a penalty of `ln(.111/.417) = −1.32` — even when the distance points at
`Double`, it gets flipped. Conditioning on the cell makes it `ln(.322/.344) = −0.066` and it disappears.
Why `Conj` is excluded: the `C–C` `deg 3–3` cell has `P(Conj) = .908`, so conditioning increases the
`Double` → `Conj` leak (measured +505).

Fitted values are in `data/scores4.json` — 15 element pairs · 54 conditioned cells · 26,075 train structures.

### ④ Valence ceiling — the **exact solution** of a hard constraint

```
maximize    Σ_e  score(e, c(e))
subject to  b_int_kek(X) + b_ML(X)  ≤  CAP(X)        for every non-metal X

  b_int_kek : k conjugated bonds count as **k + 1** (one of them is π). Not a 1.5 conversion.
  b_ML      : **not** an order sum — one unit per existing M–L bond (`pipeline.bml_budget`).
              haptic spends 0. An atom taking part in a **3c2e** spends `BML3C_COST` (default
              **1.0**) *in total* regardless of how many M–L bonds it has — one electron pair
              across three centers is one bond of valence (2026-09-06). `BML3C_COST=-1` restores
              the old one-per-bond counting, `0.0` is the `BMLSKIP3C` behaviour below.
              ⚠️ The same budget is rebuilt for ⑥ and the charge in `api.predict` — before
                 2026-09-06 that copy had **no** 3c2e term, so ⑥ could undo what ④ allowed.
              ⛔ Excluding 3c2e (`BMLSKIP3C=1`) was implemented, remeasured by CV over the whole
                 train set, and **rejected** (2026-09-03): `Double` .7157 → **.7137** · `Triple` .9814 → .9806 ·
                 `Σq_L`, `OS`, `T6` identical · structures with a valence violation 715 → 713 (2 structures).
                 The argument "μ-H has `CAP(H)=1` but `b_ML=2`, so it is unsatisfiable" is correct, but
                 **μ-H has zero internal bonds, so that constraint binds nothing in the first place.**
                 What it was actually binding is 3c2e **carbon**, and releasing it raises orders wrongly.
                 The flag remains and its default is **0 (off)**.
  CAP       : H 1 · B 4 · C 4 · N 5 · O 4 · F 4 · Si 6 · P 6 · S 6 · Cl 7 · Br/Se/As/Te/I 6
```

🔴 **There is no lower bound.** This is an ionic cut, so an anionic ligand is normal (`Cl⁻`·`RO⁻`·`Cp⁻`).
A deficient valence is not an error — it is a **negative charge**.

How it is solved: take the slack `r(X) = ⌊CAP(X) − use(X)⌋` as a capacity, duplicate each atom that many
times, and solve a **maximum weight matching** (Blossom) — a polynomial-time **exact solution**. `Triple` is
assigned first to bonds whose likelihood argmax is `Triple` and whose endpoints both have slack ≥ 2, consuming that slack.
M–L orders are decided inside the same matching (a dummy node expresses the metal side being unconstrained, up to `Triple`).

⚠️ The matching admits as candidates only bonds with `g(e) = score(e, Double) − score(e, Single) > 0`.

### ⑤ EHT fragment charge target

Extended Hückel (RDKit `rdEHTTools`) computes the charge of a ligand fragment **directly**, and it enters the search as a target.

```
q_frag(EHT) = Σ v_i  −  2 × #{Hückel orbitals : E < −10 eV}   (+ HOMO/LUMO correction)

identity    q_frag = C0(composition) + 2B,   C0 = Σ v_i − 8n   (2 for H)
        ⇒   B* = (q_EHT − C0) / 2      the **total bond order of that fragment is fixed to a single integer**

assignment ⟺ make B = B* while respecting the ceiling of ④
             if short, +1 starting from the bonds with the largest likelihood gain · if over, −1 starting from the smallest loss (≤ 12 rounds)
```

Fragments that are skipped:

```
(a) fragments where (q_EHT − q_current) is odd
(b) 🔴 fragments whose composition is `NO`, `SS`, or `CCHH`     ← EHTSKIP
```

Basis for (b) (against the reference assignment · all train fragments):

| Composition | Fragments | Target error rate | Error |
|---|---|---|---|
| `CO` (carbonyl ligand) | 22,298 | **1.7%** | — |
| **`NO`** (nitrosyl) | 493 | **99.8%** | −2 in 489/492 |
| **`SS`** | 118 | **94.9%** | +2 in 112/112 |
| **`CCHH`** (η²-acetylene) | 142 | **66.9%** | +2 in 91/95 |

⛔ **Cutting by size is wrong** — turning off every two-atom fragment also turns off the 22,298 `CO` and loses
(`Double` .6913 → .6861 · `Σq_L` .7932 → .7908).

### ⑥ Output converter (kekulization)

Turns the 4 classes back into integer S/D/T. It uses the **same maximum matching** as the charge calculation, so the assignments agree.

```
kekulize(G, el, cls, b_ML) → (orders, frag_q)
  orders  {(i,j): 1.0 | 2.0 | 3.0}
  frag_q  residual fragment charge not expressible by the skeleton
          (tropylium +1 · even-ring dianion −2 etc., measured 202/26,074 = 0.8%)
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
        otherwise                     →  sum of (a) after Kekulé maximum matching

(c) 3c2e — the decision itself is **T7 at stage 5″** of the DAG, not here.
    🔴 Corrected 2026-09-06. This section used to say "atoms taking part in a 3c2e are filtered
       out before (a)". **`charge.py` does no such thing** — it has no 3c2e branch at all, and a
       bridging atom gets exactly the same `q_atom(element, b_int)` as any other. What the tag
       actually reaches is the ④·⑥ **budget** (`BML3C_COST`) and the violation tally.
    So the tag moves the charge only **indirectly**, through the internal orders the budget
    allows:  budget → `b_int` → `q_atom` → `q_L` → `OS(M)`.
    Worked example (Co₂(CO)₈ · GFN2 geometry · measured 2026-09-06)

        cost per bond (old)   use(C) = 2 + 1 = 3  → headroom 1 → `C=O`  → q(C) −2 → q_L −2 → Co **+2**
        cost 1 in total       use(C) = 1 + 1 = 2  → headroom 2 → `C≡O`  → q(C) −1 → q_L  0 → Co **0**

    The second row is the conventional assignment (CO is a neutral 2e donor whether it bridges
    or not), and it makes a bridging CO come out with the **same** ligand SMILES as a terminal
    one, `[O+]#[C-:1]`.

(d) q_L = sum of the formal charges of **all atoms** of the ligand fragment   ← not only the coordinating atoms
    OS(M) = (q_total − Σ_L q_L) / n_M              ← distributed evenly over the metals
```

⚠️ **M–L is not counted in `b`** — it is an ionic cut, so the coordinating atom takes both M–L electron pairs entirely.

---

## Performance

### holdout (6,456 structures not used in the fit · **opened only once**)

| Metric | **holdout 6,456** ⚠️ old code | **train CV 26,075** ★ current code |
|---|---|---|
| T3 `Single` / **`Double`** / `Triple` / `Conj` | .9900 / **.7258** / .9822 / .9631 | .9884 / **.7047** / .9809 / .9558 |
| `Σq_L` exact match per structure | **80.9%** | **82.5%** |
| `OS(M)` exact match per structure | **81.3%** | **83.9%** |
| T5 haptic F1 · T6 η^k | .9748 · .9860 | **.9789** · .9812 |
| valence violation (structures) | 2.06% | **2.71%** |

⚠️ **The holdout column was measured with the code *before* the `R7` adoption and the `B` ligand switch.**
The holdout is spent by its single opening and cannot be remeasured ⇒ **generalization of the current code can be
stated only from the train CV column.**
⚠️ **The two T3 columns also score different pools** — the `B` switch newly brought 30,628 `B–X` bonds into scoring.
Compared over the same bond set, `Double` is .6888 → **.6952**.

### What is left

- **`Double` is the only weak class.** The rest are .98–.99.
- The **ceiling of `Σq_L` and `OS` is 83.4% / 85.6%**. Even feeding the CSD reference bond orders wholesale
  does not go beyond that — the `charge` of the reference labels (tmQMg-L) is **NBO-derived**, so its convention
  differs from our Lewis accounting (haptic charge convention · ionic/covalent boundary). **It is not prediction error.**
- The remaining `Double` errors cluster in carbonyl `C–O` · imine `C–N` · thiocarbonyl `C–S` · azo `N–N`,
  and close to half of them are a question of **how far `Conj` is taken**.

### 🔴 Valence violation rate — the axis that is not in the performance table

Even with a high F1, the **output can be chemically impossible**. So the valence ceiling violations are reported alongside:

```
violation(X) ⟺ b_int_kek(X) + b_ML(X) > CAP(X)          X is a non-metal
  b_int_kek : k conjugated bonds count as **k+1** (not a 1.5 conversion)
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
⛔ A version subtracting 3c2e-participating atoms from the budget (`BMLSKIP3C=1`) was tried and **rejected** —
violating structures drop only 715 → 713 (**2 structures**) while `Double` falls .7157 → **.7137**.

**The ⑥ output converter (kekulization) already solved most of this metric** — it removes the **spurious** violations
that arose from converting `Conj` to 1.5, cutting the violation rate **4.13% → 0.48%**, an 8.6× reduction (the version at that time).

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

The rules (Rule A · R2 · R3 · R4 · R5 · **R7** · `EHTSKIP` · the hypervalent charge formula) have **0 fitted parameters**.
The only global constants are the three `τ_plane = 0.05 Å` · `θ_haptic = 81.02°` · EHT cutoff `−10 eV`.

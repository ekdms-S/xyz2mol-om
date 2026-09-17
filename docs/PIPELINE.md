# Pipeline — what is decided in what order, by what formula

**Every decision rule** from one `xyz` coming in to bonds, orders, charges and oxidation states
coming out: the rules, the formulas, the thresholds and the output contracts.

Notation. `d(X,Y)` distance (Å) · `w(M,X)` xtb GFN2 **Mayer** bond order · `q_frag` fragment charge ·
`deg(X)` number of **internal** neighbors within the ligand (H included · M–L excluded) · `b_int(X)` sum of internal bond orders ·
`b_ML(X)` valence X spends on M–L bonds (see ④) · `n_ML(X)` **number** of M–L bonds of X ·
`n_σ(X)` number of **non-haptic** M–L bonds of X · `v` number of valence electrons.

## Performance

| task | holdout 6,793 | train 27,294 |
|---|---|---|
| T1 internal bond existence | **.9998** | .9998 |
| T3 `Single`/`Double`/`Triple`/`Conj` | **.9901 / .7667 / .9775 / .9617** | .9895 / .7601 / .9792 / .9592 |
| T4 M–L·M–M existence | **.9915** | .9927 |
| T5 haptic | **.9800** | .9807 |
| T6 η^k | **.9865** | .9832 |
| T8 M–L `Single`/`Double`/`Triple` | **.9935 / .7556 / .7254** | .9937 / .7621 / .7735 |
| T10 `Σq_L` · `OS` | **.8648 · .8967** | .8580 · .8910 |
| valence-violating structures | **.0035** | .0037 |

Holdout: 6,793 structures. References, pools and baselines: README `## Performance`.

**Fitted:** `LPA` (§T3 ③), `θ` (DAG 5) and the tables in §Fitted parameter summary; every other
constant is in the constants table.

---

## Terms

Short names used throughout. Task codes `T1`·`T3`… are defined in README `## Performance`, except
`T7` = the bridge tag (DAG 5″); stage numbers `①`…`⑥` are T3's internal order.

| term | meaning |
|---|---|
| `Conj` | a bond in a **delocalized π system**; ⑥ resolves it to 1 or 2 |
| `CAP(X)` | X's **valence ceiling**: `b_int + b_ML ≤ CAP` (values in ④) |
| haptic · `η^k` | an M–L bond to an atom of a π system (`η⁵`-Cp) rather than to a lone pair. `k` = how many atoms of that **ligand** bind the same metal haptically |
| 3c2e · dative | a bridging atom's tag: **3c2e** = one electron pair shared over three centres (μ-H, μ-CH₃, μ-CO, B–H–B); **dative** = two 2-centre donations (μ-Cl) |
| `q_L` · `OS(M)` | **ligand charge** — the formal charges of every ligand atom, summed **on the emitted Kekulé integers** · **metal oxidation state**, what is left of the complex charge after the ligands |
| the veto (⑤) | ⑤ may not create a **new pair of adjacent same-sign formal charges** (no `C⁻ C⁻`) |
| trust gate (⑤) | fragment classes whose extended-Hückel charge target is not trusted (parity · composition · nitro motif) |
| Rule A · R2–R5 · R7 | the named chemistry rules. A · R2 · R3 · R4 · R5 build the conjugation set (§T3 ①②); R7 returns an R2 donor to the haptic test (DAG 5′) |

---

## Order (DAG)

The numbered list below the picture is a per-step reference, not the execution order.

```text
                    0. metal / non-metal split  (METALS · `config.centers`)
                                       │
             ┌─────────────────────────┴─────────────────────────┐
             ↓                                                   ↓
   1. [T1] internal bond exists                    4. [T4] M–L · M–M bond **exists**
   2.      rings = `nx.cycle_basis`                        existence only — no type, no order
        ── metal-independent ──                                  │
             │                                                   │ contact set
             │                                                   ↓  = `coord`
             └─────────────────────────┬─────────────────────────┘
                                       ↓
                   3. [T3] pass 1     bml = {} · no M–L orders
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
                   3. [T3] pass 2     bml · M–L order scores (7)
                                       │   7. [T8] M–L orders are solved **inside** this
                                       ↓      same ④ matching
  final internal orders · final M–L orders · 5. [T5] final haptic (pass-2 π) + 5′. [R7] + 5‡
                                       ↓
       ⑥ output converter  →  post-⑥ repairs  →  6. [T6] η^k · 8. [T10] q_L · OS(M)
```

```
0.  metal / non-metal split                             METALS list · `config.centers`

━━ metal-independent ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1.  [T1] internal bond exists ⟺ d(X,Y) < d_int(X,Y)     58 element pairs + fallback (data/d_int.csv)
           never for H–H, nor when d(X,Y) > 1.8 × (r_cov(X) + r_cov(Y))
           a B–B is dropped when the two B share ≥ 2 H neighbours and neither has another B neighbour
2.       rings = `nx.cycle_basis`
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

3.  [T3] 4-class assignment ①→②→③→④→⑤→⑥                see §T3 below
         🔴 **runs twice** (`rules.pipeline.predict_T3_T5`)
              pass 1   bml = {} · no M–L orders                → provisional orders · π fragments
              pass 2   bml = the budget · M–L order scores (7)  → the output orders, M–L included

4.  [T4] M–X bond **exists**  ⟺   d(M,X) < d_bond(M,X)  AND  w(M,X) > w_veto(M,X)
         existence only — no type and no order.       element pairs 316 (M–L) · 23 (M–M)
         X is any atom, metals included — (M,M) pairs are decided here too.
         the T4 contact set, before the exclusions below, is passed to both T3 passes as `coord`
           (no under-valence penalty in ①②, no `dq` in ④)
         agostic excluded: `X–H···M` is not a bond
           ⟺ that H has exactly one metal-like neighbour AND an internal neighbour that is not
             metal-like        (μ-H and `B–H···M` are kept)
         agostic carbon excluded: X = C with an H nearer M and d(M,H) < AGOC = 2.0 Å, unless M keeps no bond to that fragment
         saturated atoms excluded: no M–X bond to an atom its own bonds already fill up
           ⟺ el(X) ∉ {H, B, Al}  AND  no internal neighbour of X is B or Al  AND  deg(X) ≥ CAP(X)
         bound halogen excluded: X ∈ {F, Cl, Br, I} with an internal neighbour AND w(M,X) < HALW = 0.30 (only with `wbo`)

5.  [T5] that bond is haptic
           ⟺  ∠(M–X–Y) < θ = 81.02°   AND  X belongs to a π fragment
           **or** the bond-level test 5* fires for a π bond X is an end of
           Y = the neighbour of X within its π fragment whose **bond midpoint is closest to M**
           if false, the bond is σ-dative
           π fragment = connected component of {Conj ∪ Double ∪ Triple} bonds
           ⇒ **X belongs to a π fragment ⟺ X touches a `Double`, `Triple`, or `Conj` bond.**

5†. **η¹ is a σ bond, so it is not haptic** (`rules.pipeline.drop_eta1`)

           drop(M, X) ⟺ X's **ligand fragment** gives M exactly one haptic atom

         🔴 Applied to the **pass-1** haptic set too, and before 5′ and 5‡.

5*. **both ends of a π bond are haptic together** (`rules.pipeline._eta2_pair`)

           both(M, X–Y) ⟺ X–Y is an internal bond with π character (`Double`/`Triple`/`Conj`)
                       AND X and Y both have a T4 bond to the **same** M
                       AND neither X nor Y is H
                       AND ∠(M–X–Y) < θ  **or**  ∠(M–Y–X) < θ

         🔴 applied in **pass 1 as well**.

5′. [R7] **Return an R2 donor inside a haptic ring to the π candidates**

           add(M,X) ⟺ X is an R2 donor (O·S·Se: deg ≥ 2 · N·P: deg ≥ 3)
                    AND X belongs to a ring r with |r| = 5  (r from `nx.cycle_basis`)
                    AND at least R7MIN = 2 of the **other atoms** of r passed 5 for the **same metal M**
                    AND (M,X) is a T4 bond  (d < d_bond AND w > w_veto)
                    AND ∠(M–X–Y) < θ = 81.02°
           Y = among X's neighbours that belong to a π fragment, the one whose bond midpoint is closest to M

         ⇒ **T3 bond orders are not changed.** Only 5's "X belongs to a π fragment" condition is
           waived.

5‡. [T5] **A σ M–L that overfills a π atom is haptic**

           haptic(M,X) ⟺ X touches a `Double`/`Triple`/`Conj` bond  AND  X is neither `B` nor 3c2e
                         AND  b_int(X) + n_σ(X) > CAP(X)       applied until X fits

5″. [T7] bridge tag — the **type** of an existing M–L bond, `rules.pipeline.bridge_tags`

           n_center(X) = n_ML(X) + (internal neighbours of X whose element is B·Al; 0 if X is B·Al)
           b_use(X)    = b_int(X) + n_ML(X)        b_int from the **pass-1** Kekulé orders
           bridge(X) ⟺ n_center >= 2
           3c2e(X)   ⟺ bridge AND el ∈ {H,C,Si,B} AND b_use > VALENCE_3C[el] (H 1 · C·Si 4 · B 3)
           dative(X) ⟺ bridge AND the above is false

           μ-H       M–H–M         b_use 0+2 = 2 > 1  →  3c2e
           μ-CH₃     M–CH₃–M       b_use 3+2 = 5 > 4  →  3c2e
           μ-CO      M–CO–M        b_use 3+2 = 5 > 4  →  3c2e    (C≡O)
           B–H···M   borohydride   n_center = M 1 + neighbour B 1 · b_use 2 > 1  →  3c2e
           B–H–B     diborane      n_center = neighbour B 2 (**no metal**) · b_use 2 > 1 → 3c2e
           μ-CR₂     bridging carbene   b_use 2+2 = 4  →  dative
           μ-Cl      M–Cl–M        Cl ∉ VALENCE_3C    →  dative (3c4e)
           terminal  M–L           n_center 1         →  no tag

         output      `ml_bonds[(m,x)]["bridge"]` = None|"3c2e"|"dative" · `["type"]` is
                     haptic > bridge > sigma · ligand-internal legs in `fragment["bonds_3c2e"]`
                     the output tag is recomputed after T5 with haptic M–L left out of `b_use`
         ⚠️ A bridge keeps both M–L bonds, and 7 assigns their orders.

6.  [T6] η^k     k = number of atoms of that ligand haptic to the same metal
         Ferrocene's two rings are two ligands, so it comes out as two η⁵.

         ⚠️ **A bridged (ansa) metallocene comes out as one η¹⁰, not η⁵:η⁵** — the two rings are
            joined, so they are one ligand; `ml_bonds` lists the ten atoms.

7.  [T8] M–L order (non-haptic bonds only)
           Single ⟺ w < t₁(M,X)    Double ⟺ t₁ ≤ w < t₂    Triple ⟺ w ≥ t₂
           57 element pairs × (t₁, t₂) · 349 pairs with one fixed class · else Single (data/b_ml_t8forms.csv)
           without `wbo`: Double ⟺ d(M,X) ≤ t₁′(M,X) · Triple ⟺ d ≤ t₂′ (data/b_ml_dist.csv)
           an infinite threshold: that class never occurs for the pair
           ④ lowers it when X has no valence left

8.  [T10] ligand charge · oxidation state                → §Charge below
```

---

## §T3 — internal bond orders within a ligand

T3 answers **six questions in order**, each constraining the next.

| | question | how |
|---|---|---|
| **①②** | which bonds share a **delocalized π system**? | Rule A · R2 · R3 · R4 · R5 |
| **③** | what order does the **geometry** want? | per-element-pair distance likelihood, prior damped by `LPA = 0.8` |
| **④** | what can the **valence budget** afford? | hard constraint, maximum-weight matching |
| **⑤** | does the fragment's **electron count** agree? | extended-Hückel fragment charge as a target |
| **⑥** | emit **integers** | Kekulé matching |
| **post-⑥** | is there a **valid assignment with less charge**? | `QSHIFT` · `QGEM` · `SIGCUT`, accepted only if `Σ|q|` falls |

### ①② The conjugation set — which bonds are delocalized

The ③ likelihood and Rule A put bonds in, four rules take them out.

```
IN   ③        a bond the likelihood labels `Conj` after a valence-repair search (saturated end: `Single` only)

IN   Rule A   a ring is delocalized if it is planar and its atoms are unsaturated
              pin(e) ⟺ e lies in a ring r (`nx.cycle_basis`) · |r| ≥ 5
                       · out-of-plane rms(r) ≤ τ_plane = 0.05 Å
                       · both ends unsaturated (deg(X) + b_ML(X) < CAP(X))
              a pinned bond is **forced** to `Conj`, overriding ③ — but R2·R3·R4 are
              subtracted from the pinned set first, so an exclusion always wins

OUT  R2       a **lone-pair donor** heteroatom is part of π but its own bonds are order 1
              forbid(X) ⟺ X ∈ {O,S,Se: deg ≥ 2} ∪ {N,P: deg ≥ 3}  ⇒ no Conj on any bond of X
              exception ⟺ X = N and its fragment's EHT charge > 0   (pyridinium N⁺)

     R3       a 5-ring holding an R2 donor is Kekulé **as a whole**
              forbid(ring r) ⟺ |r| = 5 AND r holds an R2 donor

     R4       an antiaromatic 4n carbocycle is not delocalized
              forbid(ring r) ⟺ |r| ∈ {4, 8} AND all carbon AND out-of-plane rms > τ_plane

     R5       a **lone** Conj bond is not delocalized
              forbid(e) ⟺ neither end of e touches another Conj bond
```

⚠️ **Known cost of R3:** five-rings **mixing N with O or S** (isoxazole · oxazole · thiazole)
are aromatic azoles, and R3 flattens them to Kekulé.

### ③ Distance likelihood — what the geometry wants

```
score(e, c) = − |d(e) − med[k, c]| / scl[k, c]  +  LPA · lp[k, cell(e), c]

  k       = element pair (sorted)             med = per-class distance median
  scl     = 1.4826 × MAD                      lp  = ln P(c | condition)
  cell(e) = (min(deg(X), 4), min(deg(Y), 4))  endpoint internal-degree pair, ordered with k
  LPA     = 0.8                               prior temperature — fitted

lp[k, cell, c] = ln P(c | k, cell)   when the cell holds ≥ LPCOND_NMIN = 300 samples
               = ln P(c | k)         below that
exception: c = Conj always uses the global value
```

Values in `data/scores4.json` — 19 element pairs · 60 conditioned cells, fitted on the train split.

### ④ Valence budget — what can be afforded

```
maximize    Σ_e score(e, c(e))  +  Σ_{e raised to Double} (dq(X) + dq(Y))
subject to  b_int(X) + b_ML(X)  ≤  CAP(X)         for every non-metal X

  b_int  k conjugated bonds cost **k + 1**. **Exception:** in a π fragment whose maximum matching
         leaves n atoms over, up to n atoms that can be left unmatched cost **k** (largest gain
         first); ⑥ is then held to that promise.
  b_ML   1 per non-haptic M–L bond, +1 for each step ④ raises its order; haptic costs 0;
         a **3c2e** atom pays `BML3C_COST` = 1.0 *in total* for its M–L bonds; the post-⑥ repairs
         price it the same way
  dq(X)  |q(b_int)| − |q(b_int + 1)| with q from §Charge (a)·(a′); 0 for an atom in `coord` (DAG 4)
  CAP    H 1 · B 4 · C 4 · N 5 · O 4 · F 4 · Si 6 · P 6 · S 6 · Cl 7 · Br/Se/As/Te/I 6
```

How it is solved: take the slack `r(X) = min(2, ⌊CAP(X) − use(X)⌋)` as a capacity, replicate each
atom that many times, and run a **maximum-weight matching** (Blossom). `Triple` is confirmed first,
greedily, for bonds whose likelihood argmax is `Triple` and whose ends both have slack ≥ 2.

⚠️ **Not the exact maximum** — the greedy `Triple` pass can spend slack the matching would have
used better.
⚠️ Only bonds with `score(Double) − score(Single) + dq(X) + dq(Y) > 0` are offered to the matching.

### ⑤ Fragment electron count — does the charge agree

Extended Hückel (RDKit `rdEHTTools`) gives a ligand fragment's charge directly, and that fixes the
fragment's **total** bond order to one integer:

```
q_frag(EHT) = Σ v_i − 2 × #{Hückel orbitals with E < −10 eV}   (+ HOMO/LUMO correction)
identity     q_frag = C0(composition) + 2B,   C0 = Σ v_i − 8n   (2 for H)
         ⇒   B* = (q_EHT − C0) / 2

assignment ⟺ move non-`Conj` bonds ±1 to reach B = B*, respecting ④'s ceiling, ≤ 12 rounds
             short → +1 from the largest likelihood gain · over → −1 from the smallest loss
```

**The veto.** A move that creates a **new pair of adjacent same-sign formal charges** is dropped
from the candidate list; if nothing else is available ⑤ stops and keeps the moves already made.

**The trust gate.** ⑤ skips the target when:

| gate | condition |
|---|---|
| parity | `q_EHT − q_current` is odd |
| composition | fragment is `NO` · `SS` · `CCHH` |
| motif | an N carrying **exactly two** terminal O (nitro / nitrite; nitrate is not gated) |

### ⑥ Output converter — integers

Turns the 4 classes into integer S/D/T with a maximum-cardinality matching over the `Conj` bonds.
Edge weight: `score(Double) − score(Single)` on a per-element-pair grid of `KEKQ` = 8 (ties by atom
index); `+10⁵` if both ends are a metal's only two haptic atoms; `−10⁶` if an end is an atom ④
promised to leave unmatched.

⑥ outputs integer orders 1/2/3 and, per π fragment, the charge the skeleton cannot express
(an even-ring dianion's −2).

**`pi_suppressed`.** Every fragment lists the bonds that match:

```
suspect(i,j) ⟺ orders[(i,j)] == 1  AND  q(i) < 0  AND  q(j) < 0
                AND  score(Double) − score(Single) > 0        ← raw likelihood, before ④'s −10⁶
```

Element pairs with no `Double` class fitted (`As–C`) are excluded. ⚠️ It is a **flag, not a
correction**: the orders and charges are returned unchanged.

### Post-⑥ repairs — a π in the wrong place, and a σ M–L that pays for it

Three rules act on the emitted integers; each change is kept only if `Σ|q|` falls.

```
QSHIFT  charges at the two ends of an alternating path of ≤ 5 bonds — flip the path   `C⁻–C=C–O⁻ → C=C–C=O`
        an opposite-sign pair on one bond is left alone (CO, amine oxide, ylide)
QGEM    two like-signed charges on one common neighbour — shift both bonds     `[O⁻]–C–[O⁻] → O=C=O`
          raise for anions, lower for cations; only when the bond lengths fit both new orders
SIGCUT  a pair that only a σ M–L valence unit keeps apart — drop that σ M–L and rerun T3 (both
          passes, T5, 5″ tags), ⑥ and QSHIFT·QGEM without it
          cut only when  w(M,X) < SIGCUTW = 0.40, or (adjacent pair only) the raised order
                         fits the bond length better and the current order lies outside its own distribution
          never cut      a bond to an `Al` centre; any bond with no Mayer value
```

`QSHIFT` and `QGEM` skip a pair whose two ends both coordinate a metal; `SIGCUT` skips one whose two
ends coordinate the same metal (a dianionic ligand such as dithiolene or catecholate). Peroxide
`[O⁻]–[O⁻]` is never raised. (a″) of §Charge is applied again after the repairs.

---

## §Charge — `q_L` and `OS(M)`

```
(a) per-atom formal charge
        lp(X) = max(0, 4 − b)                     lone pair count
        q(X)  = v − b − 2·lp(X)

    ⇒ b ≤ 4 :  q = v + b − 8      (octet)         v + b − 2 for H
      b > 4 :  q = v − b          (hypervalent: sulfone S, perchlorate Cl → 0)
    B:  lp = max(0, 3 − b), and 0 when B holds an internal 3c2e leg   (B(OH)₃ 0 · boryl [BR₂]⁻ −1)

(a′) the remaining sites where the octet breaks — an (element, deg, b, neighbor element) table
        free carbene   `("C", 2, 2, no N or O neighbor, no M–L bond)`        → 0     (octet formula −2)
        heteroatom-stabilized carbene `("C", 2, 2, N or O among neighbors)`  → 0     (octet formula −2)
        sulfoxide      `("S", 3, 4, O among neighbors)`                      → 0     (octet formula +2)
        nitrite        `("N", 2, 4, two O neighbors)`                        → −1    (octet formula +1)

(a″) **nitrogen never carries five bonds.** After ⑥, one `N=O` to a **terminal** O of a
     five-bonded N is demoted to `N–O` and the charge follows from (a): N `+1`, that O `−1`.
     The fragment total is unchanged.

(b) conjugated fragment charge
        monocyclic all-carbon ring, one single bond out of each atom (`CmHm`, `C₅Me₅`)
                                      →  Hückel:  z = m − h, h = the member of {2, 6, 10, 14, 18}
                                                  nearest m (larger on a tie)
        otherwise                     →  sum of (a) over the ⑥ Kekulé integers

(c) 3c2e (tagged at DAG 5″) — a bridging atom on a metal is charged like any other atom; the tag
    acts only through the budget it sets for `b_int`. An internal leg to a B/Al holds no pair when
    the bridging atom is H or has two such legs (the `B–H` of `B–H–B` or κ²-BH₄, listed in
    `fragment["bonds_3c2e"]`); its order is subtracted from `b` before (a): B₂H₆ is `B(+1)` ·
    bridging `H(−1)` · terminal `H(0)`.

(d) q_L = sum of the formal charges of **all atoms** of the ligand fragment
          🔴 counted on the **emitted Kekulé integers** (⑥), plus the residual ⑥ returns for the
          charge a skeleton cannot express
    OS(M) = (q_mol − Σ_{L ∈ mol} q_L) / n_M(mol)   ← evenly over that molecule's metals
          mol = connected component over all bonds; several metal-bearing molecules are pooled
          q_mol = q_total − the metal-free molecules' charges

(e) cluster — a fragment holding an atom whose Kekulé bond-order sum exceeds CAP takes its EHT charge as q_L
```

⚠️ **M–L is not counted in `b`** — it is an ionic cut, so the coordinating atom takes both electrons of each M–L bond.

---

## Known weak spot

⚠️ `Double` is the weakest T3 class: a heteroatom double bond (`C=N` · `C=S` · `C=O` · `N=N` ·
`N=O` · `C=Se`) whose degree-cell prior outweighs its length is not offered to ④ (§T3 ③ ④).

---

## Valence violation

```
violation(X) ⟺ b_int_kek(X) + n_σ(X) > CAP(X)          X is a non-metal
  b_int_kek : sum of the emitted bonds_kekule integers
  n_σ(X)    : number of non-haptic M–L bonds of X
  3c2e-tagged atoms and `B` are excluded
```

The CSD reference labels themselves violate on about 0.4% of structures.

---

## Fitted parameter summary

| File | What | Count |
|---|---|---|
| `data/d_int.csv` | T1 per-element-pair distance threshold | 58 + 1 fallback |
| `data/d_bond.csv` | T4 `d_bond` · `w_veto` | 316 (M–L) + 23 (M–M) + 1 fallback |
| `data/b_ml_t8forms.csv` | T8 Mayer thresholds `t₁ ≤ t₂` | 57 pairs × 2 · 349 pairs one fixed class · 1 fallback |
| `data/b_ml_dist.csv` | T8 distance thresholds, read only without `wbo` | 57 pairs × 2 · 349 pairs one fixed class · 1 fallback |
| `data/scores4.json` | T3 distance likelihood `med`·`scl`·`lp`·`lp_cell` | 19 element pairs · 60 cells |

Global constants:

| constant | value | what it is |
|---|---|---|
| `LPA` | 0.8 | prior temperature in the ③ likelihood — **fitted** |
| `θ` | 81.02° | M–X–Y angle below which an M–L bond is side-on — **fitted** |
| `τ_plane` | 0.05 Å | ring planarity tolerance (Rule A · R4) |
| `SIGCUTW` | 0.40 | Mayer ceiling below which `SIGCUT` may cut a σ M–L |
| `HALW` | 0.30 | Mayer floor for an M–halogen bond whose halogen has an internal bond (T4) |
| `AGOC` | 2.0 Å | M···H distance below which an agostic carbon loses its M–C bond (T4) |
| `KEKQ` | 8 | per-element-pair grid of the ⑥ matching weights |
| `R7MIN` | 2 | other haptic ring atoms R7 needs |
| T1 radius cap | 1.8 | no internal bond beyond 1.8 × (r_cov(X) + r_cov(Y)) |
| EHT cutoff | −10 eV | occupied-orbital cut in the fragment charge |
| `LPCOND_NMIN` | 300 | samples a degree cell needs before its own prior is used |
| `BML3C_COST` | 1.0 | valence a 3c2e atom spends in total for its M–L bonds |

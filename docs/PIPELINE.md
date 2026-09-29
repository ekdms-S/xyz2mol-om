# Pipeline — what is decided in what order, by what formula

**Every decision rule** from one `xyz` coming in to bonds, orders, charges and oxidation states
coming out: the rules, the formulas, the thresholds and the output contracts.

The pipeline has two parts. **Topology** decides which atoms are bonded (ligand-internal bonds,
metal contacts, haptic units, boron cages). **One mixed-integer linear program** then decides every
ligand bond order, atom charge, haptic reading, metal oxidation state, cluster charge and unpaired
electron together, under the charge balance `Σ q_L + Σ OS = total_charge`, and a few distinct
answers are compared.

Notation. `d(X,Y)` distance (Å) · `w(M,X)` xtb GFN2 **Mayer** bond order · `deg(X)` number of
**internal** neighbours within the ligand (H included · M–L excluded) · `b(X)` sum of internal bond
orders · `v` number of valence electrons · `λ` = `JOINTQ`, the unit every penalty is counted in.

## Performance

| task (CSD holdout, 6,396 structures) | F1 / accuracy |
|---|---|
| T1 internal bond existence | .9999 |
| T3 `Single` / `Double` / `Triple` / `Conj` | .9908 / .7946 / .9836 / .9614 |
| T4 M–L · M–M existence | .9923 |
| T5 haptic | .9824 |
| T6 η^k | .9911 |
| T8 M–L `Single` / `Double` / `Triple` | .9939 / .7641 / .7160 |
| T10 `Σq_L` · `OS` | .9396 · .9859 |
| valence-violating structures | 0 |

References, pools and definitions: README `## Performance`.

---

## Terms

| term | meaning |
|---|---|
| `Conj` | a bond in a **delocalized π system**: flipping its S/D alternation costs almost nothing (see *Reading the answer*) |
| `CAP(X)` | X's valence ceiling: H 1 · B 4 · C 4 · N 5 · O 4 · F 4 · Si 6 · P 6 · S 6 · Cl 7 · Br/Se/As/Te/I 6 |
| haptic · `η^k` | an M–L bond to an atom of a π system (`η⁵`-Cp) rather than to a lone pair. `k` = how many atoms of that **ligand** bind the same metal haptically |
| σ donor | a contacted atom that is not haptic: it gives the metal a lone pair |
| 3c2e · dative | a bridging atom's tag: **3c2e** = one electron pair over three centres (μ-H, μ-CH₃, μ-CO, B–H–B); **dative** = two 2-centre donations (μ-Cl) |
| `q_L` · `OS(M)` | ligand charge (sum of its atoms' formal charges) · metal oxidation state |
| ionic cut | M–L bonds are not counted in `b`: the donor keeps both electrons of each M–L bond |

---

## Order

```text
   0. metal / non-metal split  (METALS · `config.centers`)
   1. [T1] ligand-internal bonds
   2. [T4] M–L · M–M contacts, then the contact filters
   3.      boron cages (Wade)  ·  haptic units  ·  Hückel carbocycles
   4.      which contacts may be read as no bond (the cut) · η² partners
   5.      THE PROGRAM — orders · atom charges · h per unit · OS · cage charges · unpaired electrons
   6.      candidates → M–L orders [T8] → ranking
   7.      reading the answer: Conj · bridge tags [T7] · η^k [T6] · q_L · OS [T10]
```

### 1. [T1] ligand-internal bonds

```
bond(X,Y) ⟺ d(X,Y) < d_int(X,Y)                    58 element pairs (data/d_int.csv)
  a pair with no fitted value: d_int = 1.092 × (r_cov(X) + r_cov(Y))   (the fitted pairs' median
    ratio of d_int to the radius sum)
  never for H–H, nor when d(X,Y) > 1.8 × (r_cov(X) + r_cov(Y))
  an H bonded to two or more ligand atoms keeps only the nearest (a B–H–B bridge keeps both)
  a B–B is dropped when the two B share ≥ 2 H neighbours and neither has another B neighbour
  a 4-ring diagonal is dropped: X and Y share ≥ 2 neighbours and every shared neighbour Z sees them
    at ∠X–Z–Y ≥ 80° (bonded, X–Y–Z would be a 3-ring, and no 3-ring angle reaches 80°)
rings = `nx.cycle_basis`
```

### 2. [T4] M–L · M–M contacts

```
contact(M,X) ⟺ d(M,X) < d_bond(M,X)  AND  w(M,X) > w_veto(M,X)
  element pairs 316 (M–L) · 23 (M–M), data/d_bond.csv
  a pair with no fitted value: d_bond = (r_cov(M) + r_cov(X)) × the fitted pairs' median ratio of
    d_bond to the radius sum — 1.15 for M–L, 1.10 for M–M — and no Mayer veto
    (r_cov: Cordero 2008, low-spin for Mn · Fe · Co)
  X is any atom, metals included — M–M bonds are decided here, order 1

dropped from the M–L contacts
  agostic H          X = H with exactly one metal-like neighbour and an internal neighbour that is
                     not metal-like  (μ-H and B–H···M are kept)
  agostic carbon     X = C carrying an H nearer M with d(M,H) < AGOC = 2.0 Å, unless M would keep
                     no bond to that fragment
  saturated atom     deg(X) ≥ CAP(X), X ∉ {H, B, Al} and no internal neighbour of X is B or Al
  bound halogen      X ∈ {F, Cl, Br, I} with an internal neighbour and w(M,X) < HALW = 0.30
  bridged boron      M···B when an H on that boron is itself a contact of M and nearer to it
  no pair to give    X has no lone pair on its T1 bonds (§Charge) and d(M,X) > r_cov(M) + r_cov(X);
                     B and cage atoms are kept
  far contact        an internal neighbour Y of X touches the same metal and
                     d(M,X) > JOINTFAR · d(M,Y)   (JOINTFAR = 1.3)
```

### 3. Cages · haptic units · Hückel rings

```
cage           a fragment holding boron and a non-H atom with deg > CAP. Its vertices: B, and any
               non-H atom with a B neighbour and ≥ 2 vertex neighbours (a metal bonded to ≥ 2
               vertices counts as a vertex neighbour). Only the vertices are charged as a whole
               (Wade–Mingos, in the program); the rest of the fragment is solved as usual
haptic unit    a connected run of atoms that touch one metal and can be unsaturated
               (not H, deg < CAP), cage vertices excluded. Inside a unit, an atom none of whose
               unit bonds reads Double or Triple (③ below) leaves the unit; the rest splits into
               connected chains. A chain end whose bond to the chain reads Single leaves as a
               σ donor. Units of two metals that share atoms are one π chain (one h)
Hückel ring    an all-carbon planar ring (out-of-plane rms ≤ 0.05 Å) of atoms that can be
               unsaturated: odd rings expect one cation or anion, 4n rings two anions
```

**③ distance score** — what the geometry wants, per bond and per class:

```
score(e, c) = − |d(e) − med[k, c]| / w[k]
  k = element pair · med = the per-class distance median (data/scores4.json, 19 element pairs)
  w[k] = the narrowest per-class width (1.4826 × MAD) among Single · Double · Conj — one width for
         every class of the pair, so a class scores by how close the bond is to its median
order score  s₁ = max(score Single, score Conj) · s₂ = max(score Double, score Conj) · s₃ = score Triple
             (an order with no fitted class has no variable)
```

### 4. The cut and η² partners

A σ contact asks its donor for a lone pair. When the ligand's own geometry leaves the donor none —
a contact to one carbon of a π system, a ring C–H carbon — insisting on it would break the π bond
and push the ligand charge down by 2. So some contacts may be read as **no bond**:

```
detector   solve the program once without the lone-pair rule, every atom read as a free atom
flagged    contacted atoms left with no lone pair there, except those that have one once a
           double bond to or at P · As · S · Se · Te is read charge-separated (ylide, phosphinito)
firm       Mayer w(M,X) ≥ JOINTCUTW = 0.30   OR   d(M,X) ≤ JOINTCUTD = 0.90 × d_bond(M,X)
cuttable   flagged and not firm   (every flagged atom with `predict(firm_contacts=False)`)
```

**η² partner.** A flagged atom X outside every haptic unit takes a neighbour Y as an η² unit with the
same metal when Y is not H, can be unsaturated, the X–Y bond's best ③ class is not Single, and
`d(M,Y) < PARTNER_REACH · d_bond(M,Y)` (1.25). A cuttable X needs `∠M–X–Y < 90°` and the unit is
then like any other; a firm X takes the partner at any angle, and that unit is only **offered** — the
program may leave it off and keep X a σ donor. Y may itself be cut. Among several Y, the smallest
angle.

### 5. The program

```
variables   per bond           order 1 · 2 · 3
            per atom           one bond-order-sum level; charge = q_atom at that level (§Charge),
                               plus a carbenium level (+1, carbons of Hückel rings) and an
                               unpaired-electron level
            per haptic unit    h ∈ {0, 1}
            per cuttable atom  cut ∈ {0, 1}
            per metal          one oxidation state from os_range(element)
            per cage           one charge from its Wade candidates: closo · nido · arachno by the
                               cage edge count (3n − 6 edges = closo); the other types cost 2 λ;
                               a cage of ≤ 4 vertices prefers none. Skeletal electrons: B 2 · C 3 ·
                               other vertex v − 2 + exo · bridging or extra H 1

constraints
  free atom        |q| ≤ 1 · period-2 atoms within the octet      (a cut atom is a free atom)
  σ donor          keeps ≥ 1 lone pair; a period-2 donor b ≤ 3      (B is exempt: no pair to give)
  coordinating     σ donors and atoms of a unit that is on carry any charge: oxo −2, imido −2,
                   nitrido −3, alkylidene −2 need no exception
  haptic           a chain of ≥ 3 atoms on one metal: h = 1 · a two-atom unit, or units of two
                   metals sharing atoms: h = 1 whenever some unit bond's best ③ class is not Single
                   (an offered η² partner unit is never forced) · two atoms: h = 1 ⟺ their bond
                   is multiple
                   h = 1 ⇒ multiple bonds in the unit = its maximum matching (chain or ring ⌊k/2⌋,
                   trimethylenemethane 1); a 4n ring may have one fewer and two anions (C₄R₄²⁻ · COT²⁻)
  cut              cut + h ≤ 1 for an atom of a unit
  bent atom        a two-neighbour atom outside every unit, bent below SP_LINEAR = 150°: at most
                   one multiple bond, and a triple bond only to a terminal atom (the C of a bent
                   M–C≡N–R)
  charge           Σ q_L + Σ OS + Σ cage charges = total_charge
  unpaired         Σ unpaired = n_unpaired; on a d-block metal u ≤ min(d, 10 − d), d − u even
  CO               a two-atom C–O fragment is C≡O

objective (minimise)
    − Σ (s_o − s₁) over the chosen orders o ≥ 2
    + λ · Σ |q|                        free atoms (coordinating atoms are exempt)
    + JOINTCUT · λ per cut
    + JOINTADJ · λ per adjacent same-sign pair — not for two atoms on the same metal, unless both
      are in a haptic chain of ≥ 3
    + JOINTCHAINW · λ per charged atom off k − 2 × (multiple bonds) in a unit that is on
    + JOINTOSW · OS prior (data/os_prior.json) + JOINTSYM · |OS difference| between same-element metals
    + JOINTLIGSYM · |charge difference| between ligands with the same element graph whose bond
      lengths agree within 0.03 Å
    + JOINTCAT · λ per carbenium · JOINTRAD · λ per unpaired electron on a ligand atom
    − λ per charge a Hückel ring asks for (C₅ one anion, C₇ one cation, C₈ two anions)
```

A donor that no level lets keep a lone pair is left unconstrained and listed in
`r["joint"]["v1_skipped"]`. When the program has no solution, the free-atom charge range is dropped
for one fragment at a time (`status = relaxed_fc`), then the total charge (`q_status = q_relaxed`).
If it still has none, has more than `JOINT_MAX` variables, or scipy is missing, `predict` raises
`RuntimeError` naming the status.

### 6. Candidates · M–L orders · ranking

```
candidates   the best solution, then again with its signature forbidden, up to JOINTK = 5 — a
             signature is (metal OS, ligand and cage charges, σ-donor levels, h, cuts). Each is checked:
             every σ donor keeps a lone pair
[T8] M–L     a σ donor bonded to n metals holds max(n, −q_donor) M–L order in total: one per
order        metal, the rest to the bond the Mayer model rates highest (X-type −1 → 1,
             oxo · imido → 2, nitrido → 3, μ-O²⁻ → two singles). A haptic bond has no order
ranking      candidates within JOINTLOWQ = 0.3 of the best score: the smaller Σ|q_L| wins
             else, within JOINTTIE · λ: the better Mayer consistency (the Mayer model's score at
             the assigned orders, data/b_ml_t8forms.csv) · else the best score
```

### 7. Reading the answer

- **Conj** — a bond whose S/D alternation (a ring or path whose atoms keep their bond-order sums)
  can be flipped for at most `JOINTCONJEPS` = 2.0 of distance score.
- **[T7] bridge tags** — on the chosen orders:

  ```
  n_center(X) = n_ML(X) + (internal neighbours of X that are B · Al; 0 if X is B · Al; for
                          H · C · Si only when deg(X) ≥ VALENCE_3C[el] — σ bonds full, so the metal
                          can only share one of them)
  b_use(X)    = b(X) + non-haptic n_ML(X)
  bridge(X) ⟺ n_center ≥ 2
  3c2e(X)   ⟺ bridge AND el ∈ {H, C, Si, B} AND b_use > VALENCE_3C[el] (H 1 · C · Si 4 · B 3)
  dative(X) ⟺ bridge AND not 3c2e

  μ-H 0+2 > 1 · μ-CH₃ 3+2 > 4 · μ-CO (C≡O) 3+2 > 4 · B–H···M · B–H–B  →  3c2e
  μ-CR₂ 2+2 = 4 · μ-Cl (Cl has no entry)                                →  dative
  α-boryl C=C on one metal: deg 3 < 4, the C–B is a substituent        →  no tag
  CH₂ between B and M · cage C: deg ≥ 4, the B counts · 4+1 > 4          →  3c2e
  ```

  Output: `ml_bonds[(m,x)]["bridge"]` = None | "3c2e" | "dative"; `["type"]` is haptic > bridge >
  sigma; ligand-internal legs are in `fragment["bonds_3c2e"]`.
- **[T6] η^k** — the number of atoms of that ligand haptic to the same metal. Ferrocene's two rings
  are two ligands (two η⁵); a bridged (ansa) metallocene is one ligand, so one η¹⁰.
- **Charges** — a carbenium carbon is `+1`; cage atoms carry none and the cage's Wade charge is on
  the fragment.
- **[T10] OS** — the program's oxidation state per metal. Several metal-bearing molecules in one
  input: the split of `total_charge` between them is the program's choice, flagged
  `oxidation_is_exact = False`.
- `r["joint"]` = `{status, objective, q_status, alt_gap, alt_os, n_candidates, n_rejected, ranking,
  v1_skipped, v1_failed, far_dropped, cut, eta2_partner}`. `alt_os` is the best candidate with
  other oxidation states and `alt_gap` how much worse it scores — a small gap means a real
  alternative. `cut` lists the contacts read as no bond, `eta2_partner` the partners added.

---

## §Charge — `q_atom`, `q_L`, `OS(M)`

```
(a) per-atom formal charge on the internal bonds (the ionic cut)
        lp(X) = max(0, 4 − b)                  lone pairs
        q(X)  = v − b − 2·lp(X)
    ⇒ b ≤ 4 :  q = v + b − 8   (octet)         v + b − 2 for H
      b > 4 :  q = v − b       (hypervalent: sulfone S, perchlorate Cl → 0)
    B:  lp = max(0, 3 − b), and 0 when B holds an internal 3c2e leg   (B(OH)₃ 0 · boryl [BR₂]⁻ −1)

(a′) where the octet formula does not describe the atom
        free carbene      C, deg 2, b 2, no N/O neighbour, no M–L bond      → 0   (formula −2)
        N/O-stabilized    C, deg 2, b 2, an N or O neighbour (NHC, Fischer)  → 0   (formula −2)
        sulfoxide         S, deg 3, b 4, an O neighbour                      → 0   (formula +2)
        sulfone center    S, deg 4, b 6, two O neighbours                    → 0
        P=O · P=N         P, deg 4, b 5, an O or N neighbour                 → 0   (formula +2)
        nitrite           N, deg 2, b 4, two O neighbours                    → −1  (formula +1)
    an M-bound alkylidene (C, deg 2, b 2, no N/O) stays −2

(c) 3c2e — an internal leg to a B/Al holds no pair when the bridging atom is H or has two such
    legs (the B–H of B–H–B or κ²-BH₄, listed in `bonds_3c2e`); its order comes off `b` before (a):
    B₂H₆ is B(+1) · bridging H(−1) · terminal H(0)

(d) q_L   = sum of the formal charges of all atoms of the ligand fragment (+ a cage's Wade charge)
    OS(M) = the program's value, with Σ q_L + Σ OS = total_charge
```

---

## Flags

**`pi_suppressed`** — per fragment, the bonds written `Single` between two anionic atoms where the
distance likelihood prefers `Double` (`score(Double) − score(Single) > 0`, with the class prior of
`data/scores4.json` included; pairs with no fitted `Double`, such as As–C, are never flagged). Each
one means this fragment's charge is 2 too negative and, on a metal-bearing molecule, the metal's
oxidation state 2 too high. A **flag, not a correction**.

**`charge_balance`** — metal-free input only: `ok` is False when the emitted charges do not add up
to `total_charge` (a relaxed total charge, or a contradictory `n_unpaired`), with the atoms carrying
the unexplained charge in `sites`.

## Valence violation

```
violation(X) ⟺ b_kek(X) + n_σ(X) > CAP(X)       X a non-metal
  b_kek : sum of the emitted bonds_kekule integers · n_σ(X) : number of non-haptic M–L bonds of X
  3c2e-tagged atoms and B are excluded
```

The CSD reference labels themselves violate on about 0.4% of structures.

---

## Fitted parameters · constants

| File | What | Count |
|---|---|---|
| `data/d_int.csv` | T1 per-element-pair distance threshold | 58 |
| `data/d_bond.csv` | T4 `d_bond` · `w_veto` | 316 (M–L) + 23 (M–M) |
| `data/scores4.json` | ③ per-class distance median · width (and the class prior read by `pi_suppressed`) | 19 element pairs |
| `data/b_ml_t8forms.csv` | Mayer model of the M–L order (T8 order placement, Mayer consistency) | 57 pairs × 2 thresholds · 349 pairs one fixed class |
| `data/os_prior.json` | oxidation-state frequency per element (from CSD names) | — |

| constant | value | what it is |
|---|---|---|
| `JOINTQ` (λ) | 2.0 | charged free atom, against the distance score |
| `JOINTADJ` | 1.0 | adjacent same-sign pair, × λ |
| `JOINTCHAINW` | 1.0 | charged atom off a haptic unit's count, × λ |
| `JOINTCUT` | 0.05 | reading a contact as no bond, × λ |
| `JOINTCUTW` · `JOINTCUTD` | 0.30 · 0.90 | a contact is firm (never cut) at this Mayer, or within this × `d_bond` |
| `JOINTFIRM` | on | whether firm contacts are kept out of the cut (the default of `predict(firm_contacts=...)`) |
| `PARTNER_REACH` · η² angle | 1.25 · 90° | η² partner reach (× `d_bond`) · angle for a cuttable contact |
| `SP_LINEAR` | 150° | below this a two-neighbour atom is bent |
| `JOINTOSW` | 0.1 | OS prior weight |
| `JOINTSYM` · `JOINTLIGSYM` | 0.5 · 0.5 | per-unit OS difference (same element) · charge difference (same ligand) |
| `JOINTCAT` · `JOINTRAD` | 1.0 · 0.5 | carbenium · unpaired electron on a ligand atom, × λ |
| `JOINTFAR` | 1.3 | far-contact ratio |
| `JOINTK` · `JOINTLOWQ` · `JOINTTIE` | 5 · 0.3 · 1.0 | candidates · smaller-charge window · Mayer window (× λ) |
| `JOINTCONJEPS` | 2.0 | flip tolerance for `Conj` |
| `JOINT_MAX` · `JOINT_TIME` | 5000 · 20 s | program variables · time for one solve (best-so-far kept; candidates stop after 4 ×) |
| `HALW` · `AGOC` | 0.30 · 2.0 Å | bound-halogen Mayer floor · agostic-carbon distance (T4) |
| T1 radius cap · ratio | 1.8 · 1.092 | no internal bond beyond 1.8 × Σr_cov · `d_int` of an unfitted pair |
| 4-ring diagonal | 80° | shared-neighbour angle at or above which a T1 pair is not a bond |
| `τ_plane` | 0.05 Å | ring planarity (Hückel rings) |

Every constant is an environment variable read at import (`config.py`), so a run can be reproduced
with other values from the command line.

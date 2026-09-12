# xyz2mol-om

**From the `xyz` coordinates of a transition metal complex, derive bonds, bond orders, ligand charges, and metal oxidation states.**
It is built to handle organometallics as well, hence the `-om` in the name.

## Dependencies

| Package | Version | Used for |
|---|---|---|
| `numpy` | ≥ 1.23 | coordinates, distances |
| `networkx` | ≥ 3.0 | graphs, rings, connected components |
| `rdkit` | ≥ 2023.3 | SMILES · EHT fragment charge (`rdEHTTools`) |
| (optional) `matplotlib` | ≥ 3.5 | `draw()` — the 2D figure. Not needed for `predict` |
| (optional) `pytest`·`ruff` | — | tests, lint |

Python ≥ 3.10. Validated on Python 3.13.5 · rdkit 2025.09.6 · numpy 2.1.3 · networkx 3.4.2.

```bash
conda install -c conda-forge rdkit numpy networkx    # or pip install rdkit numpy networkx
```

## Usage

```bash
PYTHONPATH=<repo>/src python your_script.py
```

```python
from xyz2mol_om import predict

r = predict(elements, coords, total_charge=-1, wbo=wbo)
```

| Argument | Description |
|---|---|
| `elements` | list of element symbols |
| `coords` | `(n, 3)` coordinates (Å) |
| `total_charge` | total charge of the complex. Without it, oxidation states and the complex SMILES are not produced |
| `wbo` | `{(metal idx, atom idx): Mayer bond order}` — output of xtb GFN2 `--sp --wbo` |
| `n_unpaired` | unpaired electrons, `0` (default) or `1`. Pass `multiplicity − 1`; see [Limits](#-limits) |

⚠️ **You may run with `wbo=None`**, but it costs more than it looks. The M–L decision falls back to
distances alone; **internal** bond orders are essentially unchanged, but everything that touches the
metal degrades (holdout 6,793, both columns from the 2026-09-08 run — read the gap, not the levels):

| | with `wbo` | without |
|---|---|---|
| T4 M–L bond existence | .9916 | .9838 |
| T8 M–L `Double` | .7461 | .6979 |
| T5 haptic | .9778 | .9590 |
| T6 η^k | .9865 | .9737 |
| **valence-violating structures** | **1.85%** | **2.47%** |
| T3 internal `Double` | .7699 | .7696 |

⚠️ When you do pass `wbo`, fill **every** `(metal, atom)` pair. A missing pair is read as
"veto passed", not "unknown" — xtb's `wbo` file omits near-zero pairs, so build
`{(m, x): 0.0 for …}` first and overwrite with the values the file does list.

## Output

**A molecule is the top level.** The input may hold several — an IRC endpoint where the product has
separated, a salt with its counter-ion, a solvate — so the result is a list of them, each carrying
its own metals, fragments, charge and SMILES.

```python
r["total_charge"] == -1          # what you passed in, unchanged

r["molecules"] == [
  {"index": 0,
   "atoms": [0, 1, …],           # input atom indices
   "charge": -1,                 # this molecule's charge
   "charge_is_exact": True,      # False when it had to be guessed (see below)

   "metals": [
     {"index": 0, "element": "Mo", "oxidation": 6,
      "oxidation_is_exact": True,      # False = an even split, see below
      "mm_bonds": {}}],

   "fragments": [                # connected components of the internal bonds
     {"index": 0, "atoms": [1],
      "bonds_4class": {},        # {(i,j): "Single"|"Double"|"Triple"|"Conj"}
      "bonds_kekule": {},        # {(i,j): 1|2|3}
      "smiles": "[N-3:1]", "smiles_ok": True, "smiles_note": "",
      "coordinating": [1],
      "ml_bonds": {(0, 1): {"type": "sigma",   # sigma | haptic | bridge
                            "order": 3,        # None if haptic
                            "bridge": None}},  # if bridging, "3c2e" | "dative"
      "eta": {},                 # {metal: k} — counted **per ligand**, so a bridged
                                 #   (ansa) metallocene is one η¹⁰, not η⁵:η⁵
      "charge": -3,              # this fragment's charge
      "residual_charge": None,   # charge the skeleton cannot express
      "pi_suppressed": []},      # bonds written `Single` between two anionic atoms where
                                 #   the distance likelihood wanted `Double` — a **flag**,
                                 #   see ⚠️ Limits
     … ],

   "smiles": "[H][O-]->[Mo+6](<-[N-3])(<-[Cl-])(<-[Cl-])<-[Cl-]",
   "smiles_ok": True, "smiles_note": "",
   "atom_order": [...]},         # SMILES atom order, in input indices
  … ]
```

A **fragment** is a connected component of the internal bonds. Most are ligands — `ml_bonds` says
what they coordinate — but a molecule with no metal has exactly one fragment that coordinates
nothing, and that is how a free organic molecule appears.

`ml_bonds` holds exactly the M–L bonds the pipeline decided on, so it **always agrees with the
molecule's SMILES**. Contacts that T4 rejects are not in it: an agostic `C–H···M`, and a contact
to an atom whose own bonds already fill its valence. Both are real close approaches, but neither
is treated as a bond, so nothing in the output reports them. A weak σ M–L that the post-⑥ repair
drops (`SIGCUT`) is gone for the same reason — nothing downstream reports it either.

To walk the whole result without nesting loops:

```python
from xyz2mol_om import all_metals, all_fragments
all_metals(r)      # every metal record, across molecules
all_fragments(r)   # every fragment record, across molecules
```

### More than one molecule in the input

Molecules are the connected components over **all** bonds — internal, M–L and M–M. Anything
touching none of them (a free counter-ion, a departed fragment) is a molecule of its own, and each
molecule gets its own SMILES rather than one dot-joined string.

🔴 **This is not only for tidiness.** The oxidation state is `(charge − Σ q_L) / n_metals`; run
over the whole input it averages one molecule's charge into another molecule's metals. `CpTiCl₃`
alone gives Ti(IV) and `fac-[Os(CO)₃Cl₃]⁻` alone gives Os(II), but concatenated into one input
they used to come out **Ti(III) and Os(III)** — both wrong, the sum still right, every check
passing. The state is now solved **inside** each molecule.

| the input holds | what you get |
|---|---|
| one molecule | one entry in `molecules`, exact |
| one metal-bearing molecule + any number of metal-free ones | **exact** — a metal-free molecule's charge is its formal-charge sum, and the rest belongs to the metal-bearing one. This is the IRC-endpoint case |
| two or more metal-bearing molecules | the remainder is split **evenly** over all their metals and marked `oxidation_is_exact: False`, with the same warning in the molecule's `smiles_note`. It is right when the molecules are symmetric (5 of the 7 holdout structures that land here) and silently wrong otherwise, so check the flag — or pass one molecule at a time |

### SMILES format

| | Convention |
|---|---|
| Fragment SMILES | atom map `[X:n]` on the coordinating atoms · our orders and charges are pinned as they are |
| Molecule SMILES | M–L are **all dative arrows** (`->`) · metal formal charge = **oxidation state** |
| Order | collapsed in the molecule SMILES — the real value is `ml_bonds[(m,x)]["order"]` |
| Validation | `smiles_ok` = whether the round-trip check (order · charge · H · multiset) passed · the reason for failure is in `smiles_note` |

### Complex reassembly

```python
from xyz2mol_om import predict, assemble_complex

mol, atom_map = assemble_complex(r)              # atom_map: {input index -> mol index}
mol, _ = assemble_complex(r, ml_dative=False)    # M–L as integer orders (haptic stays dative)
```

To assemble ligand by ligand yourself — join the metal (formal charge = `oxidation`) + ligand SMILES +
`ml_bonds` + `mm_bonds`. The atom correspondence is this one line:

```python
input_idx = sorted(lg["coordinating"])[at.GetAtomMapNum() - 1]
```

⚠️ The map `[X:n]` in a ligand SMILES is **not the input index** — it is the n-th entry of the sorted `coordinating` list.
⚠️ Implicit hydrogens are not used — read with `sanitize=False` and sanitize with KEKULIZE and SETAROMATICITY removed.
⚠️ a molecule's `smiles` collapses the M–L orders (the real value is in `ml_bonds[…]["order"]`). Atom correspondence: the molecule's `atom_order`.

## Examples — `examples/`

Five real CSD structures. Each example comes as four files — `<name>.xyz` (coordinates) · `<name>.wbo.json`
(total charge + Mayer bond orders) · `<name>.result.json` (**the full pipeline output**) · `<name>.png` (the figure `draw()` produces).

```bash
python examples/run_examples.py            # runs all 5 and rewrites <name>.result.json
python examples/run_examples.py 02         # only those with "02" in the name
python examples/run_examples.py --no-wbo   # without wbo (distance fallback)
python examples/draw_examples.py           # redraw the PNGs
```

| # | File | Real system | What it shows | Output | Figure |
|---|---|---|---|---|---|
| ① | `01_dative_os_carbonyl` | `fac-[Os(CO)₃Cl₃]⁻` | σ-dative only · internal `C≡O` | [json](examples/01_dative_os_carbonyl.result.json) | [png](examples/01_dative_os_carbonyl.png) |
| ② | `02_haptic_cp_ticl3` | `CpTiCl₃` | η⁵ haptic | [json](examples/02_haptic_cp_ticl3.result.json) | [png](examples/02_haptic_cp_ticl3.png) |
| ③ | `03_bridge_ag2cl4` | `[Ag₂Cl₄]²⁻` | μ-Cl bridge (`bridge:dative`) · two metals | [json](examples/03_bridge_ag2cl4.result.json) | [png](examples/03_bridge_ag2cl4.png) |
| ④ | `04_3c2e_gallium_bh4` | `Me₂Ga(BH₄)` | 3c2e bridging H · `B` as a ligand atom | [json](examples/04_3c2e_gallium_bh4.result.json) | [png](examples/04_3c2e_gallium_bh4.png) |
| ⑤ | `05_mm_quadruple_re2cl8` | `[Re₂Cl₈]²⁻` | M–M bond | [json](examples/05_mm_quadruple_re2cl8.result.json) | [png](examples/05_mm_quadruple_re2cl8.png) |

To save a result yourself use `save_json(r, path)`, and to read it back `load_json(path)`
(bond keys `(i, j)` are stored as `"i,j"` and converted back on read).

### Drawing a result — `draw()`

```python
from xyz2mol_om import predict, read_xyz, draw

el, xyz = read_xyz("complex.xyz")          # or your own lists
r = predict(el, xyz, total_charge=-2, wbo=wbo)
draw(el, xyz, r, "complex.png", title="[Re2Cl8]2-")
```

It takes **the same two inputs you gave `predict`, plus what `predict` returned** — the geometry is
what it draws, and the result is what it labels.

| argument | | what it is |
|---|---|---|
| `elements` | required | the element list passed to `predict` |
| `coords` | required | the `(n, 3)` coordinates passed to `predict` — **the figure is this geometry**, not a 2D layout |
| `result` | required | the dict `predict` returned (it reads `molecules` → `metals` · `fragments` → `bonds_kekule` · `ml_bonds` · `eta` · `charge`) |
| `out` | required | where to write; the extension picks the format (`.png`, `.pdf`, `.svg`) |
| `title` | `""` | first title line |
| `subtitle` | auto | second line; by default the per-ligand charges and η, e.g. `q0=-1 η5 · q1=-1` |
| `highlight` | `()` | internal bonds `{(i, j), …}` to draw thick red — for pointing at a disputed bond |
| `projection` | auto | a projection returned by an earlier call, to put every atom in the same place |

It returns the projection it used. Pass that back as `projection=` for a second figure and the two
become comparable atom by atom — which is the only way to read a reference beside a prediction:

```python
proj = draw(el, xyz, reference, "ref.png", title="reference")
draw(el, xyz, r, "pred.png", title="prediction", projection=proj, highlight={(2, 3)})
```

**What you see.** The projection is the least cluttered view of the real geometry — an RDKit 2D
layout collapses haptic rings and chelates onto themselves, which is why this is drawn from
coordinates instead. Internal bonds get 1/2/3 lines from `bonds_kekule`; M–L bonds are arrows
(**σ** solid black · **haptic** green dotted · **3c2e bridge** brown dashed — a `dative` bridge is
an ordinary donor bond and is drawn like σ); M–M bonds are purple, one line per order; the metal is
a purple circle carrying its oxidation state; every other atom, H included, is labelled with its
formal charge when non-zero, and a neutral carbon is just a dot.

Which H is drawn follows the skeletal convention — **an H on carbon is implied, an H on anything
else is written**. So N–H · O–H · S–H are visible (they are what makes an amido `Ar–N(H)⁻` readable
as `−1` rather than as a nitrogen missing a bond), and so is an H on a metal (hydrido, 3c2e bridge)
and any H you pass in `highlight`. When several molecules are in the result they are translated
apart so they do not overlap — each one rigidly, so the geometry inside a molecule is untouched,
but the distance *between* molecules is no longer to scale.

⚠️ **`matplotlib` is required for this function only** — it is not a dependency of the package, and
`predict` does not need it. The five figures in `examples/` are made by `examples/draw_examples.py`,
which is nothing more than a loop over this call.

## Package layout

```
xyz2mol_om/
├── api.py        predict() — the only orchestrator: it calls the stages below in order
├── config.py     every constant, threshold and switch, with the measurement behind it
├── data/         the fitted tables (per-element-pair distances, thresholds, likelihood)
├── geometry/     coordinates in, connectivity out — no chemistry
├── rules/        the decision rules: conjugation · likelihood · valence solvers · M–L order · pipeline
├── charge/       formal charge, fragment charge, Kekulé conversion, extended-Hückel charge
└── output/       SMILES · RDKit mol · JSON · figure
```

Everything a caller normally needs is re-exported at the top: `from xyz2mol_om import predict,
read_xyz, draw, save_json`. The subpackages are there for reading the code, and each one's
`__init__` says what it is for.

## Tests

`pytest` — 81 tests, no workspace and no network. Every one is a small hard-coded structure with
its Mayer bond orders pinned as constants, so the suite runs on a bare install.

| file | what it pins |
|---|---|
| `test_api_smoke.py` | one `[Mo(≡N)(OH)Cl₃]⁻` through every stage — T1 · T4 · T8 `Mo≡N` · `q_L` · `OS(Mo)=+6` · ⑥ · ligand SMILES round trip |
| `test_molecules.py` | a disconnected input splits into molecules, and `OS` is solved **inside** one |
| `test_radical.py` | `n_unpaired=1` — the electron on a ligand, on the metal, and the refusals |
| `test_ml_budget_and_eta.py` | agostic and haptic spend nothing in the ④ budget · η^k is per ligand |
| `test_bridging_carbonyl.py` | μ-CO comes out `3c2e` with the `C≡O` intact |
| `test_complex_smiles_and_bridge.py` | the complex SMILES, the bridge tags, the 3c2e budget exclusion |
| `test_r7_haptic_ring.py` | R7 fires — the S of an η⁵-thienyl turns haptic |
| `test_eta2_carries_the_pi.py` | an η² is written across a `Double`, never a `Single` |
| `test_pi_shift.py` | the post-⑥ repairs — the π moves where it cancels two charges, and the dianionic-chelate and peroxide exclusions hold |
| `test_agostic_carbon.py` | the carbon side of an agostic `C–H···M` is dropped too, unless that orphans the fragment |
| `test_boron_sextet.py` | trivalent boron is neutral (sextet), four-coordinate boron unchanged |
| `test_rule_a_headroom.py` | rule A's π headroom counts the σ M–L bonds, and NHC · σ-aryl · haptic ring atoms are untouched |
| `test_pi_suppressed.py` | the `pi_suppressed` report: when it fires, when it must stay silent, that it reads the raw likelihood margin, and that it survives the JSON round trip as tuples |
| `test_invariants.py` | geometry-free unit tests on the decision functions themselves |
| `test_assemble.py` · `test_serialize.py` · `test_draw.py` | reassembly, `save_json`/`load_json` bond keys, and that `draw()` renders every bond kind it claims to |

```bash
pip install -e ".[dev]" && pytest -q
```

## Performance

holdout **6,793 structures** (not used in the fit) · measured **2026-09-13** · reference labels:
CSD `bond_type`, tmQMg-L `q_ligand`, and the roman numeral in the CSD `chemical_name` for the
oxidation state.

⚠️ Fit and evaluation both use CSD experimental structures **relaxed with GFN2-xTB**.
Coordinates from another source (raw CSD, DFT, a force field) are off-distribution.

| Task | Metric | Value | Pool | Baseline |
|---|---|---|---|---|
| T1 ligand internal bond existence | F1 | **0.9998** | 378,303 bonds | all bonded .7306 |
| T2 conjugation call | F1 | **0.9618** | 87,581 bonds | — |
| T3 internal order `Single`/`Double`/`Triple`/`Conj` | F1 | **.9906 / .7753 / .9771 / .9618** | 378,212 bonds | all `Single` .9097 / 0 / 0 |
| T4 M–L·M–M bond existence | F1 | **0.9916** | 56,510 bonds | all bonded .5276 |
| T5 haptic call | F1 | **0.9796** | 15,331 M–L bonds | all haptic .6766 |
| T6 η^k (exact match per ligand) | accuracy | **0.9865** | 4,224 ligands | all `k=0` .8704 |
| T8 M–L order `Single`/`Double`/`Triple` | F1 | **.9932 / .7473 / .7228** | 39,523 bonds | — |
| T10 ligand charge `Σq_L` (exact match per structure) | accuracy | **0.8596** | 1,161 structures | reference-order 0.8528 |
| T10 metal oxidation state `OS` (exact match per structure) | accuracy | **0.8971** | 2,779 structures | reference-order 0.8698 |

The pool differs per task because the references do: `bond_type` covers every structure,
tmQMg-L charges 23% of them, and a roman numeral in the CSD name 41%. The baseline column is the
**trivial** prediction for that task, except the two `reference-order` entries — see below.

⚠️ **`reference-order` is not an upper bound.** It is what the same charge rule produces when the
**reference** bond orders are fed to it (CSD labels through `charge.kekulize`), so it measures how
much of the gap is our Lewis notation against tmQMg-L's rather than our bond orders. Its pool is
slightly smaller (1,155 / 2,635 structures — kekulization of the reference fails on a few), so it
must not be read against the column to its left. **On the common pool** the pipeline is now
**above** it: `Σq_L` **0.8563 vs 0.8528** and `OS` **0.8774 vs 0.8725** (the common-pool pair is
from the 2026-09-08 run; both of ours have risen since). What is left of the gap
to a perfect score is notation, not order prediction.

### Against other tools

Same pool, same references, same metrics, and **a tool's failure is scored as a wrong answer**
rather than dropped. `xyz2mol_tm` runs live; it is given 60 s per structure, beyond which the
structure counts as a failure.

| | ours | `xyz2mol` | `xyz2mol_tm` | OpenBabel |
|---|---|---|---|---|
| **structures it produced an answer for** | **6,793** | 6,156 | 5,676 | **6,793** |
| T1 internal bond existence | **.9998** | .9672 | .8922 | .9928 |
| T4 M–L·M–M bond existence | **.9916** | — | .8990 | .7579 |
| T3 `Single` | **.9906** | .9691 | .9811 | .9434 |
| T3 `Double` | **.7753** | .3945 | .5693 | .3515 |
| T3 `Triple` | **.9771** | .9586 | .9770 | .1217 |
| T3 `Conj` | **.9618** | .9090 | .9323 | .7922 |
| T8 M–L `Single` | **.9932** | — | — | .9771 |
| T8 M–L `Double` | **.7473** | — | — | .0658 |
| T8 M–L `Triple` | **.7228** | — | — | .0106 |
| T5 haptic | **.9796** | — | — | — |
| T10 `Σq_L` (1,161 structures) | **.8596** | .3764 | .8071 | .1843 |
| T10 `OS` (2,779 structures) | **.8971** | — | — | — |

`—` is a task the tool cannot answer at all: `xyz2mol` strips the metal and solves the fragments,
so no M–L task; `xyz2mol_tm` gives M–L **connectivity** but no order, so no T8; OpenBabel has no
notion of haptic, so no T5 or T6.

Failures are most of what separates `xyz2mol_tm`'s T1 from ours. **On the 5,676 structures it does
solve**, its T1 rises to .9793 and its T4 to .9667 — but `Double` does not move (.5693 against our
**.7831** on that pool), and `Σq_L` reads .8120 against our **.8588**. The gap on bond order is not
a coverage artifact. ⚠️ Our two figures on that sub-pool are from the 2026-09-08 run and have
risen since; the other tools' numbers are unaffected by our changes.

### Valence violations — chemical validity of the output

`b_int(X) + b_ML(X) > CAP(X)` for a non-metal X (Kekulé count · 3c2e and B excluded).

`b_ML` is what the ④ valence constraint spends: **1.0 per non-haptic M–L bond** (a haptic bond
spends 0, and an atom in a 3c2e bridge spends 1.0 in total however many M–L bonds it has).

| Pool | Violating structures | Reference-label baseline |
|---|---|---|
| holdout 6,793 | **1.25%** | **0.4%** |

⚠️ The baseline is not 0 — the CSD reference labels themselves violate on about 0.4%
(hypervalency · where the ionic/covalent cut is drawn · CSD notation conventions), so the figure
has to be read against that.

**Violation rate against other tools** — same pools as the task table above, and the same
definition: a violating atom is a non-metal with `b_int + b_ML > CAP` (`B` excluded, and for our
own rows the 3c2e atoms too). Our `b_int` is the ⑥ Kekulé integer; a tool that emits aromatic
bonds is counted at its own 1.5.

| Pool | | ours | `xyz2mol` | `xyz2mol_tm` | OpenBabel |
|---|---|---|---|---|---|
| **holdout** 6,793 | `b_int` only | **1.25%** | 10.60% | 7.35% | 3.96% |
| | `b_int`+`b_ML` | **1.85%** | 10.60% | 37.63% | 3.96% |
| **TOOL** 5,207 (all 3 external tools succeeded) | `b_int` only | **1.50%** | 11.14% | 9.03% | 3.76% |
| | `b_int`+`b_ML` | **2.19%** | 11.14% | 45.65% | 3.76% |
| **X2M_TM** 5,676 (xyz2mol_tm succeeded) | `b_int` only | **1.41%** | 10.50% | 8.79% | 3.54% |
| | `b_int`+`b_ML` | **2.11%** | 10.50% | 45.03% | 3.54% |

⚠️ This breakdown is from the **2026-09-08** run and was not re-measured. Our current holdout
`b_int`+`b_ML` rate is **1.25%** (the table above) — the σ M–L repairs remove M–L bonds that were
spending an atom's last valence unit, which is exactly what this row counts.

The `b_int`-only row is the fair comparison — every tool produces internal bond orders, and we are
lowest in all three pools. ⚠️ **The `b_int`+`b_ML` row must not be read across tools**:
`xyz2mol_tm` emits no M–L order, so every M–L reads as one dative unit and an η⁵-Cp over-valences
five carbons at once; `xyz2mol` produces no M–L bond at all, so its two rows are identical; and
OpenBabel's flat figure comes from missing a third of the M–L bonds (T4 .7579), not from placing
them well.

## ⚠️ Limits

- **One unpaired electron, and only with `n_unpaired=1`.** Diradicals raise. Without it a radical
  still comes out as the nearest closed-shell answer **with no error** — `CH₃•` reads as `CH₃⁻`,
  and beside a metal that wrong `−1` is cancelled by the metal's `+1`, so the total charge stays
  right while the oxidation state does not. The electron goes to the one negatively charged atom
  outside the metal's own molecule **that can hold it** (`v − b ≥ 1`, so a borate's structural
  `−1` is not a candidate); with none it is on the metal and nothing changes, with several it is
  refused and `r["radical"]["note"]` says so.
- **The M–M order is a placeholder, not a prediction.** Whether two metals are bonded *is*
  predicted (`mm_bonds`, by the same distance + Mayer rule as M–L), but the order in that dict is
  the constant `1` — do not read it as "single bond". The `[Re₂Cl₈]²⁻` of example ⑤ is a
  quadruple bond and still comes out as `1`. An order model was fitted and measured on 4,027
  homonuclear M–M bonds (refcode 5-fold CV: distance .9305 · Mayer .9295 · all-`Single` baseline
  .9071) and is **not shipped** — the gain over the trivial baseline is small and the sample is
  thin where it matters (`Double` 147 · `Triple` 96 · `Quadruple` 131).
- **A suppressed π bond costs the metal `+2`, and what is left of it is reported rather than
  fixed.** When a weak M–X contact is taken as a σ bond it spends that atom's last valence unit,
  ④ then has no headroom to raise the neighbouring π bond, and ⑥ writes it `Single` with a lone
  pair on each end — so the fragment charge comes out **2 too negative** and the metal's oxidation
  state **2 too high**. The pipeline now repairs the clearest form of this: where dropping one σ
  M–L lets the two charges cancel, it is dropped and the structure re-solved (`SIGCUT`, see
  [docs/PIPELINE.md](docs/PIPELINE.md) §T3 post-⑥). What that rule declines still lands here — an
  M–L too strong to be an artefact (Mayer ≥ 0.40, unless the bond length says otherwise), a pair
  whose two ends both coordinate the metal, and peroxide. Every remaining bond is listed in that
  fragment's `pi_suppressed`:

  ```python
  from xyz2mol_om import all_fragments
  if any(fr["pi_suppressed"] for fr in all_fragments(r)):
      ...   # this structure's ligand charges and metal oxidation state are suspect
  ```

  On **Gold-DIGR 21,196 reaction endpoints** (out-of-sample — DFT geometries, not the CSD pool
  above; reference = that dataset's own mapped `rxn_smiles` metal formal charge, same ionic
  convention) it fires on **660 (3.11%)** — measured 2026-09-08, before `SIGCUT`, so the current
  rate is lower. Of those, **57.9%** disagree with the reference metal
  oxidation state against a **9.9%** base rate where it does not fire, and **96% of the
  disagreements are exactly `+2`**. It catches **57.6%** of the endpoints that are off by exactly
  `+2`. ⚠️ **A per-element oxidation-state range check sees much less of this** — only **72** of
  the 660 put the metal outside a physical range at all, so the two screens do not replace each
  other.
- **3c2e and clusters** are outside the two-center formalism — a ligand with a bridging H is **deliberately** rejected by the SMILES round-trip check, and the fragment charge of a carborane cage uses the EHT value.

Every decision rule, with its thresholds, is in [docs/PIPELINE.md](docs/PIPELINE.md).

## License · Provenance

**MIT** ([LICENSE](LICENSE)).

The reference labels used for the fit are CSD (Cambridge Structural Database) bond labels and tmQMg-L ligand charges.
**The source data is not in this repository** — what ships here is the fit result (thresholds · likelihood parameters)
and the five CSD-derived structures in `examples/`.

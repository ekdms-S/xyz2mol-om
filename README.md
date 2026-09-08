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

⚠️ **You may run with `wbo=None`**, but it costs more than it looks. The M–L decision falls back to
distances alone; **internal** bond orders are essentially unchanged, but everything that touches the
metal degrades and the output is far more often chemically impossible (holdout 6,793):

| | with `wbo` | without |
|---|---|---|
| T4 M–L bond existence | .9905 | .9764 |
| T8 M–L `Double` | .7461 | .6979 |
| T5 haptic | .9778 | .9590 |
| T6 η^k | .9865 | .9737 |
| **valence-violating structures** | **3.00%** | **8.38%** |
| T3 internal `Double` | .7699 | .7696 |

⚠️ When you do pass `wbo`, fill **every** `(metal, atom)` pair. A missing pair is read as
"veto passed", not "unknown" — xtb's `wbo` file omits near-zero pairs, so build
`{(m, x): 0.0 for …}` first and overwrite with the values the file does list.

## Output

```python
r["metals"]  == [{"index": 0, "element": "Mo", "oxidation": 6, "mm_bonds": {}}]

r["ligands"] == [
  {"index": 0, "atoms": [1],
   "bonds_4class": {},         # {(i,j): "Single"|"Double"|"Triple"|"Conj"}
   "bonds_kekule": {},         # {(i,j): 1|2|3}
   "smiles": "[N-3:1]", "smiles_ok": True, "smiles_note": "",
   "coordinating": [1],
   "ml_bonds": {(0, 1): {"type": "sigma",   # sigma | haptic | bridge
                         "order": 3,        # None if haptic
                         "bridge": None}},  # if bridging, "3c2e" | "dative"
   "eta": {},                  # {metal: k}
   "charge": -3,               # ligand charge q_L
   "residual_charge": None},   # residual charge not expressible by the skeleton
  … ]

r["complex_smiles"] == "[H][O-]->[Mo+6](<-[N-3])(<-[Cl-])(<-[Cl-])<-[Cl-]"
```

### SMILES format

| | Convention |
|---|---|
| Ligand SMILES | atom map `[X:n]` on the coordinating atoms · our orders and charges are pinned as they are |
| Complex SMILES | M–L are **all dative arrows** (`->`) · metal formal charge = **oxidation state** |
| Order | collapsed in the complex SMILES — the real value is `ml_bonds[(m,x)]["order"]` |
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
⚠️ `complex_smiles` collapses the M–L orders (the real value is in `ml_bonds[…]["order"]`). Atom correspondence: `complex_atom_order`.

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
| `result` | required | the dict `predict` returned (`metals` · `ligands` · `bonds_kekule` · `ml_bonds` · `eta` · `charge`) |
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
(**σ** solid black · **haptic** green dotted · **bridge** orange dashed); M–M bonds are purple,
one line per order; the metal is a purple circle carrying its oxidation state; a non-metal is
labelled with its formal charge when non-zero, and a neutral carbon is just a dot. Terminal H is
hidden, but an H on a metal (hydrido, 3c2e bridge) is kept.

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

## Performance

holdout **6,793 structures** (not used in the fit) · reference labels: CSD `bond_type`,
tmQMg-L `q_ligand`, and the roman numeral in the CSD `chemical_name` for the oxidation state.

⚠️ Fit and evaluation both use CSD experimental structures **relaxed with GFN2-xTB**.
Coordinates from another source (raw CSD, DFT, a force field) are off-distribution.

| Task | Metric | Value | Pool | Baseline |
|---|---|---|---|---|
| T1 ligand internal bond existence | F1 | **0.9998** | 378,303 bonds | all bonded .7306 |
| T2 conjugation call | F1 | **0.9615** | 87,581 bonds | — |
| T3 internal order `Single`/`Double`/`Triple`/`Conj` | F1 | **.9904 / .7699 / .9769 / .9615** | 378,212 bonds | all `Single` .9097 / 0 / 0 |
| T4 M–L·M–M bond existence | F1 | **0.9905** | 56,510 bonds | all bonded .5276 |
| T5 haptic call | F1 | **0.9778** | 15,331 M–L bonds | all haptic .6766 |
| T6 η^k (exact match per ligand) | accuracy | **0.9865** | 4,221 ligands | all `k=0` .8704 |
| T8 M–L order `Single`/`Double`/`Triple` | F1 | **.9932 / .7461 / .7228** | 39,540 bonds | — |
| T10 ligand charge `Σq_L` (exact match per structure) | accuracy | **0.8536** | 1,161 structures | reference-order 0.8528 |
| T10 metal oxidation state `OS` (exact match per structure) | accuracy | **0.8845** | 2,779 structures | reference-order 0.8698 |

The pool differs per task because the references do: `bond_type` covers every structure,
tmQMg-L charges 23% of them, and a roman numeral in the CSD name 41%. The baseline column is the
**trivial** prediction for that task, except the two `reference-order` entries — see below.

⚠️ **`reference-order` is not an upper bound.** It is what the same charge rule produces when the
**reference** bond orders are fed to it (CSD labels through `charge.kekulize`), so it measures how
much of the gap is our Lewis notation against tmQMg-L's rather than our bond orders. Its pool is
slightly smaller (1,155 / 2,635 structures — kekulization of the reference fails on a few), so it
must not be read against the column to its left. **On the common pool** the pipeline is now
**above** it: `Σq_L` **0.8563 vs 0.8528** and `OS` **0.8774 vs 0.8725**. What is left of the gap
to a perfect score is notation, not order prediction.

### Valence violations — chemical validity of the output

`b_int(X) + b_ML(X) > CAP(X)` for a non-metal X (Kekulé count · 3c2e and B excluded).

`b_ML` is what the ④ valence constraint spends: **1.0 per non-haptic M–L bond** (a haptic bond
spends 0, and an atom in a 3c2e bridge spends 1.0 in total however many M–L bonds it has).

| Pool | Violating structures | Reference-label baseline |
|---|---|---|
| holdout 6,793 | **3.00%** | **0.4%** |

⚠️ The baseline is not 0 — the CSD reference labels themselves violate on about 0.4%
(hypervalency · where the ionic/covalent cut is drawn · CSD notation conventions), so the figure
has to be read against that.

**Against other tools** (holdout; each tool appears only in a pool where it succeeds on every
structure — `TOOL` = all 3 external tools succeeded, `X2M_TM` = xyz2mol_tm succeeded):

| Pool · n | Tool | Viol. atoms (`b_int`) | Viol. structures (`b_int`) | (`b_int`+`b_ML`) |
|---|---|---|---|---|
| **HOLDOUT** 6,456 | **xyz2mol-om** | **0.01%** | **0.36%** | 3.55% |
| **TOOL** 4,930 | **xyz2mol-om** | **0.01%** | **0.43%** | 3.98% |
| | xyz2mol | 0.25% | 7.48% | — |
| | xyz2mol_tm | 0.18% | 4.97% | 43.79% |
| | OpenBabel | 0.06% | 1.52% | 1.66% |
| **X2M_TM** 5,479 | **xyz2mol-om** | **0.01%** | **0.40%** | 3.83% |
| | xyz2mol_tm | 0.17% | 4.87% | 43.49% |

⚠️ **Our own rows here come from an earlier revision of the pipeline** and have not been
re-measured, because the run drives the three external tools live; the external rows are
unaffected. On the whole holdout our figure has moved by less than 0.1%p since, so read these rows
as the comparison they are for — the gap between tools, not our current absolute value (that is in
the table above).

The `b_int`-only columns are the fair comparison (every tool can produce that) — we are lowest
in all three pools. ⚠️ **Do not compare the `b_int`+`b_ML` column across tools**: xyz2mol_tm
emits no M–L order, so it must be read as all-dative and every η⁵-Cp over-valences five
carbons at once (81% of its violations are at haptic sites); OpenBabel's low figure comes from
missing a third of the M–L bonds (T4 recall 0.66), not from getting them right.

## ⚠️ Limits

- **Radicals are not supported** — there is no way to write an unpaired electron, so the nearest closed-shell answer comes out **without an error**.
- **M–M orders are not produced** — only bond existence is given and the order is left at `1` (the `[Re₂Cl₈]²⁻` of example ⑤ is in fact a quadruple bond).
- **3c2e and clusters** are outside the two-center formalism — a ligand with a bridging H is **deliberately** rejected by the SMILES round-trip check, and the fragment charge of a carborane cage uses the EHT value.

Every decision rule, with its thresholds, is in [docs/PIPELINE.md](docs/PIPELINE.md).

## License · Provenance

**MIT** ([LICENSE](LICENSE)).

The reference labels used for the fit are CSD (Cambridge Structural Database) bond labels and tmQMg-L ligand charges.
**The source data is not in this repository** — what ships here is the fit result (thresholds · likelihood parameters)
and the five CSD-derived structures in `examples/`.

# xyz2mol-om

**From the `xyz` coordinates of a transition metal complex, derive bonds, bond orders, ligand charges, and metal oxidation states.**
It is built to handle organometallics as well, hence the `-om` in the name.

## Dependencies

| Package | Version | Used for |
|---|---|---|
| `numpy` | ≥ 1.23 | coordinates, distances |
| `networkx` | ≥ 3.0 | graphs, rings, connected components |
| `rdkit` | ≥ 2023.3 | SMILES · EHT fragment charge (`rdEHTTools`) |
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

⚠️ **You may run with `wbo=None`** — the M–L decision falls back to distances and **performance
drops slightly** (M–L `Double` F1 0.73 → 0.70 · bond existence F1 0.973 → 0.964 · internal bonds nearly unchanged).

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
(total charge + Mayer bond orders) · `<name>.result.json` (**the full pipeline output**) · `<name>.png` (Kekulé 2D graph).

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

## Performance

holdout **6,456 structures** (not used in the fit) · reference labels: CSD `bond_type` + tmQMg-L `q_ligand`.

| Task | Metric | Value | Trivial baseline |
|---|---|---|---|
| T1 ligand internal bond existence | F1 | **0.9998** | all bonded .7306 |
| T2 conjugation call | F1 | **0.9583** | — |
| T3 internal order `Single`/`Double`/`Triple`/`Conj` | F1 | **.9894 / .7187 / .9826 / .9583** | all `Single` .9097 / 0 / 0 |
| T4 M–L·M–M bond existence | F1 | **0.9904** | all bonded .5276 |
| T5 haptic call | F1 | **0.9768** | all haptic .6766 |
| T6 η^k (exact match per ligand) | accuracy | **0.9858** | all `k=0` .8704 |
| T8 M–L order `Single`/`Double`/`Triple` | F1 | **.9931 / .7398 / .6336** | — |
| T10 ligand charge `Σq_L` (exact match per structure) | accuracy | **0.8338** | ceiling 83.4% |
| T10 metal oxidation state `OS` (exact match per structure) | accuracy | **0.8464** | ceiling 85.6% |

### Valence violations — chemical validity of the output

| Evaluation | Pool | Violating structures | Reference-label baseline (same count) |
|---|---|---|---|
| holdout | 6,456 structures | **164 = 2.54%** | 44 = 0.68% |
| train CV | 26,075 structures | **707 = 2.71%** | 103 = 0.40% |

## ⚠️ Limits

- **Radicals are not supported** — there is no way to write an unpaired electron, so the nearest closed-shell answer comes out **without an error**.
- **M–M orders are not produced** — only bond existence is given and the order is left at `1` (the `[Re₂Cl₈]²⁻` of example ⑤ is in fact a quadruple bond).
- **3c2e and clusters** are outside the two-center formalism — a ligand with a bridging H is **deliberately** rejected by the SMILES round-trip check, and the fragment charge of a carborane cage uses the EHT value.

The decision rules, their derivation, and the measurement history live outside the library —
`ognm-bh-workspace/docs/backlog/tm-bond-remaining.md` (design) ·
`docs/analysis/2026-09-03-t3-tuning-history.md` (adoption history) · `docs/PIPELINE.md` in this repository.

## License · Provenance

**MIT** ([LICENSE](LICENSE)).

The reference labels used for the fit are CSD (Cambridge Structural Database) bond labels and tmQMg-L ligand charges.
**The source data is not in this repository** — what ships here is the fit result (thresholds · likelihood parameters)
and the five CSD-derived structures in `examples/`.

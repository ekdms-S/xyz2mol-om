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

⚠️ **You may run with `wbo=None`.** The M–L decision falls back to distances alone; **internal**
bond orders are essentially unchanged, but everything that touches the metal degrades
(holdout 6,793):

| | with `wbo` | without |
|---|---|---|
| T4 M–L bond existence | .9915 | .9833 |
| T8 M–L `Double` | .7556 | .7067 |
| T5 haptic | .9800 | .9603 |
| T6 η^k | .9865 | .9725 |
| T3 internal `Double` | .7667 | .7659 |
| **valence-violating structures** | **0.35%** | **0.35%** |

⚠️ When you do pass `wbo`, fill **every** `(metal, atom)` pair. A missing pair is read as
"veto passed", not "unknown" — xtb's `wbo` file omits near-zero pairs, so build
`{(m, x): 0.0 for …}` first and overwrite with the values the file does list.

Metals are the elements in `xyz2mol_om.METALS` — Ti Zr Hf Nb Ta V La Sc Y Ce Cr Mo W Mn Re Fe Ru
Os Co Rh Ir Ni Pd Pt Cu Ag Au Zn Al Ga In Sn Pb Mg; `B` is a ligand atom.

## Output

**A molecule is the top level.** The input may hold several — an IRC endpoint where the product has
separated, a salt with its counter-ion, a solvate — so the result is a list of them, each carrying
its own metals, fragments, charge and SMILES.

```python
r["total_charge"] == -1          # what you passed in, unchanged
r["radical"]                     # {n_unpaired, atom, site, sign, note} — see ⚠️ Limits
r["charge_balance"]              # {shortfall, ok, sites, note} — metal-free input only: `ok` is
                                 #   False when the emitted charges do not sum to total_charge

r["molecules"] == [
  {"index": 0,
   "atoms": [0, 1, …],           # input atom indices
   "charge": -1,                 # this molecule's charge
   "charge_is_exact": True,      # False (and charge None) on a metal-bearing molecule without
                                 #   total_charge, or when several metal-bearing molecules share it

   "metals": [
     {"index": 0, "element": "Mo", "oxidation": 6,
      "oxidation_is_exact": True,      # False = split across several metal-bearing molecules
                                       #   · None when oxidation is None
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
      "bonds_3c2e": [],          # [(i,j)] — ligand-internal legs of a 3c2e bridge; see below
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
molecule's SMILES**. Contacts T4 rejects (docs/PIPELINE.md, DAG 4) and σ M–L bonds dropped by the
post-⑥ repair are not in it and are not reported anywhere.

To walk the whole result without nesting loops:

```python
from xyz2mol_om import all_metals, all_fragments
all_metals(r)      # every metal record, across molecules
all_fragments(r)   # every fragment record, across molecules
```

### Where a 3c2e bridge is reported

A bridge has two **legs**, and which field a leg lands in depends only on whether it touches a
centre:

| leg | reported in |
|---|---|
| leg to a centre | `ml_bonds[(m,x)]["bridge"] == "3c2e"` |
| ligand-internal leg | **`bonds_3c2e`** on the fragment |

```
μ-H · μ-CO · μ-CH₃   2 M–L legs                  ml_bonds ×2 · bonds_3c2e []
κ²-BH₄  B–H···M      1 M–L leg + the B–H         ml_bonds ×1 · bonds_3c2e [(B,H)]   ← ex. 04
B–H–B   diborane     2 internal legs, no metal   ml_bonds []  · bonds_3c2e ×2       ← ex. 06
```

🔴 **No leg carries an electron pair of its own** — the bridging atom holds it. `bonds_kekule`
prices an entry of `bonds_3c2e` at 1 so the skeleton draws, but an electron ledger must treat it
the way it treats an M–L leg: nothing on the edge.

⚠️ Such a fragment cannot be rebuilt from two-centre bonds — `assemble_complex` refuses it, and
`smiles_ok` is `False` when one atom holds the pair for two legs at once (`B₂H₆`), on the
**molecule and the fragment alike**. The SMILES string is still written, so use `bonds_kekule`
and `bonds_3c2e` instead of parsing it. Where the bridge reaches a metal (`μ-H`, `κ²-BH₄`) the
dative arrow expresses it and the round trip passes.

### More than one molecule in the input

Molecules are the connected components over **all** bonds — internal, M–L and M–M. Anything
touching none of them (a free counter-ion, a departed fragment) is a molecule of its own, and each
molecule gets its own SMILES.

🔴 The oxidation state `(charge − Σ q_L) / n_metals` is solved **inside** each molecule.

| the input holds | what you get |
|---|---|
| one molecule | one entry in `molecules`, exact |
| one metal-bearing molecule + any number of metal-free ones | **exact** — a metal-free molecule's charge is its formal-charge sum, and the rest belongs to the metal-bearing one. This is the IRC-endpoint case |
| two or more metal-bearing molecules | the remainder is split **evenly** over all their metals, each marked `oxidation_is_exact: False`, the molecule `charge` left `None`, and a warning in `smiles_note`. It is right only when all those metals share one oxidation state, so check the flag — or pass one molecule at a time |

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
mol, _ = assemble_complex(r, ml_dative=False)    # M–L as integer orders (haptic is always dative)
```

To assemble ligand by ligand yourself — join the metal (formal charge = `oxidation`) + ligand SMILES +
`ml_bonds` + `mm_bonds`. The atom correspondence is this one line:

```python
input_idx = sorted(lg["coordinating"])[at.GetAtomMapNum() - 1]
```

⚠️ The map `[X:n]` in a ligand SMILES is **not the input index** — it is the n-th entry of the sorted `coordinating` list.
⚠️ Implicit hydrogens are not used — read with `sanitize=False` and sanitize with KEKULIZE and SETAROMATICITY removed.
⚠️ Atom correspondence for a molecule's `smiles`: its `atom_order`.

## Examples — `examples/`

Six examples — five real CSD structures and one gas-phase molecule. Each comes as four files — `<name>.xyz` (coordinates) · `<name>.wbo.json`
(total charge + Mayer bond orders) · `<name>.result.json` (**the full pipeline output**) · `<name>.png` (the figure `draw()` produces).

```bash
python examples/run_examples.py            # runs all 6 and rewrites <name>.result.json
python examples/run_examples.py 02         # only those with "02" in the name
python examples/run_examples.py --no-wbo   # without wbo (distance fallback)
python examples/draw_examples.py           # redraw the PNGs
```

| # | File | Real system | What it shows | Output | Figure |
|---|---|---|---|---|---|
| 01 | `01_dative_os_carbonyl` | `fac-[Os(CO)₃Cl₃]⁻` | σ-dative only · internal `C≡O` | [json](examples/01_dative_os_carbonyl.result.json) | [png](examples/01_dative_os_carbonyl.png) |
| 02 | `02_haptic_cp_ticl3` | `CpTiCl₃` | η⁵ haptic | [json](examples/02_haptic_cp_ticl3.result.json) | [png](examples/02_haptic_cp_ticl3.png) |
| 03 | `03_bridge_ag2cl4` | `[Ag₂Cl₄]²⁻` | μ-Cl bridge (`bridge:dative`) · two metals | [json](examples/03_bridge_ag2cl4.result.json) | [png](examples/03_bridge_ag2cl4.png) |
| 04 | `04_3c2e_gallium_bh4` | `Me₂Ga(BH₄)` | 3c2e bridging H · `B` as a ligand atom | [json](examples/04_3c2e_gallium_bh4.result.json) | [png](examples/04_3c2e_gallium_bh4.png) |
| 05 | `05_mm_quadruple_re2cl8` | `[Re₂Cl₈]²⁻` | M–M bond | [json](examples/05_mm_quadruple_re2cl8.result.json) | [png](examples/05_mm_quadruple_re2cl8.png) |
| 06 | `06_diborane_b2h6` | `B₂H₆` (gas phase) | **3c2e with no metal** — `bonds_3c2e`, bridging `[H-]`, `B(+1)`, and no `B–B` | [json](examples/06_diborane_b2h6.result.json) | [png](examples/06_diborane_b2h6.png) |

Examples 04 and 06 are a 3c2e bridge with a metal in it and one without — see
[Where a 3c2e bridge is reported](#where-a-3c2e-bridge-is-reported).

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
| `coords` | required | the `(n, 3)` coordinates passed to `predict` |
| `result` | required | the dict `predict` returned (it reads `molecules` → `metals` · `fragments` → `bonds_kekule` · `ml_bonds` · `eta` · `charge`) |
| `out` | required | where to write; the extension picks the format (`.png`, `.pdf`, `.svg`) |
| `title` | `""` | first title line |
| `subtitle` | auto | second line; by default the per-ligand charges and η, e.g. `q0=-1 η5 · q1=-1` |
| `highlight` | `()` | internal bonds `{(i, j), …}` to draw thick red — for pointing at a disputed bond |
| `projection` | auto | a projection returned by an earlier call, to put every atom in the same place |

It returns the projection it used. Pass that back as `projection=` for a second figure and the two
become comparable atom by atom:

```python
proj = draw(el, xyz, reference, "ref.png", title="reference")
draw(el, xyz, r, "pred.png", title="prediction", projection=proj, highlight={(2, 3)})
```

**What you see.** The projection is the least cluttered view of the real 3D geometry. Internal
bonds get 1/2/3 lines from `bonds_kekule`; M–L bonds are arrows
(**σ** solid black · **haptic** green dotted · **3c2e bridge** brown dashed — a `dative` bridge is
an ordinary donor bond and is drawn like σ); M–M bonds are purple, one line per order; the metal is
a purple circle carrying its oxidation state; every other atom, H included, is labelled with its
formal charge when non-zero, and a neutral carbon is just a dot.

Which H is drawn follows the skeletal convention — **an H on carbon is implied, an H on anything
else is written**. So N–H · O–H · S–H are visible, and so is an H on a metal (hydrido, 3c2e bridge)
and any H you pass in `highlight`. When several molecules are in the result they are translated
apart so they do not overlap — each one rigidly, so the geometry inside a molecule is untouched,
but the distance *between* molecules is not to scale.

## Package layout

```
xyz2mol_om/
├── api.py        predict() — the only orchestrator: it calls the stages below in order
├── config.py     every constant, threshold and switch
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

holdout **6,793 structures** · reference labels: CSD `bond_type`, tmQMg-L
`q_ligand`, and the roman numeral in the CSD `chemical_name` for the oxidation state.

⚠️ Fit and evaluation both use CSD experimental structures **relaxed with GFN2-xTB**.
Coordinates from another source (raw CSD, DFT, a force field) are off-distribution.

| Task | Metric | Value | Pool | Baseline |
|---|---|---|---|---|
| T1 ligand internal bond existence | F1 | **0.9998** | 380,315 bonds | all bonded .7306 |
| T2 conjugation call | F1 | **0.9617** | 87,602 bonds | — |
| T3 internal order `Single`/`Double`/`Triple`/`Conj` | F1 | **.9901 / .7667 / .9775 / .9617** | 380,211 bonds | all `Single` .9097 / 0 / 0 |
| T4 M–L·M–M bond existence | F1 | **0.9915** | 54,498 bonds | all bonded .5276 |
| T5 haptic call | F1 | **0.9800** | 15,331 M–L bonds | all haptic .6766 |
| T6 η^k (exact match per ligand) | accuracy | **0.9865** | 4,228 ligands | all `k=0` .8704 |
| T8 M–L order `Single`/`Double`/`Triple` | F1 | **.9935 / .7556 / .7254** | 37,638 bonds | — |
| T10 ligand charge `Σq_L` (exact match per structure) | accuracy | **0.8648** | 1,154 structures | reference-order 0.8528 |
| T10 metal oxidation state `OS` (exact match per structure) | accuracy | **0.8967** | 2,779 structures | reference-order 0.8698 |

The pool differs per task because the references do: `bond_type` covers every structure,
tmQMg-L charges cover 23% of structures and a roman numeral in the CSD name 41%; the `Σq_L` and
`OS` pools are the structures that can be scored against them. The baseline column is the
**trivial** prediction for that task, except the two `reference-order` entries — see below.

`reference-order` is the same charge rule fed the **reference** bond orders; its pool differs
(1,155 / 2,635 structures), so do not read it against the column to its left.

### Against other tools

Same pool, same references, same metrics, and **a tool's failure is scored as a wrong answer**
rather than dropped. `xyz2mol_tm` is given 60 s per structure, beyond which the structure counts
as a failure.

| | ours | `xyz2mol` | `xyz2mol_tm` | OpenBabel |
|---|---|---|---|---|
| **structures it produced an answer for** | **6,793** | 6,309 | 5,676 | **6,793** |
| T1 internal bond existence | **.9998** | .9751 | .8896 | .9983 |
| T4 M–L·M–M bond existence | **.9915** | — | .9168 | .7754 |
| T3 `Single` | **.9901** | .9717 | .9812 | .9442 |
| T3 `Double` | **.7667** | .4219 | .5693 | .3505 |
| T3 `Triple` | **.9775** | .9663 | .9770 | .1217 |
| T3 `Conj` | **.9617** | .9128 | .9323 | .7913 |
| T8 M–L `Single` | **.9935** | — | — | .9767 |
| T8 M–L `Double` | **.7556** | — | — | .0468 |
| T8 M–L `Triple` | **.7254** | — | — | .0106 |
| T5 haptic | **.9800** | — | — | — |
| T10 `Σq_L` (1,154 structures) | **.8648** | .3934 | .8120 | .1820 |
| T10 `OS` (2,779 structures) | **.8967** | — | — | — |

`—`: the tool does not produce that output.

### Valence violations — chemical validity of the output

A structure violates when a non-metal X has `b_int(X) + n_σ(X) > CAP(X)`: `b_int` is the Kekulé
bond-order sum, `n_σ(X)` = number of non-haptic M–L bonds of X. `B` and 3c2e bridging atoms are
excluded. CSD reference labels violate on 0.4% of structures.

**Violation rate against other tools** — same definition (3c2e atoms are excluded from our rows
only); a tool that emits aromatic bonds is counted at its own 1.5.

| Pool | | ours | `xyz2mol` | `xyz2mol_tm` | OpenBabel |
|---|---|---|---|---|---|
| holdout 6,793 | `b_int` only | **0.35%** | 8.91% | 7.35% | 3.96% |
| | `b_int`+`n_σ` | **0.35%** | 8.91% | 37.63% | 3.96% |
| 5,295 solved by all three other tools | `b_int` only | **0.42%** | 9.12% | 8.91% | 3.74% |
| | `b_int`+`n_σ` | **0.42%** | 9.12% | 44.91% | 3.74% |
| 5,676 solved by `xyz2mol_tm` | `b_int` only | **0.39%** | 8.77% | 8.79% | 3.54% |
| | `b_int`+`n_σ` | **0.39%** | 8.77% | 45.03% | 3.54% |

⚠️ Compare tools on the `b_int`-only row: the others emit no M–L order (`xyz2mol_tm`), no M–L bond
(`xyz2mol`) or miss many (OpenBabel).

## ⚠️ Limits

- **One unpaired electron, and only with `n_unpaired=1`.** A larger `n_unpaired` raises. Without
  it a radical comes out as the nearest closed-shell answer **with no error** — beside a metal the
  misplaced charge is cancelled by the metal's, so the total is right while the oxidation state is
  not. With it, placement can still be refused (several candidate sites, or a charge shortfall that
  does not match); `r["radical"]["note"]` says which, and `site`/`atom`/`sign` say where it went.
- **The M–M order is a placeholder.** Whether two metals are bonded *is* predicted (`mm_bonds`,
  by the same rule as M–L) but the order in that dict is the constant `1` — do not read it as
  "single bond". The `[Re₂Cl₈]²⁻` of example 05 is a quadruple bond and still comes out `1`.
- **Metals in one molecule share its remainder evenly** — mixed valence is not resolved, and
  `oxidation` is `None` when the remainder does not divide.
- **A suppressed π bond puts the oxidation state 2 too high.** When a weak M–X contact is taken as
  a σ bond and uses up an atom's valence, the neighbouring π bond is written `Single` and the
  fragment charge comes out 2 too negative. The clearest cases are repaired
  ([docs/PIPELINE.md](docs/PIPELINE.md) §T3 post-⑥); a `Single` left between two anionic atoms
  whose distance likelihood favours `Double` is flagged per fragment in `pi_suppressed`:

  ```python
  from xyz2mol_om import all_fragments
  if any(fr["pi_suppressed"] for fr in all_fragments(r)):
      ...   # this structure's ligand charges and metal oxidation state are suspect
  ```

- **3c2e and clusters are outside the two-centre formalism** — a bridging H with two internal bonds
  (`B–H–B`) fails the SMILES round-trip check, and a carborane cage's fragment charge uses the EHT
  value.

Every decision rule, with its thresholds, is in [docs/PIPELINE.md](docs/PIPELINE.md).

## License · Provenance

**MIT** ([LICENSE](LICENSE)).

The reference labels used for the fit are CSD (Cambridge Structural Database) bond labels and tmQMg-L ligand charges.
**The source data is not in this repository** — what ships here is the fit result (thresholds · likelihood parameters)
and the structures in `examples/` (five CSD-derived, one gas-phase).

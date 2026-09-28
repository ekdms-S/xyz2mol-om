# xyz2mol-om

**From the `xyz` coordinates of a transition metal complex, derive bonds, bond orders, ligand charges, and metal oxidation states.**
It is built to handle organometallics as well, hence the `-om` in the name.

## Dependencies

| Package | Version | Used for |
|---|---|---|
| `numpy` | ≥ 1.23 | coordinates, distances |
| `networkx` | ≥ 3.0 | graphs, rings, connected components |
| `rdkit` | ≥ 2023.3 | SMILES |
| `scipy` | ≥ 1.9 | the integer program (HiGHS, `scipy.optimize.milp`) |
| (optional) `matplotlib` | ≥ 3.5 | `draw()` — the 2D figure. Not needed for `predict` |

Python ≥ 3.10. The performance numbers below were measured on Python 3.11.15 · rdkit 2025.09.5 ·
numpy 2.4.6 · networkx 3.6.1 · scipy 1.17.1. With scipy 1.15.3 the program finds no solution for 3
of the 6,396 holdout structures (`predict` raises).

```bash
conda install -c conda-forge rdkit numpy networkx scipy    # or pip install rdkit numpy networkx scipy
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
| `n_unpaired` | number of unpaired electrons (default `0`). Pass `multiplicity − 1` |
| `firm_contacts` | `True` (default, or the `JOINTFIRM` environment variable): a metal contact that is firm by its own evidence — Mayer ≥ 0.30, or within 0.9 × the contact cutoff — is never read as no bond (it stays σ, or η² with its π neighbour). `False`: such a contact may also be read as no bond when the ligand's geometry leaves its donor no lone pair (useful on transition-state-like geometries, where a π carbon sits close to the metal) |

**How it decides.** Distances (and Mayer bond orders for the metal contacts) give the topology —
which atoms are bonded, which contacts are haptic, which fragments are boron cages. Then **one
integer program** decides every ligand bond order, atom charge, haptic reading, metal oxidation
state, cage charge and unpaired electron together, under the charge balance
`Σ q_L + Σ OS = total_charge`: bond lengths score the orders, charged atoms cost, and a σ donor must
keep a lone pair. A few distinct answers are compared before one is returned. Every rule is in
[docs/PIPELINE.md](docs/PIPELINE.md).

If the program has no answer (or exceeds its size limit), `predict` raises `RuntimeError` naming
the status.

⚠️ **You may run with `wbo=None`.** The metal contacts then come from distances alone, and the M–L
orders from the donor charge without Mayer's ranking. **Internal** bond orders, ligand charges and
oxidation states are essentially unchanged, but which contacts are bonds, and which are haptic,
degrade (holdout 6,396):

| | with `wbo` | without |
|---|---|---|
| T4 M–L bond existence | .9923 | .9898 |
| T8 M–L `Double` | .7641 | .7733 |
| T5 haptic | .9825 | .9607 |
| T6 η^k | .9911 | .9784 |
| T3 internal `Double` | .7945 | .7883 |
| T10 `Σq_L` | .9396 | .9386 |
| T10 `OS` | .9859 | .9863 |
| **valence-violating structures** | **0%** | **0%** |

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
r["radical"]                     # {n_unpaired, atom, atoms, site, sign, note} — where the unpaired
                                 #   electrons went: "organic" (ligand atoms), "metal", "mixed"
r["charge_balance"]              # {shortfall, ok, sites, note} — metal-free input only: `ok` is
                                 #   False when the emitted charges do not sum to total_charge
r["joint"]                       # {status, q_status, alt_os, alt_gap, cut, eta2_partner, …} — how
                                 #   the program went, and the best answer with other oxidation
                                 #   states and how much worse it scores (docs/PIPELINE.md)

r["molecules"] == [
  {"index": 0,
   "atoms": [0, 1, …],           # input atom indices
   "charge": -1,                 # this molecule's charge
   "charge_is_exact": True,      # False when several metal-bearing molecules share total_charge,
                                 #   or on a metal-bearing molecule without total_charge (charge None)

   "metals": [
     {"index": 0, "element": "Mo", "oxidation": 6,
      "oxidation_is_exact": True,      # False = the split across several metal-bearing molecules
                                       #   is not fixed by the input · None when oxidation is None
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
      "residual_charge": None,   # always None (the answer is integer); kept for the layout
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

`ml_bonds` holds exactly the M–L bonds of the answer, so it **always agrees with the molecule's
SMILES**. Contacts the topology rejects, and contacts the program reads as no bond (listed in
`r["joint"]["cut"]`), are not in it.

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

| the input holds | what you get |
|---|---|
| one molecule | one entry in `molecules`, exact |
| one metal-bearing molecule + any number of metal-free ones | **exact** — a metal-free molecule's charge is its formal-charge sum, and the rest belongs to the metal-bearing one. This is the IRC-endpoint case |
| two or more metal-bearing molecules | each metal gets the program's oxidation state and each molecule its charge, but nothing in the input fixes how `total_charge` divides between them: every such value is marked `oxidation_is_exact` / `charge_is_exact` `False`, with a note in `smiles_note`. Pass one molecule at a time when you know their charges |

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
├── api.py        predict() — topology, the program, and the result record
├── config.py     every constant and threshold
├── data/         the fitted tables (per-element-pair distances, thresholds, likelihood, OS prior)
├── geometry/     coordinates in, connectivity out — no chemistry
├── rules/        the distance likelihood · the M–L order model · the contact filters and bridge
│                 tags · the integer program (`joint.py` · `joint2.py`)
├── charge/       the per-atom formal charge
└── output/       SMILES · RDKit mol · JSON · figure
```

Everything a caller normally needs is re-exported at the top: `from xyz2mol_om import predict,
read_xyz, draw, save_json`. The subpackages are there for reading the code, and each one's
`__init__` says what it is for.

## Performance

CSD holdout **6,396 structures** · reference labels: CSD `bond_type`, tmQMg-L `q_ligand`, and the
roman numeral in the CSD `chemical_name` for the oxidation state. Structures whose references
contradict each other are left out (397 — e.g. a carbene carbon that carries an H in the
coordinates, or bond labels that give a different oxidation state than the name).

⚠️ Fit and evaluation both use CSD experimental structures **relaxed with GFN2-xTB**.
Coordinates from another source (raw CSD, DFT, a force field) are off-distribution.

| Task | Metric | Value | Pool |
|---|---|---|---|
| T1 ligand internal bond existence | F1 | 0.9999 | 354,906 bonds |
| T3 internal order `Single` / `Double` / `Triple` / `Conj` | F1 | 0.9908 / 0.7945 / 0.9835 / 0.9614 | 258,236 / 7,912 / 6,024 / 82,662 bonds |
| T4 M–L · M–M bond existence | F1 | 0.9923 | 51,748 bonds |
| T5 haptic call | F1 | 0.9825 | 14,584 M–L bonds |
| T6 η^k (exact match per ligand) | accuracy | 0.9911 | 4,037 ligands |
| T8 M–L order `Single` / `Double` / `Triple` | F1 | 0.9939 / 0.7641 / 0.7160 | 34,368 / 1,093 / 155 bonds |
| T10 ligand charge `Σq_L` (exact match per structure) | accuracy | 0.9396 | 1,043 structures |
| T10 metal oxidation state `OS` (exact match per structure) | accuracy | 0.9859 | 2,486 structures |

The pool differs per task because the references do: `bond_type` covers every structure, the
`Σq_L` pool is the structures whose metal-bound ligands tmQMg-L covers completely, and the `OS`
pool those whose name gives a single oxidation state.

**`Double` in context.** The CSD labels a `Double` either inside a conjugated system (7,310 bonds)
or on its own (602). A conjugated `Double` is right when it comes out `Conj` or as a Kekulé `2`:
0.9703. An isolated `Double` scores F1 0.7461 — most of the remainder are bonds whose length points
at another drawing of the same group than the label's (a bridging CO, whose `C–O` sits at triple
length, is written `C≡O`; an N-bound thiocyanate `N≡C–S⁻`, not `N=C=S`).

### Valence violations — chemical validity of the output

A structure violates when a non-metal X has `b_int(X) + n_σ(X) > CAP(X)`: `b_int` is the Kekulé
bond-order sum, `n_σ(X)` = number of non-haptic M–L bonds of X. `B` and 3c2e bridging atoms are
excluded. **No holdout structure violates**; the CSD reference labels themselves violate on 0.4%
of structures.

## ⚠️ Limits

- **The M–M order is a placeholder.** Whether two metals are bonded *is* predicted (`mm_bonds`,
  by the same rule as M–L) but the order in that dict is the constant `1` — do not read it as
  "single bond". The `[Re₂Cl₈]²⁻` of example 05 is a quadruple bond and still comes out `1`.
- **A π-acceptor or chelating ligand may come out in the other of its two usual charge states**
  (bipyridine as a dianion, a dithiolate as a neutral dithioketone) when the bond lengths sit
  between them; the metal's oxidation state then moves by 2. `r["joint"]["alt_os"]` and
  `alt_gap` show the best answer with other oxidation states and how close it scored.
- **A π bond can still be written `Single` between two anionic atoms.** Such bonds are flagged
  per fragment in `pi_suppressed`; each one means that fragment's charge is 2 too negative:

  ```python
  from xyz2mol_om import all_fragments
  if any(fr["pi_suppressed"] for fr in all_fragments(r)):
      ...   # this structure's ligand charges and metal oxidation state are suspect
  ```

- **A metal carbonyl is always `C≡O`**, bridging or terminal.
- **An `n_unpaired` the electron count cannot hold** (an even-electron input with one unpaired
  electron) is met by dropping the total charge: `r["joint"]["q_status"] == "q_relaxed"`, and on a
  metal-free input `r["charge_balance"]["ok"]` is `False`. When no charge can hold it, `predict`
  raises.
- **3c2e and clusters are outside the two-centre formalism** — a bridging H with two internal bonds
  (`B–H–B`) fails the SMILES round-trip check, and a boron cage's charge is its Wade–Mingos
  electron count, carried by the fragment rather than by its atoms.

Every decision rule, with its thresholds, is in [docs/PIPELINE.md](docs/PIPELINE.md).

## License · Provenance

**MIT** ([LICENSE](LICENSE)).

The reference labels used for the fit are CSD (Cambridge Structural Database) bond labels and tmQMg-L ligand charges.
**The source data is not in this repository** — what ships here is the fit result (thresholds · likelihood parameters)
and the structures in `examples/` (five CSD-derived, one gas-phase).

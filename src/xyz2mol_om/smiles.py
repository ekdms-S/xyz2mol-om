"""SMILES — built with **our bond orders and charges pinned so RDKit cannot change them**.

Produces both a single ligand (`ligand_smiles`) and the **whole complex** (`complex_smiles`).
On the complex side every M-L bond is written as a dative arrow and the metal carries its
**oxidation state** as a formal charge — see `complex_smiles` below.

⚠️ That is the sole reason this module exists. If you naively build an `RWMol` and call
`SanitizeMol`, RDKit **adds implicit hydrogens to coordinating atoms and reassigns formal charges
by its own rules** — a site that donated an electron pair to the metal (`RO⁻`, `R₂N⁻`, `Cp⁻`) is an
anion in our output, but to RDKit it looks like "a neutral atom short of hydrogens".

So for every atom we **state all three explicitly** and then lock RDKit out:

    a.SetNoImplicit(True)          pins implicit H to 0
    a.SetNumExplicitHs(0)          the H in the xyz are already **real atoms**
    a.SetFormalCharge(q)           uses our `q_atom` value as-is

And `SanitizeMol` is called **with partial flags only** — `SANITIZE_ADJUSTHS` and
`SANITIZE_SETAROMATICITY` are removed so H counts and aromatization are not recomputed.

The returned SMILES is **Kekule form** (`kekuleSmiles=True`) — the integer orders from our
step-⑥ output converter must be readable straight off it.
"""

# ruff: noqa: E501
from __future__ import annotations

from rdkit import Chem

_BOND = {1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE, 3: Chem.BondType.TRIPLE}

# 🔴 partial sanitize — H recomputation (ADJUSTHS) and aromatization (SETAROMATICITY) are
#    **turned off** (see module docstring)
_SANI = (
    Chem.SanitizeFlags.SANITIZE_ALL
    ^ Chem.SanitizeFlags.SANITIZE_ADJUSTHS
    ^ Chem.SanitizeFlags.SANITIZE_SETAROMATICITY
)


def ligand_smiles(el, atoms, bonds, charges, coord_atoms=(), with_map=True):
    """One ligand fragment -> (SMILES, atom correspondence). Returns `(None, {})` on failure.

    `el`          full element list
    `atoms`       atom indices of this fragment (in full-structure numbering)
    `bonds`       {(i, j): 1 | 2 | 3}  — **Kekule integer orders** (from the ⑥ output converter)
    `charges`     {i: q}               — our `q_atom` formal charges (integers)
    `coord_atoms` indices of atoms coordinating a metal — marked by SMILES atom map numbers
    `with_map`    whether to attach `[C:1]`-style map numbers to coordinating atoms

    ⚠️ **Atom order, formal charge and hydrogen count are all stated and locked** — see the
    module docstring.
    """
    idx = {a: k for k, a in enumerate(sorted(atoms))}
    m = Chem.RWMol()
    for a in sorted(atoms):
        at = Chem.Atom(el[a])
        at.SetNoImplicit(True)  # 🔴 no implicit H — the H in the xyz are already real atoms
        at.SetNumExplicitHs(0)
        at.SetFormalCharge(int(round(charges.get(a, 0))))
        if with_map and a in set(coord_atoms):
            at.SetAtomMapNum(1 + sorted(coord_atoms).index(a))
        m.AddAtom(at)
    for (i, j), o in bonds.items():
        if i in idx and j in idx:
            m.AddBond(idx[i], idx[j], _BOND.get(int(round(o)), Chem.BondType.SINGLE))
    mol = m.GetMol()
    try:
        # 🔴 partial sanitize — H recomputation (ADJUSTHS) and aromatization
        #    (SETAROMATICITY) are **turned off**
        Chem.SanitizeMol(mol, sanitizeOps=_SANI, catchErrors=True)
        smi = Chem.MolToSmiles(mol, kekuleSmiles=True, canonical=True)
    except Exception:
        return None, {}
    if not smi:
        return None, {}
    return smi, {a: k for a, k in idx.items()}


def verify_roundtrip(smi, el, atoms, bonds, charges):
    """Re-read the SMILES and check that **bond orders and formal charges are preserved**.

    Returns `(ok, reason)`. Atom correspondence is matched by sorted order (the same order as
    `ligand_smiles`).
    ⚠️ SMILES is canonicalized, so the atom order changes — the check uses **composition, the
    bond-order multiset and the charge sum**.
    """
    m = Chem.MolFromSmiles(smi, sanitize=False)
    if m is None:
        return False, "parse failed"
    try:
        Chem.SanitizeMol(m, sanitizeOps=_SANI, catchErrors=True)
    except Exception:
        return False, "sanitize failed"
    want_el = sorted(el[a] for a in atoms)
    got_el = sorted(a.GetSymbol() for a in m.GetAtoms())
    if want_el != got_el:
        return False, f"composition mismatch {len(want_el)} vs {len(got_el)}"
    wq = sum(int(round(charges.get(a, 0))) for a in atoms)
    gq = sum(a.GetFormalCharge() for a in m.GetAtoms())
    if wq != gq:
        return False, f"charge sum {wq} -> {gq}"
    wb = sorted(int(round(o)) for (i, j), o in bonds.items() if i in set(atoms) and j in set(atoms))
    gb = sorted(int(b.GetBondTypeAsDouble()) for b in m.GetBonds())
    if wb != gb:
        return False, f"bond-order multiset {wb[:6]}... vs {gb[:6]}..."
    if any(a.GetNumImplicitHs() for a in m.GetAtoms()):
        return False, "RDKit added implicit H"
    # 🔴 chemical validity — catches sites that cannot be written in 2-center form (2026-09-03).
    #   An `H` with 2 or more bonds is a **bridging H (3c2e)**. That is the site excluded from
    #   valence scoring in [design doc] §3 5c, and it cannot be written correctly in SMILES
    #   (2-center form) — you get things like `[H+2]`.
    for a in m.GetAtoms():
        if a.GetSymbol() == "H" and a.GetDegree() > 1:
            return False, (
                f"bridging H (3c2e) - not expressible in 2-center form (deg {a.GetDegree()})"
            )
        if abs(a.GetFormalCharge()) > 3:  # nitrido N³⁻ and carbide C³⁻ are normal
            return False, f"formal charge |q| > 3 ({a.GetSymbol()} {a.GetFormalCharge():+d})"
    return True, ""


# ══════════════════════════════════════════════════════════════════════════════
# ★ complex SMILES — the **whole complex** including metals (owner request 2026-09-03)
#   M-L bonds are **all written as dative arrows** (`->`). Bond order is collapsed here (every M-L
#   is a single arrow); the real order stays in `ml_bonds[(m,x)]["order"]` — owner's decision.
#   🔴 **The arrow is a connectivity marker only.** It must not change any formal charge (owner,
#      2026-09-03) — every atom charge is stamped with our `q_atom` / oxidation state as-is, and
#      `verify_complex` enforces that via the **(element, charge) multiset**. RDKit is locked out
#      of reassigning them.
#   Why dative: RDKit's `BondType.DATIVE` **is not counted toward the donor's valence** (measured —
#   an `N` with 3 σ bonds + 1 dative passes sanitize at charge 0). Our `q_atom` already reflects the
#   electron-pair donation as a charge, so writing M-L as a normal bond would count the donor twice.
#   The arrow points **ligand -> metal** (donor -> acceptor).
# ══════════════════════════════════════════════════════════════════════════════


def complex_smiles(el, atoms, bonds, charges, ml_pairs, mm_bonds=(), with_map=False):
    """The whole complex -> `(SMILES, output atom order)`. Returns `(None, [])` on failure.

    `el`        full element list
    `atoms`     all atom indices to include (metals + ligands)
    `bonds`     {(i, j): 1|2|3}  — **ligand-internal** Kekule integer orders
    `charges`   {i: q}           — `q_atom` formal charge for ligand atoms · **oxidation state
                                    for metals**
    `ml_pairs`  [(m, x)]         — M-L bonds. All written as `x -> m` dative
    `mm_bonds`  {(m1, m2): 1|2|3} — M-M bonds (normal bonds, not dative)
    `with_map`  whether to attach an `[X:i+1]` map number (= input atom index + 1) to every atom

    The second return value is **the input atom indices listed in SMILES output order** — the
    order changes under canonicalization, so this is what lets you match SMILES atoms to xyz atoms.
    """
    order = sorted(atoms)
    idx = {a: k for k, a in enumerate(order)}
    m = Chem.RWMol()
    for a in order:
        at = Chem.Atom(el[a])
        at.SetNoImplicit(True)  # 🔴 no implicit H — the H in the xyz are already real atoms
        at.SetNumExplicitHs(0)
        at.SetFormalCharge(int(round(charges.get(a, 0))))
        if with_map:
            at.SetAtomMapNum(a + 1)
        m.AddAtom(at)
    for (i, j), o in bonds.items():
        if i in idx and j in idx:
            m.AddBond(idx[i], idx[j], _BOND.get(int(round(o)), Chem.BondType.SINGLE))
    for (m1, m2), o in dict(mm_bonds).items():
        if m1 in idx and m2 in idx:
            m.AddBond(idx[m1], idx[m2], _BOND.get(int(round(o)), Chem.BondType.SINGLE))
    seen = set()
    for mm_, x in ml_pairs:
        if mm_ not in idx or x not in idx or (mm_, x) in seen:
            continue
        seen.add((mm_, x))
        m.AddBond(idx[x], idx[mm_], Chem.BondType.DATIVE)  # ligand -> metal
    mol = m.GetMol()
    try:
        Chem.SanitizeMol(mol, sanitizeOps=_SANI, catchErrors=True)
        smi = Chem.MolToSmiles(mol, kekuleSmiles=True, canonical=True)
    except Exception:
        return None, []
    if not smi:
        return None, []
    # `_smilesAtomOutputOrder` comes out as `[3,2,4,...]` or `[3,2,4,...,]`
    # (it differs between RDKit versions).
    try:
        raw = mol.GetProp("_smilesAtomOutputOrder").strip().strip("[]")
        out = [order[int(t)] for t in raw.split(",") if t.strip()]
    except Exception:
        out = []
    return smi, out


def verify_complex(smi, el, atoms, bonds, charges, ml_pairs, mm_bonds=(), total_charge=None):
    """Round-trip check of the complex SMILES. Returns `(ok, reason)`.

    Checked — **(element, formal charge) multiset · normal-bond order multiset · dative count ·
    implicit H = 0**.
    🔴 Charge is checked **per atom, not as a sum** — the dative arrow is only a connectivity
       marker and must not change a single one of the formal charges we stamped (owner,
       2026-09-03). A sum-only check misses a charge that moved to a different atom.
    If `total_charge` is given, it also checks that **the charge sum equals it** (⚠️ a residual
    charge that the skeleton cannot express, such as an even-ring dianion, is caught here — see
    `residual_charge`).
    """
    m = Chem.MolFromSmiles(smi, sanitize=False)
    if m is None:
        return False, "parse failed"
    try:
        Chem.SanitizeMol(m, sanitizeOps=_SANI, catchErrors=True)
    except Exception:
        return False, "sanitize failed"
    # 🔴 **The arrow is a connectivity marker only — it must not change a single charge**
    #   (owner 2026-09-03). So the check is the **(element, formal charge) multiset**, not the
    #   sum. A sum-only check cannot catch a `+1` that moved to a different atom. This check
    #   doubles as the composition check.
    want_q = sorted((el[a], int(round(charges.get(a, 0)))) for a in atoms)
    got_q = sorted((a.GetSymbol(), a.GetFormalCharge()) for a in m.GetAtoms())
    if want_q != got_q:
        bad = sorted(set(want_q) ^ set(got_q))[:4]
        return False, f"(element, charge) multiset mismatch - diff {bad}"
    gq = sum(q for _e, q in got_q)
    if total_charge is not None and gq != int(total_charge):
        return False, (
            f"charge sum differs from total charge {gq} vs {int(total_charge)} "
            "(residual fragment charge)"
        )
    aset = set(atoms)
    wb = sorted(
        [int(round(o)) for (i, j), o in bonds.items() if i in aset and j in aset]
        + [int(round(o)) for (i, j), o in dict(mm_bonds).items() if i in aset and j in aset]
    )
    gb = sorted(
        int(b.GetBondTypeAsDouble())
        for b in m.GetBonds()
        if b.GetBondType() != Chem.BondType.DATIVE
    )
    if wb != gb:
        return False, f"bond-order multiset {wb[:6]}... vs {gb[:6]}..."
    n_dat = sum(1 for b in m.GetBonds() if b.GetBondType() == Chem.BondType.DATIVE)
    want_dat = len({(mm_, x) for mm_, x in ml_pairs if mm_ in aset and x in aset})
    if n_dat != want_dat:
        return False, f"dative count {want_dat} -> {n_dat}"
    if any(a.GetNumImplicitHs() for a in m.GetAtoms()):
        return False, "RDKit added implicit H"
    return True, ""

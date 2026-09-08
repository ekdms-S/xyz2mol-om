"""Top-level API — `xyz` → bonds · orders · charges · oxidation states, per molecule.

    from xyz2mol_om import predict
    r = predict(elements, coords, total_charge=0, wbo=wbo)   # n_unpaired=1 for a doublet

Return structure (dict). **Molecules are the top level**: connected components over all bonds
(internal, M–L, M–M). An input may hold several — an IRC endpoint where the product separated, a
salt with its counter-ion — and everything below is solved inside one molecule.

    r["molecules"] = [ {
          "index":        int,
          "atoms":        [int, ...],       atom indices in the full coordinate list
          "charge":       int | None,       None when it could not be split exactly (below)
          "charge_is_exact": bool,
          "smiles":       str | None,       SMILES of the **whole molecule**, metals included.
                                            Every M–L is a dative arrow
          "smiles_ok":    bool,             passed the round-trip check
          "smiles_note":  str,              reason for failure or non-generation ("" if it passed)
          "atom_order":   [int, ...],       input atom indices in SMILES output order

          "metals":   [ {
              "index":        int,
              "element":      str,
              "oxidation":    int | None,   needs `total_charge`
              "oxidation_is_exact": bool | None,
              "mm_bonds":     {(m1, m2): 1|2|3|4},
          }, ... ],

          "fragments": [ {                  connected component of the **internal** bonds
              "index":        int,
              "atoms":        [int, ...],
              "bonds_4class": {(i,j): "Single"|"Double"|"Triple"|"Conj"},
              "bonds_kekule": {(i,j): 1|2|3},   output of the ⑥ converter (integers)
              "smiles":       str | None,   Kekule SMILES. Coordinating atoms carry map `[X:n]`
              "smiles_ok":    bool,
              "smiles_note":  str,
              "coordinating": [int, ...],
              "ml_bonds":     {(m, x): {
                    "type":   "sigma"|"haptic"|"bridge",   priority haptic > bridge > sigma
                    "order":  1|2|3|None,                 None for haptic (no order is assigned)
                    "bridge": None|"3c2e"|"dative",       T7 sub-tag (`docs/PIPELINE.md`)
              }},
              "eta":          {m: k},       η^k toward that metal (when haptic)
              "charge":       int,          fragment charge q_L
              "residual_charge": int | None,  charge the skeleton cannot express (if any)
          }, ... ],
      }, ... ]

    r["radical"]      = {"n_unpaired", "atom", "site", "note"}
                        Where the unpaired electron went when `n_unpaired=1`. `site` is
                        `"organic"` (on `atom`), `"metal"` (the oxidation state carries it, and
                        nothing else changed), or `None` with `note` giving the reason it was
                        refused. See `## Limits` in the README.
    r["total_charge"] = the input total charge (unchanged)

Most fragments are ligands — `ml_bonds` says what they coordinate — but a molecule with no metal
has one fragment that coordinates nothing, and that is how a free organic molecule appears.
`all_metals(r)` and `all_fragments(r)` flatten across molecules when the split does not matter.

🔴 **How the charge is split.** A metal-free molecule's charge is its formal-charge sum, so with
   exactly one metal-bearing molecule the remainder is exact. With two or more it is spread evenly
   over their metals and everything derived from it is flagged: `oxidation_is_exact` False,
   the molecule `charge` None, and `smiles_note` says so. Solving over the whole input instead
   would average one molecule's charge into another's metals — `CpTiCl3` alone is Ti(IV) and
   `[Os(CO)3Cl3]-` is Os(III), but fed together they came out Ti(III)/Os(III) with the sum still
   right and every check passing.
🔴 **A metal's formal charge is its oxidation state.** Without `total_charge` there is no
   oxidation state, so the molecule SMILES is not built either (the reason goes in `smiles_note`).
🔴 **The M–L orders are collapsed in the molecule SMILES.** An oxo `M=O` and a nitrido `M≡N` both
   go out as a single arrow — the real order is `ml_bonds[(m,x)]["order"]`. They are written as
   dative because RDKit's `DATIVE` is **not counted toward the donor's valence**: `q_atom` already
   reflects the electron-pair donation as a formal charge, so a normal bond would count it twice.

⚠️ **Without `wbo` (Mayer bond orders)** the M–L decision uses distance only and every order comes
   out `Single`. Pass it as `{(metal index, atom index): w}` (an xtb `--sp --wbo` output), and
   fill **every** (metal, atom) pair — a missing pair is read as "veto passed", not "unknown".
⚠️ **The SMILES is built with our orders and charges pinned** — RDKit is locked out of adding
   implicit hydrogens to coordinating atoms or reassigning formal charges (`output/smiles.py`).
   If `smiles_ok=False`, that SMILES failed the round-trip check, so **use `bonds_kekule`.**
"""


# ruff: noqa: E501
from __future__ import annotations

import collections
import warnings

import networkx as nx
import numpy as np

from .charge import frag_charge_or_eht, kekulize, octet_fix_period2, q_atom
from .config import NOCTET, RCOV, VAL, WMIN, centers
from .output import complex_smiles, ligand_smiles, verify_complex, verify_roundtrip
from .geometry import load_dint
from .charge import eht_frag_charges
from .rules import load_scores4
from .rules import bml_budget, predict_T3_T5


def _ml_candidates(el, xyz, dbond, c1g, wbo, cen):
    """T4 — presence of an M–X bond. `d < d_bond(M,X)` AND `w > w_veto(M,X)`.

    `cen` = the set of center-atom indices (`config.centers`) — **`B` is a conditional center,
    so it cannot be told apart by element alone.**
    ⚠️ Agostic exclusion (`C–H···M`) is the rule in `docs/PIPELINE.md` 3.
    """
    idx = [i for i in range(len(el)) if i not in cen]
    mets = sorted(cen)
    raw = []
    for m in mets:
        for x in idx:
            d = float(np.linalg.norm(xyz[x] - xyz[m]))
            tb, wv = dbond.get((el[m], el[x]), (c1g * (RCOV.get(el[x], 1.6) + RCOV.get(el[m], 1.6)), 0.0))
            w_mx = (wbo or {}).get((m, x), 1.0)
            if d < tb and w_mx > wv and w_mx >= WMIN:
                raw.append((m, x))
    return raw


MLIKE_EXTRA = {"B", "Al"}  # metal-like = metals ∪ {B, Al} (`docs/PIPELINE.md`))


def all_metals(r):
    """Every metal record in the result, across all molecules — order is molecule then index."""
    return [m for mol in r["molecules"] for m in mol["metals"]]


def all_fragments(r):
    """Every fragment record in the result, across all molecules."""
    return [fr for mol in r["molecules"] for fr in mol["fragments"]]


def _molecule_of(el, cls, ml_pred, mm):
    """`{atom: molecule index}` — connected components over **all** bonds, M–L and M–M included.

    A molecule here is what a chemist would call one: the internal bonds hold a ligand together,
    the M–L bonds attach it to its metal, and the M–M bonds hold a cluster together. Anything the
    input contains that touches none of those — a free counter-ion, a solvent molecule, the other
    half of a dissociated product — becomes a molecule of its own.
    """
    g = nx.Graph()
    g.add_nodes_from(range(len(el)))
    g.add_edges_from(cls)
    g.add_edges_from(ml_pred)
    g.add_edges_from(mm)
    return {a: i for i, comp in enumerate(nx.connected_components(g)) for a in comp}


def predict(elements, coords, total_charge=None, wbo=None, scores4=None, dint=None,
            complex_atom_map=False, n_unpaired=0):
    """`xyz` → bonds · orders · charges · oxidation states. See the module docstring for the
    arguments and the return value."""
    if n_unpaired not in (0, 1):
        raise ValueError(f"n_unpaired={n_unpaired!r}: only 0 (closed shell) and 1 are supported")
    el = list(elements)
    xyz = np.asarray(coords, dtype=float)
    if not wbo:
        # The Mayer bond order is the only input to the T4 veto (`w > w_veto`) and to T8
        # (M–L orders). Without it we proceed on the distance fallback — performance drops
        # (see the module docstring).
        warnings.warn(
            "no wbo (Mayer bond orders) - the M-L decision uses distance only. "
            "The T4 veto is off and M-L orders come from the distance fallback "
            "(`b_ml_dist.csv`): M-L `Double` F1 0.698 under refcode 5-fold CV "
            "(0.732 for the Mayer version). Obtain them with xtb GFN2 `--sp --wbo` and "
            "pass `wbo={(metal idx, atom idx): w}` to improve this.",
            UserWarning,
            stacklevel=2,
        )
    sc4 = scores4 if scores4 is not None else load_scores4()
    d_int, d_fb = dint if dint is not None else load_dint()

    # ① T1 — bonds inside a ligand (distance)
    #   🔴 The center atoms are decided by `centers()`, not by element — with a transition metal
    #      present, `B` is a **ligand atom** (carborane, boryl, `BH₄⁻`). `docs/PIPELINE.md`.0 0.
    cen = centers(el)
    idx = [i for i in range(len(el)) if i not in cen]
    G = nx.Graph()
    G.add_nodes_from(idx)
    for ii in range(len(idx)):
        for jj in range(ii + 1, len(idx)):
            a, b = idx[ii], idx[jj]
            # 🔴 Two guards apply **first** (aligned · same as the scorer):
            #   ① `H–H` is never a candidate
            #   ② `d > 1.8·(r_cov(a)+r_cov(b))` is not a candidate — an element pair with **no**
            #      fitted cutoff uses the global fallback `d_int = 2.0542 Å`, which is so long
            #      that it **turns hydrogen-bond contacts into covalent bonds.** Measured
            #      (`DEKKEJ` ·): 12 `F···H` contacts at 1.99 Å were taken as bonds
            #      (a covalent `F–H` is 0.92 Å and is absent from the reference labels). Those 12
            #      joined ligand fragments together and flipped 4 `C=O` bonds to `Single`.
            if el[a] == "H" and el[b] == "H":
                continue
            d_ab = float(np.linalg.norm(xyz[a] - xyz[b]))
            if d_ab > 1.8 * (RCOV.get(el[a], 1.0) + RCOV.get(el[b], 1.0)):
                continue
            if d_ab < d_int.get(tuple(sorted((el[a], el[b]))), d_fb):
                G.add_edge(a, b)

    # ② T4 — M–L bonds (distance + Mayer veto)
    import csv as _csv

    from .config import DATA

    dbond, c1g = {}, 1.3002
    for r in _csv.DictReader(open(DATA / "d_bond.csv")):
        if r["M"] == "*":
            c1g = float(r["d_bond"])
        else:
            dbond[(r["M"], r["X"])] = (float(r["d_bond"]), float(r["w_veto"]))
    ml_raw = _ml_candidates(el, xyz, dbond, c1g, wbo, cen)

    # ③④⑤ T3 · M–L orders · T5 (haptic) · R7 — **one function** produces all of it
    #.
    #   Why the caller does not assemble it: whether haptic and agostic are removed from the
    #   budget, how the M–L order candidates are chosen, and what T5's Y candidates are were each
    #   assembled differently per caller, and that diverged from the scorer in **four places**
    #   (measured · (`docs/PIPELINE.md`). Now only `ml_raw` and `wbo` are passed in.
    q_eht = eht_frag_charges(el, xyz, G)
    cls, mlout, hap, ml_pred, btag, w = predict_T3_T5(el, xyz, G, sc4, ml_raw, wbo, q_eht=q_eht)
    # the output converter and the charge use the **same budget** as ④ — haptic spends nothing,
    # and a 3c2e-participating atom spends `BML3C_COST` in total (`pipeline.bml_budget`).
    # 🔴 Before this loop had no 3c2e term at all, so ⑥ could undo what ④ allowed.
    three_c = {x for x, tg in btag.items() if tg == "3c2e"}
    bml = bml_budget([p for p in ml_pred if p not in hap], three_c)

    # ⑥ output converter — 4 classes → integer S/D/T + residual fragment charge
    orders, frag_q = kekulize(G, el, cls, dict(bml), w)
    if NOCTET:
        octet_fix_period2(el, G, orders)

    # ⑦ M–M bonds (those T4 called with a metal at both ends) — the order is left at 1 because
    #   no distance boundary is implemented yet
    mets = sorted(cen)
    mm = {}
    for a in range(len(mets)):
        for b in range(a + 1, len(mets)):
            m1, m2 = mets[a], mets[b]
            d = float(np.linalg.norm(xyz[m1] - xyz[m2]))
            tb, wv = dbond.get(
                (el[m1], el[m2]),
                (c1g * (RCOV.get(el[m1], 1.6) + RCOV.get(el[m2], 1.6)), 0.0),
            )
            if d < tb and (wbo or {}).get((m1, m2), (wbo or {}).get((m2, m1), 1.0)) > wv:
                mm[(m1, m2)] = 1

    # -- group by ligand fragment
    NAME4 = {0: "Single", 1: "Double", 2: "Triple", 3: "Conj"}
    hapset = {(min(a, b), max(a, b)) for a, b in hap}
    # T7 (`docs/PIPELINE.md`) — bridge tags `{coordinating atom: "3c2e" | "dative"}`.
    # 🔴 Taken from `predict_T3_T5` rather than recomputed: the rule now reads the
    # **pass-1** internal orders, which only that function has, and reusing its result is what
    # guarantees the output tag and the ④·⑥ budget cannot diverge.
    coord_of = collections.defaultdict(set)  # fragment representative -> coordinating atoms
    fragments = []
    q_all = {}
    qat_all = {}  # all per-atom formal charges — used by the complex SMILES
    for li, comp0 in enumerate(nx.connected_components(G)):
        comp = sorted(comp0)
        cs = set(comp)
        key = comp[0]
        b4 = {e: NAME4[v] for e, v in cls.items() if e[0] in cs}
        bk = {e: int(o) for e, o in orders.items() if e[0] in cs}
        # 🔴 For a cluster fragment (carborane and the like) the formal-charge sum cannot be
        #    trusted — use the EHT fragment charge. For the rule and its evidence see the
        #    `charge.is_cluster_frag` comment.
        qL = round(frag_charge_or_eht(G, el, cls, cs, q_eht, orders, w, frag_q))
        q_all[key] = qL
        coord = sorted({x for _m, x in ml_pred if x in cs})
        coord_of[key] = coord
        # per-atom formal charge — stamped into the SMILES as-is
        qat = {}
        for x in comp:
            bsum = sum(bk.get((min(x, w), max(x, w)), 1) for w in G[x])
            qat[x] = int(round(q_atom(el[x], float(bsum), G.degree(x),
                                      tuple(sorted(el[w] for w in G[x])))))
        qat_all.update(qat)
        smi, _map = ligand_smiles(el, comp, bk, qat, coord)
        ok, why = False, "SMILES generation failed"
        if smi:
            ok, why = verify_roundtrip(smi, el, comp, bk, qat)
        mlb_out = {}
        eta_out = {}
        # 🔴 `ml_pred`, not `ml_raw` — the agostic `C–H···M` contacts that T4 removes must not
        #   reappear in the output. They used to, so `ml_bonds` disagreed with the molecule's
        #   SMILES, which is built from `ml_pred`.
        for m, x in ml_pred:
            if x not in cs:
                continue
            e = (min(m, x), max(m, x))
            is_h = e in hapset
            # 🔴 The priority of `type` is **haptic > bridge > sigma**.
            #   The field answers in a single word, so one has to be picked when they overlap. So
            #   that nothing is lost on an overlap, the `bridge` field is **filled whenever the
            #   atom bridges** (even when haptic) — the T7 sub-tag survives.
            br = btag.get(x)
            mlb_out[(m, x)] = {
                "type": "haptic" if is_h else ("bridge" if br else "sigma"),
                "order": None if is_h else int(mlout.get((m, x), 0)) + 1,
                "bridge": br,  # None | "3c2e" | "dative"  (T7 · (`docs/PIPELINE.md`)
            }
        # 🔴 η^k is counted **per ligand**. Both the scorer
        #    (`len(comp ∩ hall)`) and the reference labels (`n_haptic_bound`) are per ligand.
        #    Counting per π fragment splits η, because a 5-ring turned Kekule by R2/R3 **breaks
        #    into 2 fragments** — `ZEGVIQ` has all 5 M–L bonds haptic yet the old count gave
        #    **η2** (truth η5 · measured).
        for m in {m0 for m0, x0 in hap if x0 in cs}:
            eta_out[m] = sum(1 for m0, x0 in hap if m0 == m and x0 in cs)
        fragments.append({
            "atoms": comp,
            "bonds_4class": b4,
            "bonds_kekule": bk,
            "smiles": smi,
            "smiles_ok": ok,
            "smiles_note": why,          # failure reason ("" if it passed)
            "coordinating": coord,
            "ml_bonds": mlb_out,
            "eta": eta_out,
            "charge": qL,
            "residual_charge": frag_q.get(key),
        })

    # -- molecules. The input may hold **several disconnected molecules** — an IRC endpoint where
    #   the product has separated, a salt with its counter-ion, a solvate. Splitting them matters
    #   for more than tidiness: the oxidation state is `(charge - sum q_L) / n_metals`, and run
    #   over the whole input that averages one molecule's charge into another's metals. Measured:
    #   `CpTiCl3` alone gives Ti(IV) and `[Os(CO)3Cl3]-` alone gives Os(II), but fed together they
    #   come out Ti(III) and Os(III) — both wrong, the sum still right, and every check passing.
    mol_of = _molecule_of(el, cls, ml_pred, mm)
    molecules = []
    for mi in sorted(set(mol_of.values())):
        atoms = sorted(a for a, k in mol_of.items() if k == mi)
        aset = set(atoms)
        molecules.append({
            "index": mi,
            "atoms": atoms,
            "metals": [m for m in mets if m in aset],
            "fragments": [fr["atoms"][0] for fr in fragments if fr["atoms"][0] in aset],
        })

    # ── one unpaired electron (`n_unpaired=1`) ────────────────────────────────────────────
    #   `q_atom` already points at the right atom; what it gets wrong is the **pricing** — an
    #   unpaired electron is read as a lone pair, so `CH3•` comes out `CH3-`. With a metal in the
    #   structure that invented `-1` is cancelled by a `+1` on the metal, so the total is right
    #   while the oxidation state is not: `Cu(I)Cl + CH3•` came out as Cu(II).
    #
    #   Placement — candidates are atoms of a **metal-free molecule** carrying a negative charge
    #   **that can still hold the electron**: neutralising the atom must leave it a non-bonding
    #   place to put it, `v(X) - b_int(X) >= 1`. A borate's `-1` fails that (B with four bonds:
    #   `3 - 4 = -1`) because the charge is structural, not a mispriced radical -- neutralising it
    #   produced a neutral four-bond boron that RDKit rejects outright. Same for a six-bond P.
    #   A carbanion passes (`4 - 3 = 1`), and so does a bare halide (`7 - 0 = 7`).
    #     0 candidates  the electron is on the metal. The oxidation state already carries it, so
    #                   nothing changes (`site = "metal"`).
    #     1 candidate   put it there and return that atom's charge to 0. The metal's oxidation
    #                   state follows, because what is left of `total_charge` is measured against
    #                   the metal-free molecules.
    #     2 or more     **refused.** A genuine counter-ion and a radical-read-as-an-anion look the
    #                   same in the graph, and nothing in the input says which is which.
    #   The result still comes back, with the closed-shell answer and `radical["note"]` saying why
    #   — a caller that wants the strict behaviour drops the structure on a non-empty note.
    radical = {"n_unpaired": n_unpaired, "atom": None, "site": None, "note": ""}
    if n_unpaired:
        bsum = collections.Counter()
        for (i, j), o in orders.items():
            bsum[i] += o
            bsum[j] += o
        free_atoms = [a for m in molecules if not m["metals"] for a in m["atoms"]]
        cand = sorted(a for a in free_atoms
                      if qat_all.get(a, 0) < 0 and VAL.get(el[a], 0) - bsum[a] >= 1)
        if not cand:
            radical["site"] = "metal" if mets else None
            if not mets:
                radical["note"] = ("no site for the unpaired electron - no metal, and no atom of "
                                   "a metal-free molecule carries a negative charge")
        elif len(cand) == 1:
            site = cand[0]
            radical["atom"], radical["site"] = site, "organic"
            qat_all[site] += 1
            for fr in fragments:
                if site in fr["atoms"]:
                    fr["charge"] += 1
                    # the fragment SMILES was written with the anionic charge - redo that one
                    bk = {e: int(o) for e, o in orders.items() if e[0] in set(fr["atoms"])}
                    qat = {a: qat_all[a] for a in fr["atoms"]}
                    smi_f, _m = ligand_smiles(el, fr["atoms"], bk, qat, fr["coordinating"],
                                              radicals={site})
                    ok_f, why_f = (False, "SMILES generation failed")
                    if smi_f:
                        ok_f, why_f = verify_roundtrip(smi_f, el, fr["atoms"], bk, qat)
                    fr["smiles"], fr["smiles_ok"], fr["smiles_note"] = smi_f, ok_f, why_f
                    break
        else:
            radical["note"] = (f"{len(cand)} candidate sites for the unpaired electron - a genuine "
                               "anion and a radical read as an anion are the same graph, so the "
                               "closed-shell answer is returned unchanged")

    # A metal-free molecule's charge is its ligands' formal-charge sum — nothing is unknown there.
    # What is left of `total_charge` belongs to the metal-bearing molecules, and with exactly one
    # of those the split is **exact**. With two or more there is one equation and two unknowns per
    # molecule, and nothing in the input says how to divide the remainder, so the oxidation state
    # is left as `None` rather than guessed.
    os_metal, os_exact = {}, True
    q_of_frag = {fr["atoms"][0]: fr["charge"] for fr in fragments}
    for mol in molecules:
        mol["charge"] = (None if mol["metals"]
                         else sum(q_of_frag[i] for i in mol["fragments"]))
        mol["charge_is_exact"] = not mol["metals"]
    with_metal = [m for m in molecules if m["metals"]]
    if total_charge is not None and with_metal:
        rest = total_charge - sum(m["charge"] for m in molecules if not m["metals"])
        if len(with_metal) == 1:
            mol = with_metal[0]
            mol["charge"], mol["charge_is_exact"] = rest, True
            num = rest - sum(q_of_frag[i] for i in mol["fragments"])
            if num % len(mol["metals"]) == 0:
                os_metal = dict.fromkeys(mol["metals"], num // len(mol["metals"]))
        else:
            # ⚠️ Two or more metal-bearing molecules: nothing in the input says how `total_charge`
            #   divides between them. The fallback spreads what is left evenly over **all** their
            #   metals, which is the pre- behaviour and is right only when the molecules
            #   happen to be symmetric. Measured on holdout: 7 structures land here and the even
            #   split gets 5 of them — so it is kept, but every value it produces is flagged
            #   `oxidation_is_exact = False` and the molecule charge is left `None`, because when
            #   it is wrong it is wrong silently (CpTiCl3 + [Os(CO)3Cl3]- gives Ti(III)/Os(III),
            #   the sum still correct and every check passing).
            allm = [x for m in with_metal for x in m["metals"]]
            num = rest - sum(q_of_frag[i] for m in with_metal for i in m["fragments"])
            if num % len(allm) == 0:
                os_metal = dict.fromkeys(allm, num // len(allm))
                os_exact = False

    # -- ⑧ complex SMILES — the whole complex. M–L bonds are **all dative arrows** (owner's
    #   decision). Bond order is collapsed here — the real M–L order is in
    #   `ml_bonds[(m,x)]["order"]`.
    #   A metal's formal charge = its **oxidation state**. Without `total_charge` the oxidation
    #   state cannot be found, so it is not built (stamping 0 would emit a SMILES whose total
    #   charge is wrong — better absent than silently wrong).
    # -- per-molecule SMILES. Each molecule gets its own — M–L bonds as dative arrows, the metal's
    #   formal charge stamped with its oxidation state. Without an oxidation state a metal-bearing
    #   molecule gets none (stamping 0 would emit a SMILES whose total charge is wrong — better
    #   absent than silently wrong); a metal-free molecule needs none.
    frag_by_key = {fr["atoms"][0]: fr for fr in fragments}
    out_mols = []
    for mol in molecules:
        frs = [frag_by_key[k] for k in mol["fragments"]]
        aset = set(mol["atoms"])
        smi = note = None
        ok, order = False, []
        if not mol["metals"]:
            # 🔴 Built the same way as a metal-bearing molecule, with the M–L and M–M lists empty
            #   — **not** by reusing the fragment SMILES. The string comes out identical (same
            #   sanitize, same canonicalization) but `complex_smiles` is what returns the output
            #   atom order, and reusing the fragment left `atom_order` empty on every metal-free
            #   molecule.
            qcx = {a: q for a, q in qat_all.items() if a in aset}
            smi, order = complex_smiles(el, mol["atoms"], {e: v for e, v in orders.items() if e[0] in aset},
                                        qcx, [], {}, with_map=complex_atom_map,
                                        radicals={radical["atom"]} & aset if radical["atom"] is not None else ())
            if smi is None:
                note = "SMILES generation failed"
            else:
                ok, note = verify_complex(smi, el, mol["atoms"],
                                          {e: v for e, v in orders.items() if e[0] in aset},
                                          qcx, [], {}, mol["charge"])
        elif not os_metal:
            note = ("oxidation state undetermined - total_charge not given, or the remainder is "
                    "not divisible by the number of metals")
        else:
            qcx = {a: q for a, q in qat_all.items() if a in aset}
            qcx.update({m: os_metal[m] for m in mol["metals"] if m in os_metal})
            sub_ml = [(m, x) for m, x in ml_pred if m in aset]
            sub_mm = {e: v for e, v in mm.items() if e[0] in aset}
            sub_or = {e: v for e, v in orders.items() if e[0] in aset}
            smi, order = complex_smiles(el, mol["atoms"], sub_or, qcx, sub_ml, sub_mm,
                                        with_map=complex_atom_map,
                                        radicals={radical["atom"]} & aset if radical["atom"] is not None else ())
            if smi is None:
                note = "SMILES generation failed"
            else:
                ok, note = verify_complex(smi, el, mol["atoms"], sub_or, qcx, sub_ml, sub_mm,
                                          mol["charge"])
            if not os_exact:
                warn = ("oxidation state is an even split - the input holds several metal-bearing "
                        "molecules and nothing says how total_charge divides between them")
                note = f"{note}; {warn}" if note else warn
        for fr in frs:
            fr.pop("charge_key", None)
        out_mols.append({
            "index": mol["index"],
            "atoms": mol["atoms"],
            "charge": mol["charge"],
            "charge_is_exact": mol["charge_is_exact"],
            "metals": [
                {
                    "index": m,
                    "element": el[m],
                    "oxidation": os_metal.get(m),
                    "oxidation_is_exact": os_exact if os_metal.get(m) is not None else None,
                    "mm_bonds": {e: v for e, v in mm.items() if m in e},
                }
                for m in mol["metals"]
            ],
            "fragments": [dict(fr, index=n) for n, fr in enumerate(frs)],
            "smiles": smi,
            "smiles_ok": ok,
            "smiles_note": note or "",
            "atom_order": order,
        })

    return {
        "radical": radical,
        "molecules": out_mols,
        "total_charge": total_charge,
    }

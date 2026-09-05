"""Top-level API — `xyz` → bonds · orders · charges · oxidation states, **per metal / per ligand**.

    from xyz2mol_om import predict
    r = predict(elements, coords, total_charge=0, wbo=wbo)

Return structure (dict)

    r["metals"]  = [ {                      one metal
          "index":        int,              atom index in the full coordinate list
          "element":      str,
          "oxidation":    int | None,       oxidation state (needs total_charge to be given)
          "mm_bonds":     {(m1, m2): 1|2|3|4},   M–M bond orders
      }, ... ]

    r["ligands"] = [ {                      one ligand fragment
          "index":        int,              fragment number (from 0)
          "atoms":        [int, ...],       atom indices in the full coordinate list
          "bonds_4class": {(i,j): "Single"|"Double"|"Triple"|"Conj"},
          "bonds_kekule": {(i,j): 1|2|3},   output of the ⑥ converter (integers)
          "smiles":       str | None,       Kekule SMILES. Coordinating atoms carry atom map `[X:n]`
          "smiles_ok":    bool,             passed the round-trip check (orders · charges · H ·
                                            chemical validity)
          "smiles_note":  str,              failure reason ("" if it passed)
          "coordinating": [int, ...],       atoms coordinating a metal
          "ml_bonds":     {(m, x): {
                "type":   "sigma"|"haptic"|"bridge",   priority haptic > bridge > sigma
                "order":  1|2|3|None,                 None for haptic (no order is assigned)
                "bridge": None|"3c2e"|"dative",       T7 sub-tag ([design doc] §3.0 5c)
          }},
          "eta":          {m: k},           η^k toward that metal (when haptic)
          "charge":       int,              ligand charge q_L
          "residual_charge": int | None,    residual charge the skeleton cannot express (if any)
      }, ... ]

    r["complex_smiles"]      = str | None   SMILES of the **whole complex**. Every M–L is a dative
                                            arrow
    r["complex_smiles_ok"]   = bool         whether the round-trip check passed
    r["complex_smiles_note"] = str          reason for failure or non-generation ("" if it passed)
    r["complex_atom_order"]  = [int, ...]   input atom indices in SMILES output order
    r["total_charge"]        = the input total charge (unchanged)

🔴 **The M–L orders are collapsed in `complex_smiles`.** An oxo `M=O` and a nitrido `M≡N` both go
   out as a single arrow — the real order is in `ligands[*]["ml_bonds"][(m,x)]["order"]` (owner's
   decision 2026-09-03). They are written as dative because RDKit's `DATIVE` **is not counted
   toward the donor's valence** — our `q_atom` already reflects the electron-pair donation as a
   formal charge, so writing them as normal bonds would count the donor twice.
🔴 **A metal's formal charge = its oxidation state.** Without `total_charge` there is no oxidation
   state, so `complex_smiles` is **not built either** (the reason goes in `complex_smiles_note`).

⚠️ **Without `wbo` (Mayer bond orders)** the M–L decision uses distance only and every order comes
   out `Single`. Pass it as `{(metal index, atom index): w}` (an xtb `--sp` output).
⚠️ **The SMILES is built with our orders and charges pinned** — RDKit is locked out of adding
   implicit hydrogens to coordinating atoms or reassigning formal charges (`smiles.py`). If
   `smiles_ok=False`, that ligand failed the round-trip check, so **do not use the SMILES; use
   `bonds_kekule`.**
"""


# ruff: noqa: E501
from __future__ import annotations

import collections
import warnings

import networkx as nx
import numpy as np

from .charge import frag_charge_or_eht, kekulize, q_atom
from .config import RCOV, centers
from .smiles import complex_smiles, ligand_smiles, verify_complex, verify_roundtrip
from .connectivity import load_dint
from .eht import eht_frag_charges
from .likelihood import load_scores4
from .pipeline import bml_budget, predict_T3_T5


def _ml_candidates(el, xyz, dbond, c1g, wbo, cen):
    """T4 — presence of an M–X bond. `d < d_bond(M,X)` AND `w > w_veto(M,X)`.

    `cen` = the set of center-atom indices (`config.centers`) — **`B` is a conditional center,
    so it cannot be told apart by element alone.**
    ⚠️ Agostic exclusion (`C–H···M`) is the rule in [design doc] §3 3.
    """
    idx = [i for i in range(len(el)) if i not in cen]
    mets = sorted(cen)
    raw = []
    for m in mets:
        for x in idx:
            d = float(np.linalg.norm(xyz[x] - xyz[m]))
            tb, wv = dbond.get((el[m], el[x]), (c1g * (RCOV.get(el[x], 1.6) + RCOV.get(el[m], 1.6)), 0.0))
            if d < tb and (wbo or {}).get((m, x), 1.0) > wv:
                raw.append((m, x))
    return raw


MLIKE_EXTRA = {"B", "Al"}  # metal-like = metals ∪ {B, Al} ([design doc] §3.1 (c))


def _drop_agostic(el, G, ml_raw):
    """Remove `C–H···M` only — μ-H and `B–H···M` (borohydride) are genuine 3c2e and are kept.

    rule  remove ⟺ el[X] = H  AND  exactly 1 metal-like neighbor  AND  some internal neighbor is
                   not metal-like
    The agostic rule of [design doc] §3.0 [T4]. Same formula as the scorer
    (`260831_propagation_prior_cv.py`).
    """
    nmet = collections.Counter(x for _m, x in ml_raw)
    out = []
    for m, x in ml_raw:
        if el[x] == "H":
            n_like = nmet[x] + sum(1 for y in G[x] if el[y] in MLIKE_EXTRA)
            if n_like == 1 and any(el[y] not in MLIKE_EXTRA for y in G[x]):
                continue
        out.append((m, x))
    return out


def predict(elements, coords, total_charge=None, wbo=None, scores4=None, dint=None,
            complex_atom_map=False):
    """`xyz` → bonds · orders · charges · oxidation states. See the module docstring for the
    arguments and the return value."""
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
    #      present, `B` is a **ligand atom** (carborane, boryl, `BH₄⁻`). [design doc] §3.0 0.
    cen = centers(el)
    idx = [i for i in range(len(el)) if i not in cen]
    G = nx.Graph()
    G.add_nodes_from(idx)
    for ii in range(len(idx)):
        for jj in range(ii + 1, len(idx)):
            a, b = idx[ii], idx[jj]
            # 🔴 Two guards apply **first** (aligned 2026-09-03 · same as the scorer):
            #   ① `H–H` is never a candidate
            #   ② `d > 1.8·(r_cov(a)+r_cov(b))` is not a candidate — an element pair with **no**
            #      fitted cutoff uses the global fallback `d_int = 2.0542 Å`, which is so long
            #      that it **turns hydrogen-bond contacts into covalent bonds.** Measured
            #      (`DEKKEJ` · 2026-09-03): 12 `F···H` contacts at 1.99 Å were taken as bonds
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
    #   (unified 2026-09-03).
    #   Why the caller does not assemble it: whether haptic and agostic are removed from the
    #   budget, how the M–L order candidates are chosen, and what T5's Y candidates are were each
    #   assembled differently per caller, and that diverged from the scorer in **four places**
    #   (measured 2026-09-03 · [design doc] §6.5). Now only `ml_raw` and `wbo` are passed in.
    q_eht = eht_frag_charges(el, xyz, G)
    cls, mlout, hap, ml_pred, btag = predict_T3_T5(el, xyz, G, sc4, ml_raw, wbo, q_eht=q_eht)
    # the output converter and the charge use the **same budget** as ④ — haptic spends nothing,
    # and a 3c2e-participating atom spends `BML3C_COST` in total (`pipeline.bml_budget`).
    # 🔴 Before 2026-09-06 this loop had no 3c2e term at all, so ⑥ could undo what ④ allowed.
    three_c = {x for x, tg in btag.items() if tg == "3c2e"}
    bml = bml_budget([p for p in ml_pred if p not in hap], three_c)

    # ⑥ output converter — 4 classes → integer S/D/T + residual fragment charge
    orders, frag_q = kekulize(G, el, cls, dict(bml))

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
    # T7 ([design doc] §3.0 5c) — bridge tags `{coordinating atom: "3c2e" | "dative"}`.
    # 🔴 Taken from `predict_T3_T5` (2026-09-06) rather than recomputed: the rule now reads the
    # **pass-1** internal orders, which only that function has, and reusing its result is what
    # guarantees the output tag and the ④·⑥ budget cannot diverge.
    coord_of = collections.defaultdict(set)  # fragment representative -> coordinating atoms
    ligands = []
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
        #    `charge.is_cluster_frag` comment (2026-09-03).
        qL = round(frag_charge_or_eht(G, el, cls, cs, q_eht))
        q_all[key] = qL
        coord = sorted({x for _m, x in ml_raw if x in cs})
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
        for m, x in ml_raw:
            if x not in cs:
                continue
            e = (min(m, x), max(m, x))
            is_h = e in hapset
            # 🔴 The priority of `type` is **haptic > bridge > sigma** (owner request 2026-09-03).
            #   The field answers in a single word, so one has to be picked when they overlap. So
            #   that nothing is lost on an overlap, the `bridge` field is **filled whenever the
            #   atom bridges** (even when haptic) — the T7 sub-tag survives.
            br = btag.get(x)
            mlb_out[(m, x)] = {
                "type": "haptic" if is_h else ("bridge" if br else "sigma"),
                "order": None if is_h else int(mlout.get((m, x), 0)) + 1,
                "bridge": br,  # None | "3c2e" | "dative"  (T7 · [design doc] §3.0 5c)
            }
        # 🔴 η^k is counted **per ligand** (aligned 2026-09-03). Both the scorer
        #    (`len(comp ∩ hall)`) and the reference labels (`n_haptic_bound`) are per ligand.
        #    Counting per π fragment splits η, because a 5-ring turned Kekule by R2/R3 **breaks
        #    into 2 fragments** — `ZEGVIQ` has all 5 M–L bonds haptic yet the old count gave
        #    **η2** (truth η5 · measured 2026-09-03).
        for m in {m0 for m0, x0 in hap if x0 in cs}:
            eta_out[m] = sum(1 for m0, x0 in hap if m0 == m and x0 in cs)
        ligands.append({
            "index": li,
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

    os_metal = {}
    if total_charge is not None and mets:
        num = total_charge - sum(q_all.values())
        if num % len(mets) == 0:
            os_metal = dict.fromkeys(mets, num // len(mets))

    # -- ⑧ complex SMILES — the whole complex. M–L bonds are **all dative arrows** (owner's
    #   decision 2026-09-03). Bond order is collapsed here — the real M–L order is in
    #   `ligands[*]["ml_bonds"][(m,x)]["order"]`.
    #   A metal's formal charge = its **oxidation state**. Without `total_charge` the oxidation
    #   state cannot be found, so it is not built (stamping 0 would emit a SMILES whose total
    #   charge is wrong — better absent than silently wrong).
    cx_smi, cx_ok, cx_note, cx_order = None, False, "", []
    if not mets:
        cx_note = "no metal - use ligand SMILES"
    elif not os_metal:
        cx_note = (
            "oxidation state undetermined - total_charge not given, or not divisible by "
            "the number of metals"
        )
    else:
        qcx = dict(qat_all)
        qcx.update(os_metal)
        cx_smi, cx_order = complex_smiles(
            el, list(range(len(el))), orders, qcx, ml_pred, mm, with_map=complex_atom_map
        )
        if cx_smi is None:
            cx_note = "complex SMILES generation failed"
        else:
            cx_ok, cx_note = verify_complex(
                cx_smi, el, list(range(len(el))), orders, qcx, ml_pred, mm, total_charge
            )

    return {
        "complex_smiles": cx_smi,
        "complex_smiles_ok": cx_ok,
        "complex_smiles_note": cx_note,
        "complex_atom_order": cx_order,
        "metals": [
            {
                "index": m,
                "element": el[m],
                "oxidation": os_metal.get(m),
                "mm_bonds": {k: v for k, v in mm.items() if m in k},
            }
            for m in mets
        ],
        "ligands": ligands,
        "total_charge": total_charge,
    }

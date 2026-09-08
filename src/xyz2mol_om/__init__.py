"""xyz2mol-om — bonds · orders · charges · oxidation states from a transition-metal `xyz`.

    from xyz2mol_om import predict
    r = predict(elements, coords, total_charge=0, wbo=wbo)

`draw(elements, coords, r, "out.png")` renders the result from the real geometry.

What is used (decision rules and formulas) and the performance figures are in `docs/PIPELINE.md`.
"""

from .api import all_fragments, all_metals, predict
from .charge import eht_frag_charges, frag_charge, kekulize, pi_suppressed, q_atom
from .config import METALS, METALS_HARD, RCOV
from .geometry import load_dint, read_xyz
from .output import (assemble_complex, draw, from_jsonable, ligand_smiles, load_json,
                     projection_axes, save_json, to_jsonable, verify_roundtrip)
from .rules import (deg_cell, fit_scores4, load_b_ml_mayer, load_scores4, ml_order_scores,
                    predict_T3_EHT, predict_T8, scores4_meta)

__version__ = "0.1.0"
__all__ = [
    "predict",
    # walk the nested result — every metal / every fragment, across molecules
    "all_metals",
    "all_fragments",
    # small helpers a caller needs to feed `predict` — reading an xyz file and
    # asking which elements this pipeline treats as metals (`METALS` includes B, `METALS_HARD`
    # does not · `RCOV` are the covalent radii the connectivity uses)
    "read_xyz",
    "METALS",
    "METALS_HARD",
    "RCOV",
    "assemble_complex",
    # 2D figure from the real geometry (needs matplotlib, an optional dependency)
    "draw",
    "projection_axes",
    "predict_T3_EHT",
    "load_scores4",
    "scores4_meta",
    "fit_scores4",
    "deg_cell",
    "load_dint",
    "load_b_ml_mayer",
    "predict_T8",
    "ml_order_scores",
    "eht_frag_charges",
    "q_atom",
    "frag_charge",
    "kekulize",
    "pi_suppressed",
    "ligand_smiles",
    "verify_roundtrip",
    "save_json",
    "load_json",
    "to_jsonable",
    "from_jsonable",
    "__version__",
]

"""xyz2mol-om — bonds · orders · charges · oxidation states from a transition-metal `xyz`.

    from xyz2mol_om import predict
    r = predict(elements, coords, total_charge=0, wbo=wbo)

What is used (decision rules and formulas) and the performance figures are in `docs/PIPELINE.md`.
"""

from .api import predict
from .assemble import assemble_complex
from .charge import frag_charge, kekulize, q_atom
from .config import METALS, METALS_HARD, RCOV
from .connectivity import load_dint
from .eht import eht_frag_charges
from .geometry import read_xyz
from .likelihood import deg_cell, fit_scores4, load_scores4, scores4_meta
from .ml_order import load_b_ml_mayer, ml_order_scores, predict_T8
from .pipeline import predict_T3_EHT
from .serialize import from_jsonable, load_json, save_json, to_jsonable
from .smiles import ligand_smiles, verify_roundtrip

__version__ = "0.1.0"
__all__ = [
    "predict",
    # small helpers a caller needs to feed `predict` — reading an xyz file and
    # asking which elements this pipeline treats as metals (`METALS` includes B, `METALS_HARD`
    # does not · `RCOV` are the covalent radii the connectivity uses)
    "read_xyz",
    "METALS",
    "METALS_HARD",
    "RCOV",
    "assemble_complex",
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
    "ligand_smiles",
    "verify_roundtrip",
    "save_json",
    "load_json",
    "to_jsonable",
    "from_jsonable",
    "__version__",
]

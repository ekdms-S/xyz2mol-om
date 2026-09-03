"""xyz2mol-om — 전이금속 착물 `xyz` 에서 결합 · 차수 · 전하 · 산화수를 낸다.

    from xyz2mol_om import predict
    r = predict(elements, coords, total_charge=0, wbo=wbo)

무엇을 쓰는지(판정 규칙·수식)와 성능은 `docs/PIPELINE.md` 에 있다.
"""

from .api import predict
from .charge import frag_charge, kekulize, q_atom
from .connectivity import load_dint
from .eht import eht_frag_charges
from .likelihood import deg_cell, fit_scores4, load_scores4, scores4_meta
from .ml_order import load_b_ml_mayer, predict_T8
from .pipeline import predict_T3_EHT
from .smiles import ligand_smiles, verify_roundtrip

__version__ = "0.1.0"
__all__ = [
    "predict",
    "predict_T3_EHT",
    "load_scores4",
    "scores4_meta",
    "fit_scores4",
    "deg_cell",
    "load_dint",
    "load_b_ml_mayer",
    "predict_T8",
    "eht_frag_charges",
    "q_atom",
    "frag_charge",
    "kekulize",
    "ligand_smiles",
    "verify_roundtrip",
    "__version__",
]

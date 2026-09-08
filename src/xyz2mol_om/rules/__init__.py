"""The decision rules — what order a bond gets, and which M–L bonds are haptic.

`conjugation` builds the delocalized set (Rule A · R2–R5) · `likelihood` scores a bond against the
per-element-pair distance model · `solvers` are the valence-cap and conjugation searches ·
`ml_order` assigns M–L orders from the Mayer bond order · `pipeline` runs ①–⑥ and T5.
Every rule and its threshold is documented in `docs/PIPELINE.md`.
"""

from .conjugation import conj_forbidden, lp_donor, rule_a_ok
from .likelihood import deg_cell, fit_scores4, load_scores4, scores4_meta
from .ml_order import load_b_ml_mayer, ml_order_scores, predict_T8
from .pipeline import bml_budget, bridge_tags, predict_T3_EHT, predict_T3_T5

__all__ = ["predict_T3_EHT", "predict_T3_T5", "bml_budget", "bridge_tags", "load_scores4",
           "scores4_meta", "fit_scores4", "deg_cell", "load_b_ml_mayer", "ml_order_scores",
           "predict_T8", "conj_forbidden", "lp_donor", "rule_a_ok"]

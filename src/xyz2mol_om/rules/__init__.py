"""The decision rules — the per-element-pair distance model, the M–L order model and the joint solve.

`likelihood` scores a bond against the per-element-pair distance model · `ml_order` scores M–L
orders from the Mayer bond order · `joint2` is the MILP that decides every order, charge, haptic
reading and oxidation state at once (`joint` holds its MILP wrapper and the order scores) ·
`pipeline` holds the ③ bond scores and the bridge tags the output reads. Every rule and its
threshold is documented in `docs/PIPELINE.md`.
"""

from .likelihood import deg_cell, fit_scores4, load_scores4, scores4_meta
from .ml_order import load_b_ml_mayer, ml_order_scores
from .pipeline import bridge_tags

__all__ = ["bridge_tags", "load_scores4", "scores4_meta", "fit_scores4", "deg_cell",
           "load_b_ml_mayer", "ml_order_scores"]

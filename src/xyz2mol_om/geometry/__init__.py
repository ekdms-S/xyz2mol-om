"""Coordinates in, connectivity out — the part of the pipeline that knows nothing about chemistry.

`io` reads an `xyz` file and measures planarity; `connectivity` loads the per-element-pair
distance table that decides which atom pairs are bonded at all (T1).
"""

from .connectivity import load_dint
from .io import ang3, plane_dev, plane_rms, read_xyz

__all__ = ["read_xyz", "plane_rms", "plane_dev", "ang3", "load_dint"]

"""Save and restore a `predict()` result **as JSON**.

The bond keys of the result dict are `(i, j)` tuples, which JSON cannot hold as-is. They are
turned into `"i,j"` strings and converted back on load. No other value is touched.

    from xyz2mol_om import predict, save_json, load_json

    save_json(predict(el, xyz, total_charge=0, wbo=wbo), "out.json")
    r = load_json("out.json")          # bond keys are (i, j) tuples again
"""

from __future__ import annotations

import json
from pathlib import Path

_BOND_KEYED = ("bonds_4class", "bonds_kekule", "ml_bonds")


def _k2s(d):
    return {(",".join(str(t) for t in k) if isinstance(k, tuple) else str(k)): v for k, v in d.items()}


def _s2k(d):
    return {tuple(int(t) for t in k.split(",")) if "," in k else k: v for k, v in d.items()}


def to_jsonable(r: dict) -> dict:
    """A **JSON-serializable** copy with tuple keys turned into `"i,j"`."""
    out = dict(r)
    out["metals"] = [{**m, "mm_bonds": _k2s(m.get("mm_bonds") or {})} for m in r.get("metals", [])]
    ligs = []
    for lg in r.get("ligands", []):
        g = dict(lg)
        for key in _BOND_KEYED:
            if key in g and isinstance(g[key], dict):
                g[key] = _k2s(g[key])
        if isinstance(g.get("eta"), dict):
            g["eta"] = {str(k): v for k, v in g["eta"].items()}
        ligs.append(g)
    out["ligands"] = ligs
    return out


def from_jsonable(r: dict) -> dict:
    """Inverse of `to_jsonable` — turns bond keys back into `(i, j)` tuples."""
    out = dict(r)
    out["metals"] = [{**m, "mm_bonds": _s2k(m.get("mm_bonds") or {})} for m in r.get("metals", [])]
    ligs = []
    for lg in r.get("ligands", []):
        g = dict(lg)
        for key in _BOND_KEYED:
            if key in g and isinstance(g[key], dict):
                g[key] = _s2k(g[key])
        if isinstance(g.get("eta"), dict):
            g["eta"] = {int(k): v for k, v in g["eta"].items()}
        ligs.append(g)
    out["ligands"] = ligs
    return out


def save_json(r: dict, path, indent: int = 1) -> Path:
    """Write the result to a JSON file. Returns the path written."""
    p = Path(path)
    p.write_text(json.dumps(to_jsonable(r), indent=indent, ensure_ascii=False) + "\n")
    return p


def load_json(path) -> dict:
    """Read a file written by `save_json` and return it in the same shape as `predict()`."""
    return from_jsonable(json.loads(Path(path).read_text()))

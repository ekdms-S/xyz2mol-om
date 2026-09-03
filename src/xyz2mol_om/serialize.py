"""`predict()` 결과를 **JSON 으로** 저장·복원한다.

결과 dict 의 결합 키는 `(i, j)` 튜플이라 JSON 이 그대로 못 담는다. `"i,j"` 문자열로 바꾸고,
읽을 때 되돌린다. 그 외 값은 손대지 않는다.

    from xyz2mol_om import predict, save_json, load_json

    save_json(predict(el, xyz, total_charge=0, wbo=wbo), "out.json")
    r = load_json("out.json")          # 결합 키가 다시 (i, j) 튜플이다
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
    """튜플 키를 `"i,j"` 로 바꾼 **JSON 직렬화 가능한** 사본."""
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
    """`to_jsonable` 의 역변환 — 결합 키를 `(i, j)` 튜플로 되돌린다."""
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
    """결과를 JSON 파일로 쓴다. 반환은 쓴 경로."""
    p = Path(path)
    p.write_text(json.dumps(to_jsonable(r), indent=indent, ensure_ascii=False) + "\n")
    return p


def load_json(path) -> dict:
    """`save_json` 이 쓴 파일을 읽어 `predict()` 반환과 같은 형태로 돌려준다."""
    return from_jsonable(json.loads(Path(path).read_text()))

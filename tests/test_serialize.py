"""`save_json`/`load_json` 왕복 — 결합 키 `(i, j)` 가 살아 돌아온다."""

from __future__ import annotations

import json

from xyz2mol_om import from_jsonable, to_jsonable


def _sample():
    return {
        "total_charge": -2,
        "metals": [{"index": 0, "element": "Ag", "oxidation": 1, "mm_bonds": {(0, 3): 1}}],
        "ligands": [
            {
                "index": 0,
                "atoms": [1],
                "bonds_4class": {(1, 2): "Conj"},
                "bonds_kekule": {(1, 2): 2},
                "ml_bonds": {(0, 1): {"type": "bridge", "order": 1, "bridge": "dative"}},
                "eta": {0: 5},
                "charge": -1,
            }
        ],
    }


def test_roundtrip_restores_tuple_keys():
    r = _sample()
    assert from_jsonable(json.loads(json.dumps(to_jsonable(r)))) == r


def test_jsonable_is_json_serializable():
    json.dumps(to_jsonable(_sample()))  # 예외가 나면 실패다

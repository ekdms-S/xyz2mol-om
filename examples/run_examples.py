"""예시를 파이프라인에 넣고 결과를 JSON 으로 저장한다 — `python examples/run_examples.py`.

각 예시는 `<이름>.xyz` + `<이름>.wbo.json`(총전하 + xtb GFN2 Mayer 결합차수) 한 쌍이고,
결과는 `<이름>.result.json` 으로 쓴다(`xyz2mol_om.save_json`).

    python examples/run_examples.py            # 5종 전부
    python examples/run_examples.py 02         # 이름에 "02" 가 든 것만
    python examples/run_examples.py --no-wbo   # wbo 없이 (거리 폴백 · 성능 소폭 감소)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

from xyz2mol_om import predict, save_json  # noqa: E402


def read_xyz(p: Path):
    lines = p.read_text().splitlines()
    n = int(lines[0].split()[0])
    el, xyz = [], []
    for ln in lines[2 : 2 + n]:
        w = ln.split()
        el.append(w[0])
        xyz.append([float(v) for v in w[1:4]])
    return el, np.array(xyz), lines[1]


def main(argv: list[str]) -> None:
    use_wbo = "--no-wbo" not in argv
    pats = [a for a in argv if not a.startswith("-")]
    for f in sorted(HERE.glob("*.xyz")):
        if pats and not any(p in f.name for p in pats):
            continue
        el, xyz, title = read_xyz(f)
        meta = json.loads(f.with_suffix(".wbo.json").read_text())
        wbo = {tuple(int(t) for t in k.split(",")): v for k, v in meta["wbo"].items()}
        r = predict(el, xyz, total_charge=meta["total_charge"], wbo=wbo if use_wbo else None)
        out = save_json(r, f.with_suffix(".result.json"))
        eta = [max(lg["eta"].values()) for lg in r["ligands"] if lg.get("eta")]
        tags = sorted({d["bridge"] or d["type"] for lg in r["ligands"] for d in lg["ml_bonds"].values()})
        print(
            f"{f.name:28} → {out.name:32} "
            f"금속 {len(r['metals'])} · 리간드 {len(r['ligands'])}"
            f" · M–L {','.join(tags)}" + (f" · η{max(eta)}" if eta else "")
        )


if __name__ == "__main__":
    main(sys.argv[1:])

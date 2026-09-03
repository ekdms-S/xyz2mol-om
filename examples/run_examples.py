"""Run the examples through the pipeline and save the results as JSON -
`python examples/run_examples.py`.

Each example is a pair of `<name>.xyz` + `<name>.wbo.json` (total charge + xtb GFN2 Mayer bond
orders), and the result is written to `<name>.result.json` (`xyz2mol_om.save_json`).

    python examples/run_examples.py            # all five
    python examples/run_examples.py 02         # only names containing "02"
    python examples/run_examples.py --no-wbo   # without wbo (distance fallback, slightly worse)
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
            f"metals {len(r['metals'])} · ligands {len(r['ligands'])}"
            f" · M–L {','.join(tags)}" + (f" · η{max(eta)}" if eta else "")
        )


if __name__ == "__main__":
    main(sys.argv[1:])

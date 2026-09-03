"""예시 5종을 돌려 출력을 요약한다 — `python examples/run_examples.py`.

각 예시는 `<name>.xyz` 와 `<name>.wbo.json` 한 쌍이다.
`wbo`(xtb GFN2 Mayer 결합차수)는 M–L 판정의 입력이라 같이 실어 뒀다 — 없이 돌리면
M–L 차수가 거리 폴백으로 매겨진다(README "입력" 절 참조).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

from xyz2mol_om import predict  # noqa: E402


def read_xyz(p: Path):
    lines = p.read_text().splitlines()
    n = int(lines[0].split()[0])
    el, xyz = [], []
    for ln in lines[2 : 2 + n]:
        w = ln.split()
        el.append(w[0])
        xyz.append([float(v) for v in w[1:4]])
    return el, np.array(xyz), lines[1]


def main() -> None:
    for f in sorted(HERE.glob("*.xyz")):
        el, xyz, title = read_xyz(f)
        meta = json.loads(f.with_suffix(".wbo.json").read_text())
        wbo = {tuple(int(t) for t in k.split(",")): v for k, v in meta["wbo"].items()}
        r = predict(el, xyz, total_charge=meta["total_charge"], wbo=wbo)
        print("=" * 78)
        print(f"▌{f.name}  —  {title.split('|')[2].strip()}")
        print(f"  총전하 {r['total_charge']}")
        for m in r["metals"]:
            print(f"  금속 {m['element']}{m['index']}  산화수 {m['oxidation']}"
                  + (f"  M–M {m['mm_bonds']}" if m.get("mm_bonds") else ""))
        for lg in r["ligands"]:
            heavy = {k: v for k, v in lg["bonds_4class"].items()
                     if el[k[0]] != "H" and el[k[1]] != "H"}
            ml = ", ".join(
                f"{el[m]}{m}–{el[x]}{x}:{d['type']}"
                + (f"({d['bridge']})" if d.get("bridge") else "")
                + (f" 차수 {d['order']}" if d.get("order") else "")
                for (m, x), d in sorted(lg["ml_bonds"].items())
            )
            print(f"  · 리간드 {lg['index']}  전하 {lg['charge']:+d}"
                  + (f"  η^k {lg['eta']}" if lg.get("eta") else "")
                  + (f"  잔여전하 {lg['residual_charge']}" if lg.get("residual_charge") else ""))
            if heavy:
                print("      내부 결합  " + " ".join(f"{el[a]}{a}-{el[b]}{b}:{v}" for (a, b), v in sorted(heavy.items())))
            print(f"      M–L        {ml}")
            print(f"      SMILES     {lg['smiles']}   (왕복검증 {'통과' if lg['smiles_ok'] else '실패: ' + lg['smiles_note']})")
        cs = r.get("complex_smiles")
        print(f"  착물 SMILES  {cs if cs else '(미생성: ' + (r.get('complex_smiles_note') or '') + ')'}")


if __name__ == "__main__":
    main()

"""Draw the five examples with `xyz2mol_om.draw` — `python examples/draw_examples.py [filter …]`.

The figure comes from the **real 3D coordinates**, projected onto the least cluttered plane, so
haptic rings and chelates stay readable (an RDKit 2D layout collapses them). See
`xyz2mol_om/drawing.py` for what the colours and line styles mean.

Needs `matplotlib`, which is an optional dependency of the package.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

from xyz2mol_om import draw, predict, read_xyz  # noqa: E402

TITLE = {
    "01_dative_os_carbonyl": "fac-[Os(CO)$_3$Cl$_3$]$^-$   σ-dative only",
    "02_haptic_cp_ticl3": "CpTiCl$_3$   η$^5$ haptic",
    "03_bridge_ag2cl4": "[Ag$_2$Cl$_4$]$^{2-}$   μ-Cl bridge · two metals",
    "04_3c2e_gallium_bh4": "Me$_2$Ga(BH$_4$)   3c2e bridging H",
    "05_mm_quadruple_re2cl8": "[Re$_2$Cl$_8$]$^{2-}$   M–M bond",
}


def main(argv):
    want = argv[1:]
    for xyz_path in sorted(HERE.glob("*.xyz")):
        name = xyz_path.stem
        if want and not any(w in name for w in want):
            continue
        meta = json.loads((HERE / f"{name}.wbo.json").read_text())
        wbo = {(int(k.split(",")[0]), int(k.split(",")[1])): v for k, v in meta["wbo"].items()}
        el, xyz = read_xyz(xyz_path)
        r = predict(el, xyz, total_charge=meta.get("total_charge"), wbo=wbo)
        out = HERE / f"{name}.png"
        draw(el, xyz, r, out, title=TITLE.get(name, name))
        print(f"{name + '.png':32s} drawn from the input geometry")


if __name__ == "__main__":
    main(sys.argv)

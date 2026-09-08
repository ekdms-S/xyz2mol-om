"""🔴 Regression — `draw()` renders, and it renders **every** bond kind it claims to.

The figure is the only place a reader sees the answer whole, so a bond the drawing silently drops
is worse than a wrong number: it looks like a clean result. Two such gaps are pinned here.

  1. **M–M bonds are drawn.** `mm_bonds` sits on `result["metals"]`, not on a ligand, and the
     first version of the drawing code never read that key — `[Re₂Cl₈]²⁻` came out as two
     unconnected metals.
  2. **The projection scan counts M–L bonds as bonds.** Its clutter score needs a length scale
     taken from real bonds; with ligands that are single atoms (`[Re₂Cl₈]²⁻` — eight Cl⁻, no
     internal bond anywhere) the internal-bond list is empty, every rotation scores 0, and the
     scan returns the first candidate. That drew the eight chlorides as four superimposed pairs.
"""

# ruff: noqa: E501
from __future__ import annotations

import numpy as np
import pytest

from xyz2mol_om.output.drawing import _clutter, projection_axes

pytest.importorskip("matplotlib")


def test_projection_uses_ml_bonds_when_there_are_no_internal_bonds():
    """Eight single-atom ligands on two metals — the scan must still be able to score a view."""
    # Re at ±1.1 on x, four Cl around each, so a view along x superimposes the two sets
    xyz = [[-1.1, 0, 0], [1.1, 0, 0]]
    ml = []
    for k, m in enumerate((0, 1)):
        sx = -1 if m == 0 else 1
        for dy, dz in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
            ml.append((m, len(xyz)))
            xyz.append([sx * 2.3, 1.6 * dy, 1.6 * dz])
    xyz = np.array(xyz, dtype=float)
    keep = list(range(len(xyz)))

    # with no bonds at all the score is blind
    assert _clutter(xyz[:, :2], np.zeros((0, 2), dtype=int)) == 0

    ax = projection_axes(xyz, keep, [], ml)
    assert ax.shape == (2, 3)
    pos = xyz @ ax.T
    d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    np.fill_diagonal(d, 9e9)
    # no two atoms may land on top of each other
    assert d.min() > 0.5, "the chosen projection superimposes atoms"


def test_draw_emits_the_mm_bond(tmp_path, monkeypatch):
    """A two-metal result whose only inter-metal information is `mm_bonds` must draw that line."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import xyz2mol_om.output.drawing as D

    el = ["Re", "Re", "Cl", "Cl"]
    xyz = np.array([[-1.1, 0, 0], [1.1, 0, 0], [-3.4, 0, 0], [3.4, 0, 0]], dtype=float)
    result = {
        "metals": [
            {"index": 0, "element": "Re", "oxidation": 3, "mm_bonds": {"0,1": 1}},
            {"index": 1, "element": "Re", "oxidation": 3, "mm_bonds": {"0,1": 1}},
        ],
        "ligands": [
            {"index": 0, "atoms": [2], "coordinating": [2], "bonds_kekule": {}, "eta": {},
             "charge": -1, "ml_bonds": {(0, 2): {"type": "sigma", "order": 1, "bridge": None}}},
            {"index": 1, "atoms": [3], "coordinating": [3], "bonds_kekule": {}, "eta": {},
             "charge": -1, "ml_bonds": {(1, 3): {"type": "sigma", "order": 1, "bridge": None}}},
        ],
    }

    # keep the axes `draw` builds so its contents can be inspected after it closes the figure
    seen = {}
    real_subplots = plt.subplots

    def spy(*a, **k):
        fig, ax = real_subplots(*a, **k)
        seen["ax"] = ax
        return fig, ax

    monkeypatch.setattr(plt, "subplots", spy)
    monkeypatch.setattr(D, "_import_pyplot", lambda: plt)

    out = tmp_path / "mm.png"
    n_before = len(plt.get_fignums())
    proj = D.draw(el, xyz, result, out, title="Re2Cl8")

    assert out.exists() and out.stat().st_size > 0
    assert proj.shape == (2, 3)
    assert len(plt.get_fignums()) == n_before, "draw must close the figure it made"

    mm = [ln for ln in seen["ax"].lines
          if matplotlib.colors.to_hex(ln.get_color()).lower() == D.MM_COLOR.lower()]
    assert mm, "the Re–Re bond was not drawn"
    # it must run between the two metals, not somewhere else
    pos = xyz @ proj.T
    (x0, x1), (y0, y1) = mm[0].get_xdata(), mm[0].get_ydata()
    assert np.allclose([x0, y0], pos[0], atol=0.2) and np.allclose([x1, y1], pos[1], atol=0.2)


def test_draw_is_reproducible_with_a_given_projection(tmp_path):
    """Passing a projection back in must be accepted — that is what pairs two figures."""
    from xyz2mol_om import draw

    el = ["Ti", "Cl"]
    xyz = np.array([[0.0, 0, 0], [2.3, 0, 0]])
    result = {
        "metals": [{"index": 0, "element": "Ti", "oxidation": 4, "mm_bonds": {}}],
        "ligands": [{"index": 0, "atoms": [1], "coordinating": [1], "bonds_kekule": {},
                     "eta": {}, "charge": -1,
                     "ml_bonds": {(0, 1): {"type": "sigma", "order": 1, "bridge": None}}}],
    }
    p1 = draw(el, xyz, result, tmp_path / "a.png", title="a")
    p2 = draw(el, xyz, result, tmp_path / "b.png", title="b", projection=p1)
    assert np.allclose(p1, p2)

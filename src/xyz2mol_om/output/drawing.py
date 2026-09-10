"""2D figure of a complex — drawn from the **real 3D coordinates**, projected.

RDKit's 2D layout collapses on metal complexes: with a haptic ligand (five ring carbons on one
metal) or a constrained chelate, atoms land on top of each other and the picture cannot be read.
Using the input geometry avoids that entirely — the atoms are where they really are, and the only
choice is which plane to look at.

What the figure shows

  * projection plane — chosen by scanning rotations around the coordinate principal axes for the
    least cluttered view (`projection_axes`)
  * terminal H is hidden; **H on a metal** (hydrido, 3c2e bridge) is kept
  * internal bonds are drawn with 1 / 2 / 3 lines from `bonds_kekule`
  * M–L bonds are arrows — `sigma` solid black · `haptic` **green dotted** · `bridge` orange dashed
  * M–M bonds are drawn purple, one line per order
  * the metal is a **purple circle** with its symbol and oxidation state (`Ti⁺⁴`)
  * a non-metal is labelled with its formal charge when non-zero (`O⁻`); a neutral carbon is a dot
  * two title lines — the caption, then per-ligand charge and η (`q0=-1 η5 · q1=-1`)

`matplotlib` is required for this module only, and is **not** a dependency of the package. Import
it lazily so the rest of `xyz2mol_om` works without it.

    from xyz2mol_om import predict, read_xyz, draw

    el, xyz = read_xyz("complex.xyz")
    r = predict(el, xyz, total_charge=0, wbo=wbo)
    draw(el, xyz, r, "complex.png", title="CpTiCl3")
"""

from __future__ import annotations

import numpy as np

from ..charge.formal import q_atom

__all__ = ["draw", "projection_axes"]

# CPK-ish colours, muted for line art
CPK = {
    "C": "#404040", "H": "#909090", "N": "#1f5fd0", "O": "#d02020", "Cl": "#20a020",
    "F": "#40c040", "Br": "#a05020", "I": "#8020a0", "S": "#c0a000", "P": "#e07000",
    "Si": "#907060", "B": "#c07070",
}
ML_COLOR = {"sigma": "#000000", "haptic": "#1a7f37", "bridge": "#b34700"}
ML_STYLE = {"sigma": "-", "haptic": ":", "bridge": "--"}
METAL_COLOR = "#8000a0"
MM_COLOR = "#8000a0"
HIGHLIGHT_COLOR = "#d00000"


def _import_pyplot():
    try:
        import matplotlib
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "xyz2mol_om.draw needs matplotlib, which is not a dependency of this package. "
            "Install it with `pip install matplotlib` (or `conda install -c conda-forge "
            "matplotlib`)."
        ) from exc
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _clutter(pts, bonds):
    """How crowded a 2D layout is — pairs of non-bonded atoms closer than 0.55 mean bond lengths."""
    if len(bonds) == 0:
        return 0
    bl = np.linalg.norm(pts[bonds[:, 0]] - pts[bonds[:, 1]], axis=1)
    lim = 0.55 * bl.mean()
    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
    np.fill_diagonal(d, 9e9)
    for a, b in bonds:
        d[a, b] = d[b, a] = 9e9
    return int((d < lim).sum() // 2)


def projection_axes(xyz, keep, bonds, ml_bonds=()):
    """Pick the least cluttered viewing plane. Returns a `(2, 3)` matrix — project with `xyz @ ax.T`.

    The principal-component plane is only the starting point; a ring seen edge-on is flat there.
    Azimuth and elevation are scanned around it and the lowest-scoring rotation wins.

    score = crowded atom pairs + 3 × (M–L bonds that the projection shortens by more than half)

    The M–L term matters: an M–L bond close to the viewing direction collapses in projection, and
    the little that is left hides under the metal's circle and the coordinating atom's label — the
    atom then looks unbonded. Trimming the arrow margins does not help, because what covers the
    line is the label box.
    """
    sub = xyz[keep] - xyz[keep].mean(0)
    _u, _s, vt = np.linalg.svd(sub, full_matrices=False)
    idx = {a: k for k, a in enumerate(keep)}
    # 🔴 M–L bonds count as bonds here. Without them a complex whose ligands are single atoms
    #    (`[Re₂Cl₈]²⁻`: eight Cl⁻, no internal bond anywhere) has an empty bond list, the clutter
    #    score is 0 for every rotation, and the scan silently returns the first candidate — which
    #    is how eight chlorides ended up drawn as four superimposed pairs.
    pairs = list(bonds) + [(m, x) for m, x in ml_bonds]
    bb = np.array([[idx[a], idx[b]] for a, b in pairs if a in idx and b in idx], dtype=int)
    if bb.size == 0:
        bb = np.zeros((0, 2), dtype=int)
    mlb = np.array([[m, x] for m, x in ml_bonds], dtype=int) if len(ml_bonds) else None
    d3 = np.linalg.norm(xyz[mlb[:, 0]] - xyz[mlb[:, 1]], axis=1) if mlb is not None else None
    best, score = vt[:2], None
    for th in np.linspace(0, np.pi, 13, endpoint=False):
        for ph in np.linspace(0, np.pi, 13, endpoint=False):
            ct, st, cp, sp = np.cos(th), np.sin(th), np.cos(ph), np.sin(ph)
            rot = np.array([[ct, -st, 0], [st, ct, 0], [0, 0, 1]]) @ np.array(
                [[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]]
            )
            ax2 = (rot @ vt.T).T[:2]
            sc = _clutter(sub @ ax2.T, bb)
            if mlb is not None:
                proj = xyz @ ax2.T
                d2 = np.linalg.norm(proj[mlb[:, 0]] - proj[mlb[:, 1]], axis=1)
                sc += 3 * int((d2 < 0.5 * d3).sum())
            if score is None or sc < score:
                best, score = ax2, sc
    return best


def draw(elements, coords, result, out, *, title="", subtitle=None, highlight=(), projection=None):
    """Draw one complex to `out` (any path matplotlib can write). Returns the projection used.

    | argument | meaning |
    |---|---|
    | `elements`·`coords` | the same inputs `predict` was given |
    | `result` | what `predict` returned |
    | `title` | first title line |
    | `subtitle` | second title line; by default the per-ligand charges and η |
    | `highlight` | internal bonds `{(i, j), …}` to draw thick red — for marking a disputed bond |
    | `projection` | reuse a projection from an earlier call, so two figures share one layout |

    Pass the returned projection back in as `projection=` to draw a second figure (a reference
    beside a prediction, say) with every atom in the same place; letting each pick its own makes
    the two impossible to compare.
    """
    plt = _import_pyplot()
    el, xyz = list(elements), np.asarray(coords, dtype=float)

    ml, kek = {}, {}
    for lg in (fr for mol in result["molecules"] for fr in mol["fragments"]):
        kek.update(lg.get("bonds_kekule") or {})
        ml.update(lg["ml_bonds"])
    met = {m["index"]: m for mol in result["molecules"] for m in mol["metals"]}
    # M–M bonds are reported per metal, keyed "i,j" — collect them once
    mm = {}
    for m in (x for mol in result["molecules"] for x in mol["metals"]):
        for key, order in (m.get("mm_bonds") or {}).items():
            a, b = (key if isinstance(key, tuple) else tuple(int(t) for t in str(key).split(",")))
            mm[(min(a, b), max(a, b))] = order
    hl = {(min(a, b), max(a, b)) for a, b in highlight}

    # atoms to draw — hide terminal H, but keep any H bound to a metal
    nbr = {}
    for a, b in kek:
        nbr.setdefault(a, set()).add(b)
        nbr.setdefault(b, set()).add(a)
    # ★ a **highlighted** H is kept as well. `highlight` means "look here", so hiding its
    #   atom defeats the argument: an R/P pair whose only change is a proton transfer came out
    #   as two identical-looking skeletons with a red bond floating at nothing.
    _hlatoms = {a for e_ in hl for a in e_}
    hide = {
        i for i, e in enumerate(el)
        if e == "H" and len(nbr.get(i, ())) <= 1 and i not in _hlatoms
        and not any((m, i) in ml for m in met)
    }
    keep = [i for i in range(len(el)) if i not in hide]
    if projection is None:
        projection = projection_axes(xyz, keep, list(kek), list(ml))
    pos = xyz @ projection.T

    fig, ax = plt.subplots(figsize=(9.5, 7.2), dpi=130)

    # internal bonds — one line per order, offset sideways
    for (a, b), order in kek.items():
        if a in hide or b in hide:
            continue
        v = pos[b] - pos[a]
        n = np.array([-v[1], v[0]])
        n = n / (np.linalg.norm(n) + 1e-9) * 0.055
        hot = (min(a, b), max(a, b)) in hl
        for k in {1: [0.0], 2: [-1.0, 1.0], 3: [-1.3, 0.0, 1.3]}[int(order)]:
            ax.plot(
                [pos[a][0] + k * n[0], pos[b][0] + k * n[0]],
                [pos[a][1] + k * n[1], pos[b][1] + k * n[1]],
                "-", lw=3.0 if hot else 1.5,
                color=HIGHLIGHT_COLOR if hot else "#303030", zorder=2 if hot else 1,
            )

    # M–M bonds — plain lines in the metal colour, one per order
    for (a, b), order in mm.items():
        v = pos[b] - pos[a]
        n = np.array([-v[1], v[0]])
        n = n / (np.linalg.norm(n) + 1e-9) * 0.055
        for k in {1: [0.0], 2: [-1.0, 1.0], 3: [-1.3, 0.0, 1.3],
                  4: [-1.95, -0.65, 0.65, 1.95]}.get(int(order), [0.0]):
            ax.plot(
                [pos[a][0] + k * n[0], pos[b][0] + k * n[0]],
                [pos[a][1] + k * n[1], pos[b][1] + k * n[1]],
                "-", lw=1.8, color=MM_COLOR, zorder=1,
            )

    # M–L arrows. The margins are in points, so a bond the projection shortens would lose its
    # whole arrow and the atom would look unbonded — scale the margins down for short bonds.
    span_x = float(pos[keep][:, 0].max() - pos[keep][:, 0].min()) or 1.0
    pt_per_unit = 9.5 * 72 / max(span_x, 1e-9)
    for (m, x), d in ml.items():
        length = float(np.linalg.norm(pos[m] - pos[x])) * pt_per_unit
        sa, sb = 9.0, 13.0
        if sa + sb > 0.55 * length:
            k = 0.55 * length / (sa + sb)
            sa, sb = sa * k, sb * k
        kind = d.get("type", "sigma")
        ax.annotate(
            "", xy=pos[m], xytext=pos[x], zorder=1,
            arrowprops=dict(
                arrowstyle="-|>", lw=1.4, color=ML_COLOR.get(kind, "k"),
                linestyle=ML_STYLE.get(kind, "-"), shrinkA=sa, shrinkB=sb,
            ),
        )

    # per-atom formal charge, recomputed with the library's own rule — the ligand total alone
    # would not say which atom carries the charge
    qat = {}
    for i in keep:
        if i in met or el[i] == "H":
            continue
        bsum = sum(o for (x, y), o in kek.items() if i in (x, y))
        nbs = tuple(sorted(el[y if x == i else x] for (x, y) in kek if i in (x, y)))
        qat[i] = int(round(q_atom(el[i], float(bsum), len(nbs), nbs)))

    for i in keep:
        e = el[i]
        if i in met:
            ox = met[i]["oxidation"]
            ax.text(
                pos[i][0], pos[i][1], f"{e}$^{{{'' if ox is None else f'{ox:+d}'}}}$",
                ha="center", va="center", fontsize=17, fontweight="bold", color=METAL_COLOR,
                bbox=dict(boxstyle="circle,pad=0.22", fc="white", ec=METAL_COLOR, lw=1.4),
                zorder=3,
            )
        elif e == "C" and not qat.get(i):
            ax.plot(pos[i][0], pos[i][1], "o", ms=3.2, color=CPK["C"], zorder=2)
        else:
            q = qat.get(i, 0)
            lab = e if not q else f"{e}$^{{{q:+d}}}$".replace("+1", "+").replace("-1", "−")
            ax.text(
                pos[i][0], pos[i][1], lab, ha="center", va="center", fontsize=12,
                color=CPK.get(e, "#606060"), fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none"), zorder=3,
            )

    ax.set_aspect("equal")
    ax.axis("off")
    # Set the limits from **every atom drawn**. Autoscale looks at `ax.plot` (the bond lines) but
    # not at `ax.text` (the atom labels), so a free counter-ion — bonded to nothing — lands
    # outside the axes, and `bbox_inches="tight"` then grows the canvas until the title sits on
    # top of the molecule. The top band is reserved for the title.
    pts = pos[keep]
    xa, xb = float(pts[:, 0].min()), float(pts[:, 0].max())
    ya, yb = float(pts[:, 1].min()), float(pts[:, 1].max())
    mx, my = 0.06 * max(xb - xa, 1e-9), 0.06 * max(yb - ya, 1e-9)
    band = 0.17 * max(yb - ya, 1e-9)
    ax.set_xlim(xa - mx, xb + mx)
    ax.set_ylim(ya - my, yb + my + band)

    if subtitle is None:
        subtitle = " · ".join(
            f"q{lg['index']}={lg['charge']:+d}"
            + (f" η{max(lg['eta'].values())}" if lg.get("eta") else "")
            for mol in result["molecules"] for lg in mol["fragments"]
        )
    ax.text(
        (xa + xb) / 2, yb + my + band * 0.55, f"{title}\n{subtitle}",
        ha="center", va="center", fontsize=11, zorder=5,
    )
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return projection

"""xyz 읽기 · 평면성 · 각도.

⚠️ **`ognm-bh-workspace/code/analysis/scratch/260830_fit_t10_charge.py` 에서 이관한 코드다**
(2026-09-03). 함수 본문은 **그대로** 옮겼다 — 판정 규칙을 바꾸지 않는다.
"""

# ruff: noqa: E501
from __future__ import annotations

import numpy as np


def read_xyz(p):
    L = p.read_text().split("\n")
    n = int(L[0])
    el, x = [], np.empty((n, 3))
    for k in range(n):
        q = L[2 + k].split()
        el.append(q[0])
        x[k] = (float(q[1]), float(q[2]), float(q[3]))
    return el, x

def plane_rms(pts):
    c = pts - pts.mean(axis=0)
    return float(np.sqrt(np.mean((c @ np.linalg.svd(c, full_matrices=False)[2][-1]) ** 2)))

def plane_dev(c, nb):
    a, b, d = nb
    n = np.cross(b - a, d - a)
    ln = np.linalg.norm(n)
    return 0.0 if ln < 1e-9 else float(abs(np.dot(c - a, n / ln)))

def ang3(a, b, c):
    import math

    v1, v2 = a - b, c - b
    d = np.linalg.norm(v1) * np.linalg.norm(v2)
    return (
        180.0 if d < 1e-9 else math.degrees(math.acos(max(-1, min(1, float(np.dot(v1, v2) / d)))))
    )

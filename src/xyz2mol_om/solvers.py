"""탐색 — 원자가 상한 정확 해(④) · 공액 집합 국소 탐색(③) · R6 교환.

⚠️ **`ognm-bh-workspace/code/analysis/scratch/260830_fit_t10_charge.py` 에서 이관한 코드다**
(2026-09-03). 함수 본문은 **그대로** 옮겼다 — 판정 규칙을 바꾸지 않는다.
"""

# ruff: noqa: E501
from __future__ import annotations

import collections

import networkx as nx
import numpy as np

from .config import CAP, ORD4, R6SWAP, VTGT
from .charge import _qfrag, frag_charge, q_atom


def _kek_val(G, el, cls):
    """원자별 **Kekulé 셈** 원자가. 공액 결합 `k` 개는 `k+1` 로 센다(그중 하나가 π)."""
    nk = collections.Counter()
    bn = collections.defaultdict(float)
    for e, v in cls.items():
        if v == 3:
            nk[e[0]] += 1
            nk[e[1]] += 1
        else:
            bn[e[0]] += ORD4[v]
            bn[e[1]] += ORD4[v]
    return {x: bn[x] + (nk[x] + 1 if nk[x] else 0) for x in set(bn) | set(nk)}

def _solve_cap(G, el, sc, conj, bml, ml_sc=None, ml_max=2):
    """상한만 강제하는 **정확 해** — 상한을 만족하는 배정 중 우도 최대 (Blossom, 다항시간).

    `ml_sc` 를 주면 **M–L 차수도 같은 최적화 안에서** 정한다(차수를 1단위씩 올리고,
    단위마다 더미를 하나만 두어 같은 단위가 두 번 쓰이는 것을 막는다 ⇒ `ml_max=2` 면 Triple).
    반환 `(내부 클래스, M–L 클래스)`.
    """
    k_of = collections.Counter()
    for e in conj:
        k_of[e[0]] += 1
        k_of[e[1]] += 1
    nonc = [e for a, b in G.edges for e in [(min(a, b), max(a, b))] if e not in conj]
    use = collections.defaultdict(float)
    for x in G.nodes:
        use[x] = (k_of[x] + 1 if k_of[x] else 0.0) + bml.get(x, 0.0)
    for e in nonc:
        use[e[0]] += 1.0
        use[e[1]] += 1.0
    out = {e: 3 for e in conj}
    for e in nonc:  # ① Triple — 우도 argmax 가 Triple 이고 양쪽 여유가 2 이상인 것만
        s3 = sc.get(e)
        if not s3 or max(s3, key=s3.get) != 2:
            continue
        if all(CAP.get(el[x], 4) - use[x] >= 2 - 1e-9 for x in e):
            out[e] = 2
            use[e[0]] += 2
            use[e[1]] += 2
    r = {}
    for x in G.nodes:
        v = int(np.floor(CAP.get(el[x], 4) - use[x] + 1e-9))
        if v > 0:
            r[x] = min(v, 2)
    H = nx.Graph()
    for e in nonc:  # ② Double — 여유 안에서 정확 최대 가중 매칭
        if e in out:
            continue
        s3 = sc.get(e)
        if not s3 or 1 not in s3 or 0 not in s3:
            continue
        g = s3[1] - s3[0]
        if g <= 0 or r.get(e[0], 0) < 1 or r.get(e[1], 0) < 1:
            continue
        for ia in range(r[e[0]]):
            for ib in range(r[e[1]]):
                H.add_edge((e[0], ia), (e[1], ib), weight=g, e=e)
    mlout = {}
    if ml_sc:
        for key, sm in ml_sc.items():
            m_, x_ = key
            # 🔴 기준선은 **그 쌍에 존재하는 최저 클래스**다 (2026-09-03 수정).
            #   옛 판은 0 으로 못박아 T8 상수 `Double`/`Triple` 쌍을 `Single` 로 내보냈고,
            #   `sm[0]` 을 무조건 읽어 그런 쌍에서 KeyError 로 죽었다.
            base = min(sm)
            mlout[key] = base
            if 1 not in sm or base != 0 or r.get(x_, 0) < 1:
                continue
            incs = [sm[1] - sm[0]]
            if ml_max >= 2 and 2 in sm:
                incs.append(sm[2] - sm[1])
            for u, g in enumerate(incs):
                if g <= 0:
                    break
                du = ("_mlu", m_, x_, u)
                for ia in range(r[x_]):
                    H.add_edge((x_, ia), du, weight=g, e=("ML", key))
    if H.number_of_edges():
        cnt = collections.Counter()
        for u, v in nx.max_weight_matching(H, maxcardinality=False):
            tg = H[u][v]["e"]
            if isinstance(tg, tuple) and tg and tg[0] == "ML":
                cnt[tg[1]] += 1
            else:
                out[tg] = 1
        for key, c in cnt.items():
            mlout[key] = min(c, 2)
    for e in nonc:
        out.setdefault(e, 0)
    return out, mlout

def _solve_sc(G, el, sc, ringA, coord, bml, lam_hi=10.0, lam_lo=10.0, maxit=50):
    """우도 위에서 **양쪽 원자가 제약** 국소 탐색 — 배위 원자는 미달을 안 벌한다.

    `D`·`D_satA` 가 쓰는 것과 같은 탐색이다. 여기서는 **공액 집합을 정하는 데만** 쓴다
    (차수 자체는 뒤의 정확 해가 다시 정한다).
    """
    cur = {}
    for a, b in G.edges:
        e = (min(a, b), max(a, b))
        cur[e] = max(sc[e], key=sc[e].get) if e in sc else 0
    sm = collections.defaultdict(float)
    for e, v in cur.items():
        sm[e[0]] += ORD4[v]
        sm[e[1]] += ORD4[v]
    inc = collections.defaultdict(list)
    for e in sc:
        inc[e[0]].append(e)
        inc[e[1]].append(e)

    def pn(x):
        if el[x] == "B":
            return 0.0
        hi = lam_hi * max(0.0, sm[x] + bml.get(x, 0.0) - CAP.get(el[x], 4))
        lo = 0.0 if x in coord else lam_lo * max(0.0, VTGT.get(el[x], 4) - sm[x])
        return hi + lo

    for _ in range(maxit):
        hot = {x for x in sm if pn(x) > 0}
        if not hot:
            break
        best = (1e-9, None, None)
        for e in {y for x in hot for y in inc[x]}:
            i2, j2 = e
            c0 = cur[e]
            base = sc[e][c0] - pn(i2) - pn(j2)
            for c1 in sc[e]:
                if c1 == c0:
                    continue
                dv = ORD4[c1] - ORD4[c0]
                sm[i2] += dv
                sm[j2] += dv
                nw = sc[e][c1] - pn(i2) - pn(j2)
                sm[i2] -= dv
                sm[j2] -= dv
                if nw - base > best[0]:
                    best = (nw - base, e, c1)
        if best[1] is None:
            break
        e, c1 = best[1], best[2]
        dv = ORD4[c1] - ORD4[cur[e]]
        sm[e[0]] += dv
        sm[e[1]] += dv
        cur[e] = c1
    return {e: (3 if e in ringA else v) for e, v in cur.items()}

def r6_swap(G, el, xyz, cls, bml=None, maxit=6):
    """R6 — 같은 중심의 동일 원소 결합에서 **거리 순서와 차수 순서를 맞춘다** (2026-09-03).

    `cls` 를 제자리에서 고친다. 반환 = 교환 횟수. 판정·근거는 `R6SWAP` 주석 참조.
    """
    if not R6SWAP:
        return 0
    bml = bml or {}
    ORD = [1.0, 2.0, 3.0, 1.5]
    n_sw = 0
    for _ in range(maxit):
        moved = False
        for x in G.nodes:
            by = collections.defaultdict(list)
            for y in G[x]:
                e = (min(x, y), max(x, y))
                if e not in cls or cls[e] == 3:  # Conj 는 대상 아님
                    continue
                by[el[y]].append((float(np.linalg.norm(xyz[x] - xyz[y])), y, e))
            for _ey, lst in by.items():
                if len(lst) < 2:
                    continue
                lst.sort()
                for ii in range(len(lst)):
                    for jj in range(ii + 1, len(lst)):
                        (_d1, y1, e1), (_d2, _y2, e2) = lst[ii], lst[jj]
                        if cls[e1] >= cls[e2]:
                            continue  # 짧은 쪽이 이미 같거나 높다
                        dv = ORD[cls[e2]] - ORD[cls[e1]]
                        # 교환 후 Y1 은 +dv, Y2 는 −dv. Y1 의 상한만 확인하면 된다.
                        b1 = sum(
                            ORD[cls[(min(y1, w), max(y1, w))]]
                            for w in G[y1]
                            if (min(y1, w), max(y1, w)) in cls
                        ) + bml.get(y1, 0.0)
                        if b1 + dv > CAP.get(el[y1], 4) + 1e-9:
                            continue
                        cls[e1], cls[e2] = cls[e2], cls[e1]
                        n_sw += 1
                        moved = True
        if not moved:
            break
    return n_sw

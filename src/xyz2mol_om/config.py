"""판정에 쓰는 상수와 채택 플래그 — **여기 값이 파이프라인의 전부다**.

⚠️ **`ognm-bh-workspace/code/analysis/scratch/260830_fit_t10_charge.py` 에서 이관한 코드다**
(2026-09-03). 함수 본문은 **그대로** 옮겼다 — 판정 규칙을 바꾸지 않는다.
"""

# ruff: noqa: E501
from __future__ import annotations

import os
import re
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data"


CONJ = {"Aromatic", "Delocalised"}
CLS = {"Single": 0, "Double": 1, "Triple": 2}
ORD = [1.0, 2.0, 3.0]
VAL = {
    "H": 1,
    "B": 3,
    "C": 4,
    "N": 5,
    "O": 6,
    "F": 7,
    "Si": 4,
    "P": 5,
    "S": 6,
    "Cl": 7,
    "As": 5,
    "Se": 6,
    "Br": 7,
    "Te": 6,
    "I": 7,
}
FULL = {"H": 2}  # 채움 정원. 기본 8, H 만 2
METALS = set(
    "Ti Zr Hf Nb Ta V La Sc Y Ce Cr Mo W Mn Re Fe Ru Os Co Rh Ir Ni Pd Pt "
    "Cu Ag Au Zn Al Ga In Sn Pb Mg B".split()
)
HUCKEL = [2, 6, 10, 14, 18]

USE_DINT = os.environ.get("USE_DINT", "0") == "1"
# ★ 채택안 D (§5.0.11 ⑭) — 기본 켜짐. USE_D3=0 이면 옛 T2 게이트 경로로 되돌린다.
USE_D3 = os.environ.get("USE_D3", "1") == "1"
# ★ 채택안 D_eht (오너 확정 2026-09-01) — 기본 켜짐. USE_EHT=0 이면 옛 D 경로.
USE_EHT = os.environ.get("USE_EHT", "1") == "1"
EHT_CACHE = os.environ.get("EHT_CACHE", "")  # 조각 전하 캐시 CSV (없으면 매번 직접 계산)

T8FORM = os.environ.get("T8FORM", "thr")

TAU_P, TAU_E, LAM, MAX_ITER = 0.05, 0.02, 10.0, 50  # ⑩ CV 확정값
SP2_EL = {"C", "N", "O", "S", "B", "P", "Se"}
RCOV = {
    "H": 0.31,
    "B": 0.84,
    "C": 0.76,
    "N": 0.71,
    "O": 0.66,
    "F": 0.57,
    "Si": 1.11,
    "P": 1.07,
    "S": 1.05,
    "Cl": 1.02,
    "As": 1.19,
    "Se": 1.20,
    "Br": 1.20,
    "Te": 1.38,
    "I": 1.39,
}
VMAX = {
    "H": 1,
    "B": 4,
    "C": 4,
    "N": 4,
    "O": 3,
    "F": 1,
    "Si": 4,
    "P": 5,
    "S": 6,
    "Cl": 1,
    "As": 5,
    "Se": 6,
    "Br": 1,
    "Te": 6,
    "I": 1,
}

ORD4 = [1.0, 2.0, 3.0, 1.5]  # 0 Single · 1 Double · 2 Triple · 3 Conj
VTGT = {
    "H": 1,
    "B": 3,
    "C": 4,
    "N": 3,
    "O": 2,
    "F": 1,
    "Si": 4,
    "P": 3,
    "S": 2,
    "Cl": 1,
    "Br": 1,
    "I": 1,
    "Se": 2,
    "As": 3,
    "Te": 2,
}
LAM_LO = 10.0  # 미달 벌점 (배위 원자에는 안 건다 — M–L 이 흡수)

CAP = {  # 고립쌍 용량 — `2·b_int + 2·lp = 8` ⇒ `b_int + b_ML <= 4` (H 는 1)
    "H": 1,
    "B": 4,
    "C": 4,
    "N": 5,
    "O": 4,
    "F": 4,
    "Si": 6,
    "P": 6,
    "S": 6,
    "Cl": 7,
    "As": 6,
    "Se": 6,
    "Br": 6,
    "Te": 6,
    "I": 6,
}

# ★ `CAPSET` — `CAP` 상한 변형 (2026-09-02 오너 지적, §5.4.1 C-2). 기본은 현행 `octet`.
#   현재 표는 **원자가 상한이 아니라 옥텟 상한**이다(`2b + 2lp = 8 ⇒ b ≤ 4`) — `lp` 를 0까지
#   자유롭게 두어 2주기가 전부 4(N 5)가 된다. `CAP(O)=4` 는 O²⁺ 를, `CAP(N)=5` 는 5가 N 을
#   허용한다. 옥텟과 형식전하를 같이 풀면 옳은 상한은 **`b_max = 8 − v + q`** 다:
#       C 4 · N 3(+1 → 4) · O 2(+1 → 3) · F 1(+1 → 2)
#   `tight` = 양이온까지 허용한 상한 · `mid` = O·N 만 조이고 F 는 둔다.
_CAPSET = os.environ.get("CAPSET", "octet")
if _CAPSET == "tight":
    CAP = dict(CAP, O=3, N=4, F=2)
elif _CAPSET == "mid":
    CAP = dict(CAP, O=3, N=4)
elif _CAPSET != "octet":
    raise SystemExit(f"CAPSET={_CAPSET!r} 는 octet|mid|tight 중 하나여야 한다")

EHT_CUTOFF = -10.0
_EHT_VE = {
    "H": 1,
    "B": 3,
    "C": 4,
    "N": 5,
    "O": 6,
    "F": 7,
    "Si": 4,
    "P": 5,
    "S": 6,
    "Cl": 7,
    "As": 5,
    "Se": 6,
    "Br": 7,
    "Te": 6,
    "I": 7,
}

RULEA = os.environ.get("RULEA", "ge5")
if RULEA not in ("ge5", "eq6", "off"):
    raise SystemExit(f"RULEA={RULEA!r} 는 ge5|eq6|off 중 하나여야 한다")

R2CONJ = os.environ.get("R2CONJ", "1") == "1"
# 🔴 R3 (2026-09-02, 시험) — R2 는 **헤테로원자에 붙은 결합만** 막는다. 피롤·이미다졸 5원 고리의
#    `C=C` 는 탄소-탄소라 안 걸려 `Conj` 로 새고, 그게 §8 정답지에서 `Double` 오답 6,171 결합의
#    정체다(설계도 §8). R3 은 **R2 가 걸린 원자를 포함하는 5원 고리를 통째로** Kekulé 로 둔다.
#    ⚠️ CSD 가 `Aromatic` 으로 적은 5원 헤테로고리에서는 반대로 오답이 될 수 있다 — 재봐야 안다.
R3RING = os.environ.get("R3RING", "1") == "1"  # ★ 채택 2026-09-02
# R3 범위 — all(도너 아무거나) · **N(질소 도너만 · 채택)** · mono(도너 1개 + 나머지 전부 탄소)
#   실측(CV · CVPOOL 26,075 · §8 채점): Double  all .5586 · N .5297 · mono .4226
#                                        Sq_L   all .7796 · N .7932 · mono .7914
#   ⇒ N 이 all 이득의 82% 를 대가의 37% 로 얻는다.
R3MODE = os.environ.get("R3MODE", "N")  # ★ 채택 범위 = 질소 도너만
# 🔴 ROP — T3 우도의 **두 번째 차원** (2026-09-02). 거리는 `C–C` 에서 Double↔Conj 를
#    못 가른다(1차원 최적 임계 F1 0.4473 vs ROP 0.6171 · 표본 1,500 · in-sample 상한).
#    각도 3종(이면각·결합각·평면이탈)은 자명 기준선과 동일해 분리력 0 이었다.
# 🔴 R4 (2026-09-02, 시험) — **4n 전탄소 고리(4·8원)** 는 반방향족이라 비편재가 아니다
#    (COT 는 욕조형 D2d 중성). 우리가 `Conj` 로 내는 오답 1,269 결합이 여기다.
#    ⚠️ η⁸-COT²⁻ 는 **평면 10π 방향족**이므로 제외해야 한다 ⇒ **비평면일 때만** 건다.
R4RING = os.environ.get("R4RING", "1") == "1"  # ★ 채택 2026-09-02
# 🔴 R5 (2026-09-02, 시험) — **결합 1개짜리 `Conj` 조각은 비편재가 아니다.**
#   `Conj` 는 최소 2결합에 걸쳐야 성립한다. 오답 분해에서 "비고리 고립 이중결합인데 `Conj`"
#   가 3,411 결합이었다(정답 `Double` 오답의 22%). 강등하면 `_solve_cap` 이 우도·제약으로
#   정수(S/D/T)를 고른다. 파라미터 0개.
#   실측(CV · `CVPOOL` · 원본 CSD 정답지): `Double` .6682 → **.6913** · `Conj` .9593 → .9612 ·
#   유출 10,274 → 8,752 · `Sq_L` .7916 → .7932 · `OS` .8044 → .8054 (전하는 오히려 미세 상승).
#   ⚠️ **규칙을 `conj_forbidden` **밖**에 두면 CV 경로에 안 걸린다** — `260831_propagation_prior_cv`
#      는 `predict_T3_EHT` 를 부르지 않고 자체 경로(`conj_lik` -> `solve_cap`)로 돈다.
#      R5 는 양쪽에 각각 넣었다. 초판은 이 함정으로 **효과 0 이 나왔다.**
R5SOLO = os.environ.get("R5SOLO", "1") == "1"  # ★ 채택 2026-09-03
# ★ `QHV` — 초원자가 전하식 일반화 (2026-09-03 · 기본 off · 판정은 `q_atom` 주석 참조)
QHV = os.environ.get("QHV", "1") == "1"  # ★ 채택 2026-09-03
# ★ `R6SWAP` — **같은 중심의 동일 원소 결합은 거리 순서와 차수 순서가 같아야 한다** (2026-09-03).
#   나이트로 `N(=O)=O` · 카복실레이트 `C(=O)O` 처럼 **한 중심에 같은 원소가 2개 이상** 붙은 자리는
#   지금 각 결합을 **독립으로** 채점한다 ⇒ 더 긴 결합이 `Double`, 더 짧은 것이 `Single` 이 될 수 있다.
#   판정  교환한다 ⟺ e1=(X,Y1) · e2=(X,Y2) · el[Y1] == el[Y2]
#                  AND d(e1) < d(e2)  AND  ord(e1) < ord(e2)
#                  AND 교환 후 Y1·Y2 가 `CAP` 을 안 넘는다
#   ⚠️ **교환이므로 X 의 원자가도 조각 결합차수 총합도 불변** — ⑤ 의 EHT 전하 목표를 깨뜨리지 않는다.
#   `Conj`(1.5)는 대상에서 제외한다. 파라미터 0개.
R6SWAP = os.environ.get("R6SWAP", "0") == "1"
# ★ `GNEG` — ④ 정확 해의 `g ≤ 0` 탈락을 **조각 전하가 상향을 요구할 때만** 해제 (2026-09-03).
#   `q_frag = C0 + 2B` 이므로 `q_EHT > q(전부 Single)` 인 조각은 결합차수를 **더 올려야** 한다.
#   그런데 ④ 는 `score(Double) − score(Single) > 0` 인 결합만 후보로 넣는다 ⇒ 우도가 `Single` 을
#   선호하면 전하가 요구해도 ④ 에서 못 올리고, 뒤의 ⑤ 가 탐욕으로 아무 데나 올린다.
#   실측(단계 덤프): 표적 2,913 중 **1,890(64.9%)이 이 필터에 걸려 후보로도 안 올랐다.**
GNEG = os.environ.get("GNEG", "0") == "1"
# ★ `EHTMINFRAG` — ⑤ EHT 조각 전하 목표를 **원자 수가 이 값 미만인 조각에는 적용하지 않는다**
#   (2026-09-03 · 기본 0 = 현행, 전 조각에 적용).
#   근거(실측 · 표적 2,916 · 정답 배정 대비): 조각 크기별 EHT 목표 오류율이
#     **2원자 99% (496/502)** · 3~9원자 39% · 10원자 이상 29% 이고,
#   2원자 조각 표적의 **고침률이 0.0%** 다. 2원자 조각 = 나이트로실 `M–N=O` · `N₂` 류이고
#   실물(`GOFYOQ`·`MENKAR`)에서 **우도도 ④ 상한 해도 `Double` 을 맞히는데 ⑤ 만 `Single` 로 내린다**
#   (EHT 목표 −3, 정답 조각 전하 −1). 오차는 92%가 −2 로 계통적이다.
#   ⚠️ 적합 임계가 아니라 **구조 게이트**다 — 새 파라미터를 적합하지 않는다.
EHTMINFRAG = int(os.environ.get("EHTMINFRAG", "0"))
# ★ `EHTSKIP` — ⑤ EHT 조각 전하 목표를 **이 조성의 조각에만** 적용하지 않는다 (2026-09-03).
#   조성 키 = 조각 원소를 정렬해 이어붙인 문자열(`NO` · `SS` · `CCHH`). 쉼표로 구분.
#   왜 크기(`EHTMINFRAG`)가 아니라 조성인가 — 실측(정답 배정 기준 · train · **표적으로
#   조건화하지 않은 전 조각**):
#       `CO`  22,298 조각 → 오류  1.7%   ← 2원자인데 거의 안 틀린다
#       `NO`     493 조각 → 오류 99.8%   (오차 −2 가 489/492)
#       `SS`     118 조각 → 오류 94.9%   (오차 +2 가 112/112)
#       `CCHH`   142 조각 → 오류 66.9%   (오차 +2 가 91/95)
#   ⇒ 2원자 조각 전체를 끄면(`EHTMINFRAG=3`) `CO` 22,298 까지 끈다 — CV 에서 실제로 손해였다
#     (`Double` .6913 → .6861 · `Σq_L` .7932 → .7908). 조성으로 좁힌다.
#   ⚠️ 적합 파라미터가 아니라 **측정으로 고른 조성 목록**이다. 오차가 조성마다 한 방향으로
#      고정이므로 "목표를 상수만큼 보정" 하는 대안도 있으나 그건 조성당 상수 1개가 늘어난다.
EHTSKIP = {v for v in os.environ.get("EHTSKIP", "NO,SS,CCHH").split(",") if v}  # ★ 채택 2026-09-03
# ★ `LPCOND` — 4클래스 우도의 **사전확률을 끝점 내부차수로 조건화한다** (2026-09-03).
#   `lp[c] = ln P(c | 원소쌍)`  →  `ln P(c | 원소쌍, (deg_x, deg_y))`. `med`·`scl` 은 안 건드린다.
#   셀 표본이 `LPCOND_NMIN` 미만이면 **원소쌍 전역으로 폴백**한다. 차수는 T1(내부 결합,
#   CV F1 1.0000)에서 나오므로 정답지 누출이 아니다.
#   왜 — `Double`→`Single` 오답의 마진은 거리항 +0.79 를 사전확률 −1.13 이 상쇄한 것이었다.
#   `C–O` 전역은 `Single .417 / Double .111` 인데 셀 `deg(C)=3, deg(O)=1`(카보닐)에서는
#   `.344 / .322` 로 벌점이 사라진다.
#   ⚠️ `LPCOND_NOCONJ` — `Conj`(클래스 3)만 전역 사전확률로 되돌린다. 조건화를 전 클래스에
#      걸면 `C–C` deg3–deg3 셀의 `P(Conj) = .908` 때문에 **`Double`→`Conj` 유출이 +505** 였다
#      (CV 실측 8,752 → 9,257). `Conj` 를 빼면 오히려 **8,549** 로 준다.
LPCOND = os.environ.get("LPCOND", "1") == "1"  # ★ 채택 2026-09-03
LPCOND_NOCONJ = os.environ.get("LPCOND_NOCONJ", "1") == "1"  # ★ 채택 2026-09-03
LPCOND_NMIN = int(os.environ.get("LPCOND_NMIN", "300"))
# 사전확률 온도 — `score = 거리항 + LPA·ln P(c)`. 1.0 = 현행 · 0.0 = `D_flat`(기각).
LPA = float(os.environ.get("LPA", "1.0"))
# 🔴 `EHTCOST` — EHT 조각 전하 목표를 **절대명령으로 다루지 않는다** (2026-09-03).
#   실측: EHT 목표가 정답 배정과 어긋나는 조각이 **5.8%(5,434)** 이고, 그 조각의 결합 오답률이
#   **8.6% vs 3.2%**(2.7배) · **59.7%가 오답 보유**(맞은 조각 11.0%). 초과 오답 ≈ **6,500 결합**.
#   `AFOKAH` 은 목표(−2, 정답은 0)를 맞추려 1.295 A `C=N` 포함 **6개**를 바꿨다.
#   ⇒ 목표를 맞추는 **우도 비용**이 임계를 넘으면 그 조각은 목표를 포기하고 되돌린다.
#   −1 = 무제한(옛 동작).
EHTCOST = float(os.environ.get("EHTCOST", "-1"))
# 🔴 `LNORM=1` — Laplace 로그사후확률의 **정규화항 `−log(2·scl)`** 을 넣는다 (2026-09-03).
#   지금 식은 그 항이 빠져 있어 **산포가 좁은 클래스가 보상을 못 받는다.** `C=O`(결합길이
#   분포가 좁다)가 그 수혜자다. 파라미터 0개 · 통계적으로 옳은 형태.
#   ⚠️ 방향이 과제마다 반대일 수 있다 — `Conj` 도 scl 이 좁아(`C–C` 0.0089) 같이 유리해진다.
#   실측(2026-09-03): 전량 적용은 **쌍마다 방향이 반대**다 — `C=O`→`Single` 오답 851→625
#   (−26.6%)로 노린 것은 되는데 `Conj` 과잉이 `C–C` +352 · `C–N` +285 · `C–O` +203 늘어
#   순효과 +783 악화. `Conj` 의 좁은 `scl` 은 실제 결합길이 분포가 아니라 **방향족 고리
#   길이가 균일해서 생긴 인공물**이라 Laplace 보상을 주는 것이 애초에 틀렸다.
#   ⇒ `LNORM=2` — 정규화항을 **S/D/T 에만** 적용하고 `Conj` 는 뺀다.
LNORM = os.environ.get("LNORM", "0")
LNORM_ON = LNORM in ("1", "2")
LNORM_SKIP_CONJ = LNORM == "2"
# ⛔ 미채택 — R3+R4 위에서 이득이 폴드 분산 안이다(Double +0.0014 · x2m판 +0.0018).
#   단독으로는 +0.059 지만 R3·R4 와 **같은 오답을 노린다**. 파라미터 1개를 더 쓸 근거가 없다.
USE_ROP = os.environ.get("USE_ROP", "0") == "1"
ROPW = float(os.environ.get("ROPW", "1.0"))
_LP_DEG = {"O": 2, "S": 2, "Se": 2, "N": 3, "P": 3}

NAMEEL = {
    "Ti": "titanium",
    "Zr": "zirconium",
    "Hf": "hafnium",
    "V": "vanadium",
    "Nb": "niobium",
    "Ta": "tantalum",
    "Cr": "chromium",
    "Mo": "molybdenum",
    "W": "tungsten",
    "Mn": "manganese",
    "Re": "rhenium",
    "Fe": "iron",
    "Ru": "ruthenium",
    "Os": "osmium",
    "Co": "cobalt",
    "Rh": "rhodium",
    "Ir": "iridium",
    "Ni": "nickel",
    "Pd": "palladium",
    "Pt": "platinum",
    "Cu": "copper",
    "Ag": "silver",
    "Au": "gold",
    "Zn": "zinc",
    "Sc": "scandium",
    "Y": "yttrium",
    "La": "lanthanum",
    "Ce": "cerium",
    "B": "boron",
    "Al": "aluminium",
    "Ga": "gallium",
    "In": "indium",
    "Sn": "tin",
    "Pb": "lead",
    "Mg": "magnesium",
}
ALT = {
    "Fe": ["ferr"],
    "Cu": ["cupr"],
    "Au": ["aur"],
    "Ag": ["argent"],
    "Sn": ["stann"],
    "Pb": ["plumb"],
    "Ni": ["nickel"],
    "Pt": ["platin"],
    "Mn": ["mangan"],
    "Al": ["alumin"],
}
ROMAN = {"0": 0, "i": 1, "ii": 2, "iii": 3, "iv": 4, "v": 5, "vi": 6, "vii": 7, "viii": 8}
R = r"(?:0|i{1,3}|iv|vi{0,3})"
PAT, PATM = re.compile(rf"([a-z]+)\(({R})\)"), re.compile(rf"([a-z]+)\(({R}(?:,{R})+)\)")

# ★ T5 — haptic 판정 각도 임계 (전역 1개 · 설계도 §3 4a).
#   원래 이 상수는 채점 스크립트(`260831_propagation_prior_cv.py`)에 있었다.
THETA_HAPTIC = 81.02

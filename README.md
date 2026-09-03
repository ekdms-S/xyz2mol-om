# xyz2mol-om

**전이금속 착물의 `xyz` 좌표에서 결합 · 결합차수 · 리간드 전하 · 금속 산화수를 낸다.**
유기금속(organometallic)까지 다루도록 만든 것이라 이름이 `-om` 이다.

```python
from xyz2mol_om import predict

r = predict(elements, coords, total_charge=-1, wbo=wbo)
```

반환은 **금속별 / 리간드별**로 묶인 dict 다:

```python
r["metals"]  == [{"index": 0, "element": "Mo", "oxidation": 6, "mm_bonds": {}}]

r["ligands"] == [
  {"index": 0, "atoms": [1],
   "bonds_4class": {},                      # {(i,j): "Single"|"Double"|"Triple"|"Conj"}
   "bonds_kekule": {},                      # {(i,j): 1|2|3}   ⑥ 출력 변환기
   "smiles": "[N-3:1]", "smiles_ok": True, "smiles_note": "",
   "coordinating": [1],
   "ml_bonds": {(0, 1): {"type": "sigma", "order": 3}},   # type ∈ sigma|haptic
   "eta": {},                               # {금속: k}  (haptic 일 때)
   "charge": -3,                            # 리간드 전하 q_L
   "residual_charge": None},                # 골격으로 표현 안 되는 잔여 전하
  … ]
```

**실제 출력** — `[Mo(≡N)(OH)Cl₃]⁻` (Mayer 는 xtb `--sp --wbo` 실측):

```
metal  Mo0   산화수 +6
ligand [N1]      charge −3   [N-3:1]     M–L Mo0–N1  sigma  order=3    ← ≡N 나이트라이도
ligand [O2 H3]   charge −1   [H][O-:1]   M–L Mo0–O2  sigma  order=1    ← 하이드록소
ligand [Cl4]     charge −1   [Cl-:1]     ×3
```

🔴 **SMILES 는 우리 차수·전하를 그대로 고정해서 만든다** — RDKit 이 배위 원자에 암묵적 수소를
붙이거나 형식전하를 다시 매기지 못하게 잠그고(`SetNoImplicit`·`SetFormalCharge`·부분 sanitize),
**왕복 검증**(조성·차수 다중집합·전하 합·H 개수·화학적 타당성)을 통과해야 `smiles_ok=True` 다.
실측 통과율 **99.4%**(리간드 974 · train 200구조). 실패하면 `smiles_note` 에 사유가 담기니
**그 리간드는 SMILES 대신 `bonds_kekule` 을 쓴다.**

`elements` = 원소기호 리스트 · `coords` = `(N,3)` 좌표(Å) ·
`wbo` = `{(금속 인덱스, 원자 인덱스): Mayer 결합차수}` (xtb `--sp` 산출물).

⚠️ **`wbo` 없이도 돌지만 M–L 판정이 거리만 쓰고 차수는 전부 `Single` 이 된다** — M–L 다중결합
(옥소 · 이미도 · 나이트라이도)이 필요하면 반드시 넣는다.

## 무엇으로 판정하나

거리 · 각도 · Mayer 결합차수 · 확장 Hückel 조각전하 **네 가지 신호**와 **화학 규칙 5개**만
쓴다. 학습된 신경망이 없고, 적합된 값은 전부 텍스트 파일로 실려 있다
(`src/xyz2mol_om/data/`). 판정식 전문과 성능은 **[docs/PIPELINE.md](docs/PIPELINE.md)**.

| 단계 | 무엇 | 신호 |
|---|---|---|
| T1 | 배위자 내부 결합 유무 | 원소쌍별 거리 임계 |
| T3 | 내부 결합 **차수** {Single, Double, Triple, Conj} | 거리 우도 + 원자가 상한 + 규칙 A·R2~R5 + EHT 조각전하 |
| T4 | M–X 결합 유무 | 거리 + Mayer 거부권 |
| T5·T6 | haptic 판정 · η^k | 각도 (전역 임계 1개) |
| T8 | M–L 결합 차수 | Mayer 단조 임계 |
| T10 | 리간드 전하 · 산화수 | 형식전하 셈 (파라미터 없음) |

## 성능

CSD 유래 34,087 구조에서 적합/평가했다. **holdout 6,456 구조**(적합에 쓰지 않은 분):

| | 값 |
|---|---|
| T1 내부 결합 유무 | F1 **0.9998** |
| T3 `Single` / **`Double`** / `Triple` / `Conj` | .9900 / **.7258** / .9822 / .9631 |
| T4 M–L 결합 유무 | F1 **0.993** |
| T5 haptic | F1 **0.975** |
| T6 η^k (리간드 단위 정확 일치) | **0.986** |
| 리간드 전하 `Σq_L` (구조 단위 정확 일치) | **80.9%** |
| 금속 산화수 `OS` | **81.3%** |

외부 도구 대조와 상한(무엇이 남았나)은 [docs/PIPELINE.md](docs/PIPELINE.md) §성능.

## 설치

```bash
pip install -e .          # 개발
pip install .             # 사용
```

의존성은 `numpy` · `networkx` · `rdkit` 뿐이다. **xtb 는 필요 없다** —
Mayer 결합차수를 넣을 때만 별도로 돌리면 된다.

## 검증

```bash
python tests/test_api_smoke.py                   # 공개 API — 워크스페이스 없이 돈다
python tests/test_regression_vs_workspace.py     # 원본과 결합 단위 일치 확인 (연구 환경에서만)
```

이 저장소는 `ognm-bh-workspace` 의 연구 코드에서 이관한 것이다. 회귀 테스트는 원본이 있는
환경에서만 돌고, 없으면 자동으로 건너뛴다.

## 라이선스 · 출처

적합에 쓴 정답지는 CSD(Cambridge Structural Database) 결합 라벨과 tmQMg-L 리간드 전하다.
**데이터 자체는 이 저장소에 없다** — 실린 것은 적합 결과(임계값 · 우도 파라미터)뿐이다.

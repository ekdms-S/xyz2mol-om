# xyz2mol-om

**전이금속 착물의 `xyz` 좌표에서 결합 · 결합차수 · 리간드 전하 · 금속 산화수를 낸다.**
유기금속(organometallic)까지 다루도록 만든 것이라 이름이 `-om` 이다.

## 쓰는 법

```bash
PYTHONPATH=<repo>/src python your_script.py
```

```python
from xyz2mol_om import predict

r = predict(elements, coords, total_charge=-1, wbo=wbo)
```

| 인자 | 설명 |
|---|---|
| `elements` | 원소 기호 리스트 |
| `coords` | `(n, 3)` 좌표 (Å) |
| `total_charge` | 착물 총전하. 없으면 산화수·착물 SMILES 를 안 낸다 |
| `wbo` | `{(금속 idx, 원자 idx): Mayer 결합차수}` — xtb GFN2 `--sp --wbo` 산출물 |

⚠️ **`wbo=None` 으로 돌려도 된다** — M–L 판정이 거리 기반 폴백으로 떨어지고 **성능이 소폭
감소**한다(M–L `Double` F1 0.73 → 0.70 · 결합 유무 F1 0.973 → 0.964 · 내부 결합은 거의 불변).

## 출력

```python
r["metals"]  == [{"index": 0, "element": "Mo", "oxidation": 6, "mm_bonds": {}}]

r["ligands"] == [
  {"index": 0, "atoms": [1],
   "bonds_4class": {},         # {(i,j): "Single"|"Double"|"Triple"|"Conj"}
   "bonds_kekule": {},         # {(i,j): 1|2|3}
   "smiles": "[N-3:1]", "smiles_ok": True, "smiles_note": "",
   "coordinating": [1],
   "ml_bonds": {(0, 1): {"type": "sigma",   # sigma | haptic | bridge
                         "order": 3,        # haptic 이면 None
                         "bridge": None}},  # 다리면 "3c2e" | "dative"
   "eta": {},                  # {금속: k}
   "charge": -3,               # 리간드 전하 q_L
   "residual_charge": None},   # 골격으로 표현 안 되는 잔여 전하
  … ]

r["complex_smiles"] == "[H][O-]->[Mo+6](<-[N-3])(<-[Cl-])(<-[Cl-])<-[Cl-]"
```

### SMILES 형식

| | 규약 |
|---|---|
| 리간드 SMILES | 배위 원자에 원자 맵 `[X:n]` · 우리 차수·전하를 그대로 고정한다 |
| 착물 SMILES | M–L 은 **전부 dative 화살표**(`->`) · 금속 형식전하 = **산화수** |
| 차수 | 착물 SMILES 에서는 뭉갠다 — 실제 값은 `ml_bonds[(m,x)]["order"]` |
| 검증 | `smiles_ok` = 왕복 검증(차수·전하·H·다중집합) 통과 여부 · 실패 사유는 `smiles_note` |

### 착물 되조립

`complex_smiles` 하나로 착물 전체를 읽을 수 있고, 원자 대응은 `complex_atom_order`
(SMILES 출력 순서 → 입력 원자 인덱스)에 있다.

리간드 단위 출력으로 직접 조립하려면 `assemble_complex` 를 쓴다:

```python
from xyz2mol_om import predict, assemble_complex

mol, atom_map = assemble_complex(predict(el, xyz, total_charge=0, wbo=wbo))
# atom_map : {입력 원자 인덱스 -> mol 원자 인덱스}  (금속 · 배위 원자)
# ml_dative=False 로 부르면 M–L 을 ml_bonds[...]["order"] 정수 차수로 넣는다
```

⚠️ **리간드 SMILES 의 원자 맵 `[X:n]` 은 입력 인덱스가 아니다** — 그 리간드 `coordinating` 을
정렬한 목록의 **n 번째(1-based)** 다. `assemble_complex` 가 그 변환을 한다.

#### 직접 조립할 때 (다른 툴체인에서)

```
① 금속 원자를 만든다            형식전하 = metals[i]["oxidation"]
② 리간드 SMILES 를 파싱해 붙인다  각 ligands[j]["smiles"]
③ 맵 번호를 입력 인덱스로 되돌린다  입력 인덱스 = sorted(ligands[j]["coordinating"])[n-1]
④ M–L 을 잇는다                 ligands[j]["ml_bonds"] 의 키 (금속 인덱스, 입력 원자 인덱스)
                               차수는 값의 "order" · 하프틱이면 None ⇒ dative 로
⑤ M–M 을 잇는다                 metals[i]["mm_bonds"] 의 키 (금속, 금속)
```

```python
coord = sorted(lg["coordinating"])
for at in Chem.MolFromSmiles(lg["smiles"], sanitize=False).GetAtoms():
    n = at.GetAtomMapNum()
    if n:
        input_idx = coord[n - 1]          # ← ③ 이 한 줄이 핵심이다
```

⚠️ 리간드 SMILES 는 **암묵적 수소를 쓰지 않는다**(`xyz` 의 H 가 실제 원자로 들어 있다) —
파싱할 때 `sanitize=False` 로 읽고 `SANITIZE_KEKULIZE`·`SANITIZE_SETAROMATICITY` 를 뺀 채로
sanitize 해야 차수·전하가 그대로 유지된다.
⚠️ 같은 원자가 금속 2개에 붙는 다리 리간드는 `ml_bonds` 에 **키가 두 개**다(`(m1,x)`·`(m2,x)`).

## 예시 — `examples/`

실물 CSD 구조 5종. 각 예시는 네 벌이다 — `<이름>.xyz`(좌표) · `<이름>.wbo.json`(총전하 +
Mayer 결합차수) · `<이름>.result.json`(**파이프라인 출력 전량**) · `<이름>.png`(Kekulé 2D 그래프).

```bash
python examples/run_examples.py            # 5종을 돌려 <이름>.result.json 을 다시 쓴다
python examples/run_examples.py 02         # 이름에 "02" 가 든 것만
python examples/run_examples.py --no-wbo   # wbo 없이 (거리 폴백)
python examples/draw_examples.py           # PNG 다시 그리기
```

| # | 파일 | 실물 | 보여주는 것 | 출력 | 그림 |
|---|---|---|---|---|---|
| ① | `01_dative_os_carbonyl` | `fac-[Os(CO)₃Cl₃]⁻` | σ-dative 만 · 내부 `C≡O` | [json](examples/01_dative_os_carbonyl.result.json) | [png](examples/01_dative_os_carbonyl.png) |
| ② | `02_haptic_cp_ticl3` | `CpTiCl₃` | η⁵ 하프틱 | [json](examples/02_haptic_cp_ticl3.result.json) | [png](examples/02_haptic_cp_ticl3.png) |
| ③ | `03_bridge_ag2cl4` | `[Ag₂Cl₄]²⁻` | μ-Cl 다리(`bridge:dative`) · 금속 2개 | [json](examples/03_bridge_ag2cl4.result.json) | [png](examples/03_bridge_ag2cl4.png) |
| ④ | `04_3c2e_gallium_bh4` | `Me₂Ga(BH₄)` | 3c2e 다리 H · `B` 가 리간드 원자 | [json](examples/04_3c2e_gallium_bh4.result.json) | [png](examples/04_3c2e_gallium_bh4.png) |
| ⑤ | `05_mm_quadruple_re2cl8` | `[Re₂Cl₈]²⁻` | M–M 결합 | [json](examples/05_mm_quadruple_re2cl8.result.json) | [png](examples/05_mm_quadruple_re2cl8.png) |

결과를 직접 저장하려면 `save_json(r, path)` · 읽으려면 `load_json(path)` 를 쓴다
(결합 키 `(i, j)` 를 `"i,j"` 로 바꿔 담고 읽을 때 되돌린다).

## 성능

holdout **6,456 구조**(적합에 안 쓴 분) · 정답지 CSD `bond_type` + tmQMg-L `q_ligand`.

| 과제 | 지표 | 값 | 자명 기준선 |
|---|---|---|---|
| T1 리간드 내부 결합 유무 | F1 | **0.9998** | 전부 결합 .7306 |
| T2 공액 판정 | F1 | **0.9583** | — |
| T3 내부 차수 `Single`/`Double`/`Triple`/`Conj` | F1 | **.9894 / .7187 / .9826 / .9583** | 전부 `Single` .9097 / 0 / 0 |
| T4 M–L·M–M 결합 유무 | F1 | **0.9904** | 전부 결합 .5276 |
| T5 하프틱 판정 | F1 | **0.9768** | 전부 haptic .6766 |
| T6 η^k (리간드 정확 일치) | 정확도 | **0.9858** | 전부 `k=0` .8704 |
| T8 M–L 차수 `Single`/`Double`/`Triple` | F1 | **.9931 / .7398 / .6336** | — |
| T10 리간드 전하 `Σq_L` (구조 정확 일치) | 정확도 | **0.8338** | 상한 83.4% |
| T10 금속 산화수 `OS` (구조 정확 일치) | 정확도 | **0.8464** | 상한 85.6% |

### 원자가 위반 — 출력의 화학적 유효성

| 평가 | 풀 | 위반 구조 | 정답지 기준선 (같은 셈) |
|---|---|---|---|
| holdout | 6,456 구조 | **164 = 2.54%** | 44 = 0.68% |
| train CV | 26,075 구조 | **707 = 2.71%** | 103 = 0.40% |

## ⚠️ 한계

- **라디칼을 지원하지 않는다** — 홀전자를 적을 수단이 없어 **오류 없이** 가장 가까운 닫힌 껍질 답이 나간다.
- **M–M 차수를 안 낸다** — 결합 유무만 내고 차수는 `1` 로 둔다(예시 ⑤ 의 `[Re₂Cl₈]²⁻` 는 실제로 사중결합이다).
- **3c2e·클러스터**는 2중심 형식 밖이다 — 다리 H 가 있는 리간드는 SMILES 왕복 검증을 **일부러 거부**하고, 카보란 케이지의 조각 전하는 EHT 값을 쓴다.

판정 규칙·유도·측정 이력은 라이브러리 밖이다 —
`ognm-bh-workspace/docs/backlog/tm-bond-remaining.md`(설계도) ·
`docs/analysis/2026-09-03-t3-tuning-history.md`(채택 이력) · 이 저장소의 `docs/PIPELINE.md`.

## 라이선스 · 출처

적합에 쓴 정답지는 CSD(Cambridge Structural Database) 결합 라벨과 tmQMg-L 리간드 전하다.
**데이터 자체는 이 저장소에 없다** — 실린 것은 적합 결과(임계값 · 우도 파라미터)뿐이다.

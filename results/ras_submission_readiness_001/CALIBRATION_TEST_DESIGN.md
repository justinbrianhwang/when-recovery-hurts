# Calibration · Final Test 설계 (Task A1–A5)

상태: **설계·구현 완료, 실행 안 함.** Amendment 017 최종안이 승인되지 않았고, 유료 실행 승인도 없습니다.

## 1. 후보와 비후보

| Arm | Calibration 후보? | 이유 |
|---|---|---|
| `VISION_CENTER_V2` | **예 (유일)** | C1 Development: rescue/harm 14/1, 16/0; H_gen·H_mem 귀무가설 기각; 개발 안전 기준 충족 |
| `CENTER_ORACLE` | 아니오 | 시뮬레이터 큐브 위치를 읽는 특권 참조 |
| `VISION_CENTER_V2S` | 아니오 | C1 준비에서 사전 기준으로 제외, 미실행 |
| `V2_SINGLE_FRAME`, `V2_NO_HAND/BASE/HOLD_MEMORY` | 아니오 | Ablation(진단) |
| `V2_RULES_OFF` | 아니오 | 진단용 비교군 |
| `VISION_CENTER_V1` | 아니오 | 개발 자료에서 V2보다 열등, family만 키움 |

Calibration에서는 장면마다 **C0와 V2만** 실행합니다(같은 snapshot, 같은 세션). 다른 arm을 함께 돌리면 비용만 늘고
인증 대상이 아니므로 넣지 않습니다.

## 2. 해악 인증 (Calibration, 주 결과)

- 정의: H = P(Y_V2 = 0 | Y_C0 = 1), ε = 0.05, δ = 0.05.
- 정책별: n₊ = C0 성공 장면 수, k = 그중 V2 실패 수, `p = BinomialCDF(k; n₊, ε)` (H0: H > ε).
- Family = {(V2, seed1), (V2, seed2)}, Holm at δ. n₊ = 0 불통과.
- 주장: 각 통과 정책에 대해 "calibration 분포에서 V2의 모집단 해악률 ≤ 0.05 (잘못된 인증 확률 ≤ 0.05)".
  두 정책 모두 통과 시에만 공동 주장. 개별 상태의 안전, OOD, 실물은 주장하지 않음.

### 검정력 (`calibration_power_table.json`)

| 정책 (기대 n₊) | family | 통과하는 최대 k | 진짜 해악률 0.6% | 1% | 2% | 3% | 4% |
|---|---|---|---|---|---|---|---|
| seed 1 (342) | 2 | 9 | 1.00 | 0.997 | 0.848 | 0.424 | 0.120 |
| seed 2 (328) | 2 | 8 | 1.00 | 0.994 | 0.786 | 0.348 | 0.090 |
| seed 1 (342) | 1 | 10 | 1.00 | 0.999 | 0.915 | 0.550 | 0.192 |
| seed 2 (328) | 1 | 9 | 1.00 | 0.998 | 0.874 | 0.476 | 0.153 |

C1 Development에서의 V2 해악: 1/171(seed 1), 0/164(seed 2). 이 값은 개발 자료이며 보증이 아닙니다.
진짜 해악률이 개발 자료 수준(≈0.3–0.6%)이면 통과 확률은 사실상 1이고, 3% 이상이면 인증이 어렵습니다.
0건일 때 필요한 n₊는 family 1에서 59, family 2에서 72입니다.

## 3. 효용 (Final Test, 주 결과, 별도 family)

- HA: 각 정책에서 V2 성공 장면 수 > C0. exact one-sided paired test(불일치 장면), Holm over 2 policies, α = 0.05.
- Test 장면의 해악률도 같은 방식으로 계산해 보고하지만, 보증은 Calibration 인증에서만 나옵니다.
- C1 Development 기준(참고): V2 − C0 순이득 13/200, 16/200. 효과가 이 정도면 검정력이 충분하고, 절반이면 크게 떨어집니다.

## 4. 데이터

| 집합 | seed | 감사 결과 |
|---|---|---|
| Calibration | 30000–30399 | 미사용(seed 수준 사용 0, ledger 없음) |
| Final Test | 90000–90399 | 미사용, 잠금 |

- Placement 검사(정책 없이 초기 큐브 배치만)는 Calibration 실행 체인의 첫 단계입니다. 기존 규칙대로 지금은 두 집합을
  reset하지 않았습니다. 기록된 모든 배치(시연, 이전 파일럿, C1 dev 포함)와 비교하며, 같은 배치가 있으면 실행하지 않습니다.

## 5. 실행 순서 (승인 후)

1. 사용자: Amendment 017 최종안 승인 기록 작성, 유료 실행 승인(상한 포함).
2. 서버 설치 → `pip freeze` 저장 → `scripts/40_v2_confirm.py freeze --stage calibration` (manifest commit·push·Drive 복사).
3. 실환경 smoke: 사용된 Development 장면 3개 × 2 정책으로 runner를 실제 시뮬레이터에서 실행(검증 전용 출력 폴더, ledger 없음 —
   `--seeds`는 calibration/test seed를 거부). C1 기록과 trace·결과 일치 확인.
4. `placement` (calibration) → `run --policy train_seed_1`, `run --policy train_seed_2` → `analyze`.
5. 결과 보고. 이후 `freeze --stage test`(Calibration 분석·manifest 포함) → 사용자의 `RUN_FROZEN_TEST` 제공 시에만 Test.

## 6. 비용 추정 (RTX 4090, Vast, ≈ US$0.35/h)

C1 실측: 400 scene-runs, 9 arms, 평균 22.6초/scene-run(spatial logging 포함). C0+V2 2 arm으로 줄이면 ≈ 10초로 추정
(trunk ≈ 6초 + branch 2개 × ≈ 1.9초; 가정이며 smoke에서 측정).

| 단계 | rollouts | 예상 시간 | 예상 비용 | 상한 제안 |
|---|---|---|---|---|
| 설치·smoke·placement | 6 scene-runs + reset | 0.6 h | US$0.25 | — |
| Calibration | 400 장면 × 2 정책 = 800 scene-runs (1,600 arm branch) | 2.2 h | US$0.80 | 합계 US$2.00 |
| Final Test | 동일 | 2.2 h + 0.4 h | US$0.95 | US$2.00 |

저장량: C1이 400 scene-runs에 4.2 GB(9 arms, 프레임 포함) → 2 arms 800 scene-runs ≈ 3–4 GB.

## 7. 실패 시 절차

- 장면 오류: `<seed>.error.json` 기록 후 pass 중단, INCOMPLETE. 재개 없음. 분석은 부분 실행을 거부.
- 호스트 오류·중단: 원인·관측된 결과·수정 범위를 개정 문서로 기록하고 사용자 승인 아래 새 집합으로 재설계.
  ledger·marker를 지우지 않음.
- Calibration 결과를 보고 후보·규칙·임계값·검정을 바꾸지 않음.

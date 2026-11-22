# Amendment 017 — 최종 제안안 (승인 전, 효력 없음)

작성일: 2026-11-22 (UTC). 상태: **사용자 승인 대기. 승인 전에는 어떤 효력도 없습니다.**
초안 `amendments/AMENDMENT_017_vision_center_protocol_DRAFT.md`(2026-11-18/19)를 C1 Development 결과와 사후
진단(물리 진단, S4/S5 hull 민감도)에 맞춰 확정 가능한 형태로 정리한 것입니다. 초안 파일은 수정하지 않았습니다.

승인 방법: 사용자가 이 파일의 SHA256을 적은 승인 기록 `approval_amendment_017.json`
(`{"approved_by": "user", "amendment_sha256": "<이 파일의 sha256>", "scope": [...]}`)을 직접 작성합니다.
이 저장소의 어떤 코드도 그 파일을 만들지 않으며, `scripts/40_v2_confirm.py freeze`는 그 파일이 없으면 거부합니다.

---

## 1. 대상과 질문 (v0.3 대비 변경)

| 항목 | v0.3 | 이 개정 |
|---|---|---|
| 개입 | Continue / Refresh / Retreat–Retry, 학습된 선택기 | 트리거 시 1회의 수평 centring 이동 또는 기권(관측 기반 규칙 제어기, 학습 없음) |
| 기본 정책 | Stage 1 정책 | Vision-V1 ResNet-18 DP, train seed 1 (`a37be256…`), train seed 2 (`52eff6a9…`) — 둘 다 80k EMA, 동결 |
| 후보 | 19개 선택기 설정 | **`VISION_CENTER_V2` 1개** |
| 동결·평가 코드 | `hci freeze` / `evaluate` / `frozen-test` | `src/hci/v2_confirm.py`, `scripts/40_v2_confirm.py` (모든 기존 거부 조건 계승) |

유지: StackCube-v1, `panda_wristcam`, CPU physics, `success_once_absorbing`, 200 step horizon,
`H = P(Y_g = 0 | Y_0 = 1)`, ε = 0.05, δ = 0.05, exact binomial + Holm, seed 단위 split, 1회 개입, 특권 입력 금지,
`RUN_FROZEN_TEST` 토큰.

연구 질문: 관측만 쓰는 고정 centring 이동이 동결된 조작 정책의 성공률을 높이면서, 성공을 실패로 바꾸는 비율을
ε 이하로 유지하는가?

## 2. 가설과 역할 분리

| 가설 | 내용 | 데이터 | 검정 |
|---|---|---|---|
| HB (해악 인증, 주) | 각 정책에서 P(V2 실패 \| C0 성공) ≤ ε | **Calibration 30000–30399** | exact binomial, Holm over family |
| HA (효용, 주) | 각 정책에서 V2가 C0보다 많은 장면에서 성공 | **Final Test 90000–90399** | exact one-sided paired test, Holm over 2 policies (α = 0.05) |
| HB-test (보고) | Test 장면에서의 해악률 | Final Test | 같은 계산, 보증 아님(기술적 재확인) |
| HC, HD | 메모리 기여, 2번째 seed 일반화 | C1 Development에서 이미 검정(H_mem, H_gen) | 재검정하지 않음 |

- 해악 인증(HB)과 효용 검정(HA)은 서로 다른 family이며 섞지 않습니다.
- HB family = {(V2, train_seed_1), (V2, train_seed_2)} — **기본 권고**. 두 정책 모두 통과할 때만
  "두 정책 모두에서 해악률 ≤ ε" 공동 주장을 합니다(Holm이 개별 주장의 FWER를 δ로 통제하므로 공동 주장도 유지).
  한 정책만 통과하면 그 정책에 대한 주장만 합니다.
- 대안(사용자 선택 사항): 정책별로 따로 인증(family 크기 1 × 2개, 공동 주장 없음).

## 3. 해악 인증 절차 (Calibration)

1. 동결(§6) → placement 검사(정책 없이 초기 큐브 배치만, 기록된 모든 배치와 비교) → 정책별 1회 실행.
2. 장면마다 같은 세션·같은 호스트에서 하나의 snapshot으로 C0와 V2를 실행(`hci.vision_center_c1.collect_scene`,
   arms = (C0, V2)). 기권한 V2는 C0와 같은 궤적이어야 하며 다르면 오류로 중단.
3. 정책별 n₊ = C0 성공 장면 수, k = 그중 V2 실패 수. `p = BinomialCDF(k; n₊, ε)`. Holm(δ) over family.
   n₊ = 0이면 통과하지 않음.
4. 결과와 무관하게 후보·정책·ε·δ·규칙·검정을 바꾸지 않음. 바꾸면 그 Calibration은 개발 자료가 되고 새 보정 집합이 필요.
5. 함께 보고(보증 없음): V2가 움직인 C0 성공 장면에서의 해악, 기권 사유, 메모리 오류, 목표 밀림(>1 mm), Check 4/7.

## 4. Final Test

- Calibration 후 별도 동결(Calibration 분석 결과와 Calibration 동결 manifest의 SHA256 포함).
- 사용자가 `RUN_FROZEN_TEST`를 명시적으로 제공할 때만 실행. 정책별 `TEST_STARTED_<policy>.json`과 단일 사용 ledger를
  첫 rollout 전에 배타적으로 생성(두 번째 생성 실패).
- 1회만 실행. 불리해도 재실행·설정 변경 없음. 부분 실행은 요약하지 않고 그대로 보고.

## 5. 데이터 사용

| 집합 | seed | 상태(2026-11-22 감사) | 용도 |
|---|---|---|---|
| Development | 50000–51099, 51100–51299 | 사용됨 | 추가 증거 용도 없음 |
| Calibration | 30000–30399 | 미사용(6,198개 파일 구조 검사, 사용 0) | 1회 보정 |
| Test | 90000–90399 | 미사용, 잠금 | 1회 최종 Test, 명시 승인 후 |

일반화 실험(별도 실험 ID)은 새 Development seed를 따로 선언하며 위 두 집합을 쓰지 않습니다.

## 6. 동결 대상 (manifest에 SHA256)

이 개정안과 승인 기록, 후보와 family, ε/δ/α, runner·분석 코드와 그 테스트, 재사용하는 수집·제어 코드
(`vision_center_c1/v2/vision_center/recovery_v1_oracle/simulation/vision_v1/statistics/spatial_probe` 등),
두 정책의 weights/metadata, config, split 파일, upstream commit(`a4a4f92`), 실행 호스트의 전체 `pip freeze`.
초안 manifest: `FREEZE_MANIFEST_DRAFT.json`(동결 아님).

## 7. C1 및 사후 진단과의 정합성 (초안 대비 확정 사항)

| 초안 §9 항목 | 확정 제안 | 근거 |
|---|---|---|
| 1. 후보 목록 | V2 단독. V2S·oracle·ablation 제외 | V2S는 C1 준비에서 사전 기준으로 제외됨; oracle은 특권; ablation은 진단용 |
| 2. 정책 범위 | 두 정책, Holm family 2 | C1에서 H_gen·H_mem 귀무가설 기각, 두 정책 모두 개발 안전 기준 충족 |
| 3. 표본·검정력 | 400 장면 유지 | n₊≈342/328에서 k ≤ 9/8 통과; 진짜 해악률 1%면 통과확률 ≥ 0.99, 2%면 0.85/0.79 (`calibration_power_table.json`) |
| 4. Runner | 신규 runner 구현·테스트 완료(mock) | `CALIBRATION_RUNNER_VALIDATION.md` |
| 5. Placement | Calibration 실행 체인의 첫 단계 | 보호 규칙상 지금은 reset하지 않음 |
| 6. 호스트 | RTX 4090, 드라이버 고정, Y0/Yg 같은 세션 | Phase B에서 호스트 간 C0 결과 16/300 차이; rgb_replay_001에서 같은 GPU·드라이버로 완전 재현 |
| 7. 메모리 오류 | 보고만(제한 아님) | C1 V2 메모리 오류 0/0; 인증과 별개 |
| 8. docs 개정 | 승인 시 docs/00–06에 부록으로 반영 | — |

사후 진단과의 관계: S6 FAIL(측정 타당성)과 S4 1건, S5 11 branch FAIL은 **해악 정의(작업 결과)와 무관**합니다.
해악·구조·성공은 시뮬레이터의 성공 신호로만 판정하므로 S6의 재구성 오차가 HB/HA 판정에 들어가지 않습니다.
다만 접촉에 관한 서술은 impulse와 물체 변위에만 근거한다는 제한을 유지합니다.

## 8. 다루지 않음

실제 이미지, 다른 작업, 반복 개입, 학습된 선택기, 물리적 안전 주장. H는 시뮬레이션 작업 결과의 악화입니다.

## 9. 템플릿 항목

- 기존 버전: v0.3 + Amendments 001–016. 새 버전: v0.4 (승인 시).
- 변경 시점: C1 Development 후, Calibration 전. 본 결과: Development만. Calibration·Test 결과 없음.
- 무효화되는 이전 보정: 없음(보정 실행 이력 없음).
- 재실행할 검사: 로컬 테스트, runner 실환경 smoke(사용된 Development 장면, 유료 GPU 필요 — 별도 승인), placement 검사.

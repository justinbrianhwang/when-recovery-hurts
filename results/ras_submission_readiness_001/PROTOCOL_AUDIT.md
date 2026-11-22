# 프로토콜 감사 (Task 0)

작성일: 2026-11-22 (UTC). 읽은 파일(실제 내용 확인): `AGENTS.md`, `docs/00_SCOPE_LOCK.md`, `docs/01_PROTOCOL.md`,
`docs/02_DATA_AND_MODELS.md`, `docs/03_STATISTICAL_PROTOCOL.md`, `docs/05_STAGE_GATES.md`, `docs/06_FREEZE_AND_TEST.md`,
`src/hci/freeze.py`, `src/hci/statistics.py`, `configs/splits.json`, `configs/vision_center_c1_splits.json`,
`results/vision_center_v2_c1_001/preparation_v3/plan.json`, `amendments/AMENDMENT_017_vision_center_protocol_DRAFT.md`,
Drive `results/vision_center_v2_001/V2_CALIBRATION_READINESS.md`, `V2_FREEZE_MANIFEST_DRAFT.json`,
`results/vision_center_v2_c1_001/C1_STATISTICAL_ANALYSIS_PLAN.md`, `C1_DEVELOPMENT_EXECUTION_REPORT.md`.
Calibration·Test 데이터는 이 단계에서 전혀 사용하지 않았습니다(seed 수준 문자열 검사만).

## 1. 결론

1. **v0.3 프로토콜과 현재 연구 대상은 다릅니다.** v0.3은 Continue/Refresh/Retreat–Retry를 고르는 학습 선택기를
   다루고, 현재 대상은 관측 기반 규칙형 centring 제어기(V2)입니다. 기존 `hci freeze/evaluate/frozen-test`는 heads,
   registry, selection 파일을 요구하므로 V2에 쓸 수 없습니다(`freeze.py` 확인).
2. **Amendment 017은 초안이며 효력이 없습니다.** C1 실행 승인은 이 개정의 승인이 아니었습니다(초안 본문에 명시).
3. **초안은 C1 결과·사후 진단과 충돌하지 않습니다.** 미결 8개 항목에 대한 확정 제안을
   `AMENDMENT_017_FINAL_PROPOSAL.md`로 작성했습니다. 승인 전에는 효력 없음.
4. **Calibration(30000–30399), Test(90000–90399)는 미사용입니다**(`split_usage_audit.json`: 6,198개 파일, 사용 0,
   frozen 폴더 0, TEST_STARTED 0, 원장 7개 모두 다른 집합).

## 2. 기존 프로토콜 vs C1 대상

| 항목 | v0.3 (docs/00–06) | C1 Vision-CENTER V2 | 차이의 결과 |
|---|---|---|---|
| 개입 | 3가지 행동, 결정 시점 간격 4 step | 해제 직전 1회 트리거, 최대 12 step 수평 이동 | scope lock 밖 → 개정 필요 |
| 선택 | 학습된 Q/Joint 헤드 | 고정 규칙, 학습 없음 | registry·selection 개념 재정의 |
| 기본 정책 | Stage 1 정책 | Vision-V1 ResNet-18 DP seed 1, 2 | 정책 해시로 재선언 |
| 후보 | 19개 | V2 1개(제안) | Holm family 축소 |
| 위험 정의 | H = P(Yg=0\|Y0=1), ε=δ=0.05 | 동일 | 재사용 |
| 검정 | BinomialCDF + Holm | 동일 (`statistics.binomial_safe_p`, `holm_reject`) | 재사용, 방향 일치 확인(§3) |
| 동결·Test | `hci freeze`, `frozen-test`, `RUN_FROZEN_TEST`, `TEST_STARTED.json` | 신규 runner가 같은 거부 조건을 계승 | 구현·테스트 완료 |

## 3. 검정 방향과 귀무가설의 코드–문서 일치

- `docs/03`: `p = BinomialCDF(k; n+, ε)`, Holm으로 FWER δ, 통과 후보 중 개발 성공률 최고 선택, n₊=0 불통과.
- `statistics.binomial_safe_p(k, n, ε)` = `binom.cdf(k, n, ε)`, n=0이면 1.0 → 일치.
  해석: 귀무가설 H0: 해악률 > ε(정확히는 ≥ ε 경계), p가 작으면 "해악률 ≤ ε" 인증. 단측, 정확 검정.
- `holm_reject`: 오름차순 p를 δ/(m−j)와 비교, 처음 실패에서 중단 → 표준 Holm step-down. 일치.
- 신규 `v2_confirm.certificate`는 위 두 함수를 그대로 호출하고 n₊=0을 명시적으로 불통과 처리. 문서 일치.
- 효용 검정(C1 `exact_one_sided`, 신규 `v2_confirm.exact_one_sided`): 불일치 장면에서 P(X ≥ b), X~Bin(b+c, ½), 단측.
  해악 인증과 별개 family로 둠.

## 4. 두 정책: 공동 주장과 정책별 주장

- 두 정책은 같은 장면을 공유하므로 400 rollouts을 독립 400개로 합치지 않습니다(C1 SAP와 동일 원칙).
- 해악 인증 family = {(V2, seed1), (V2, seed2)}. Holm이 두 개별 주장의 FWER를 δ로 통제하므로, 두 개 모두 통과할 때의
  공동 주장("두 정책 모두 해악률 ≤ ε")도 δ 수준으로 보호됩니다. 공유 장면 의존성은 Holm에 영향을 주지 않습니다.
- 정책별 보고는 항상 따로 합니다.

## 5. Amendment 017 초안과 C1·사후 진단의 정합성

| 초안 내용 | C1·진단 결과 | 판단 |
|---|---|---|
| V2S 제외 | C1 준비에서 사전 기준으로 제외, C1에서 실행 안 함 | 일치 |
| Oracle은 후보 아님 | C1 oracle은 목표를 >1 mm 민 경우 7/4건, S4 FAIL 1건(51152) | 일치, 제외 근거 강화 |
| HC(메모리) 제한된 해석 | C1 H_mem 귀무가설 기각(교집합-합집합 검정), single-frame은 확인 수 1로 완화된 비교군 | 일치 |
| HD(2번째 seed) | C1 H_gen 귀무가설 기각(p=1.5e-5) | Calibration에서 두 정책 family로 이어짐 |
| 메모리 오류 보고 | C1 V2 0/0 | 보고만(제한 아님) 제안 |
| 호스트 의존성 | rgb_replay_001: 같은 GPU·드라이버에서 trace·프레임 완전 일치 | 호스트 고정 조건 확정 |
| 물리 측정 타당성 | S6 FAIL(유지), S4 1, S5 11 branch FAIL(hull 무관) | 해악 정의는 성공 신호 기반이라 인증과 독립; 접촉 서술 제한 유지 |

## 6. 승인에 필요한 사용자 결정 (기본 권고 포함)

| 결정 | 기본 권고 | 대안 |
|---|---|---|
| D1 후보 | V2 단독 | V2 + 다른 후보(새 개발 자료 필요) |
| D2 정책 범위 | 두 정책, Holm family 2, 공동 주장 조건부 | 정책별 family 1 |
| D3 Calibration 크기 | 400 장면 그대로 | 확대 시 보정 전 확정 필요 |
| D4 효용 검정 위치 | Final Test에서만, 정책별 Holm | Calibration에서도 기술 보고 |
| D5 메모리 오류 | 보고만 | 후보 제한 조건으로 사용 |
| D6 Amendment 017 최종안 승인 | — | 수정 요청 |

## 7. 프로토콜 개정이 아닌 운영상 확인

- `configs/splits.json`의 calibration·test 목록은 정확히 30000–30399, 90000–90399 (신규 테스트로 검증).
- 기존 placement 검사들은 calibration·test를 의도적으로 `seed_level_only`로 남김. 이 규칙을 따라 이번에도 reset하지
  않았고, placement 검사는 승인 후 Calibration 체인의 첫 단계로 넣음.

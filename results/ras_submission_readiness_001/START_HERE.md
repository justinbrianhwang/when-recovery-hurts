# RAS 투고 준비 001 — 시작 문서

작성일: 2026-11-22 (UTC). 대상 저널: Robotics and Autonomous Systems(Elsevier, subscription 경로).
**이번 작업에서 Calibration, Final Test, 학습, 새 장면 rollout, 유료 GPU는 실행하지 않았습니다.** 기존 C1 원본·결과·사후
진단은 바꾸지 않았습니다. Amendment 017은 여전히 승인 전이며 효력이 없습니다.

## 세 문제별 해결 수준

| 문제 | 실행·확보한 근거 | 남은 것 | 상태 |
|---|---|---|---|
| A. Development 한정 결과 → Calibration/Test | 프로토콜 감사; Amendment 017 최종안; 검정 방향 코드–문서 일치 확인; V2 전용 runner·분석 구현과 테스트(mock 끝-끝 포함); split 미사용 감사(6,198 파일, 사용 0); 검정력 표; 동결 manifest 초안 | Amendment 승인 → 동결 → 실환경 smoke → placement → Calibration 1회 → (별도 승인) Test 1회 | **실행 직전까지 준비 완료** |
| B. S6 측정 타당성 | 독립 통제 두 상자 실험(59 cases): 정지 상태 separation = 해석적 거리(≤ 3e-6 mm), 목록 기준 = offset 합 0.04 m, **separation은 마지막 substep 시작 자세 기준**(v·dt 차이 정확 재현), sleep 중 값 고정 | 정지·느린 손가락·카메라 잔차는 미해결 → substep 계측 replay(승인 필요) | **원인 2개 확정, 1개 미해결** |
| C. 일반화 | ManiSkill 태스크·정책 검토; 1순위 후보 선정(Stage 1D PlainConv 동결 정책, V2 무변경); 실험 계획·가설 동결용 plan; runner·테스트 | 본 실행(승인 필요) | **준비 완료, 미실행** |

## 파일

| 파일 | 내용 |
|---|---|
| `PROTOCOL_AUDIT.md` | v0.3 vs C1 대상, 검정 방향, 두 정책 주장, Amendment 017 정합성, 필요한 결정 |
| `AMENDMENT_017_FINAL_PROPOSAL.md` | 승인 가능한 최종안(효력 없음) |
| `CALIBRATION_TEST_DESIGN.md` | 후보, 해악 인증, 효용 검정, 검정력, 실행 순서, 비용 |
| `CALIBRATION_RUNNER_VALIDATION.md` | runner 기능과 테스트 대응표 |
| `FREEZE_MANIFEST_DRAFT.json` | 동결될 27개 파일 SHA256(초안, 동결 아님) |
| `S6_VALIDITY_EXTENSION_REPORT.md`, `s6_static_validation/` | B1 문서 확인, B2 통제 실험 결과, S6 분해, B3 계획 |
| `GENERALIZATION_CANDIDATE_REVIEW.md`, `GENERALIZATION_EXPERIMENT_PLAN.md` | 후보 평가, gen_arch_plainconv_001 설계 |
| `RAS_MANUSCRIPT_OUTLINE.md`, `RAS_FIGURE_AND_SUPPLEMENT_PLAN.md` | 원고·그림·부록 구성, 쓸 수 있는 영문 주장 |
| `APPROVAL_REQUIRED.md` | 승인 항목·비용 한 번에 |
| `split_usage_audit.json`, `calibration_power_table.json` | 감사·검정력 자료 |
| `logs/` | pytest 로그 |
| `SHA256_MANIFEST.json` | 이 폴더 모든 파일의 SHA256 |

코드: `src/hci/v2_confirm.py`, `scripts/40_v2_confirm.py`, `scripts/41_gen_arch_plainconv.py`,
`tests/test_v2_confirm.py`, `tests/test_gen_arch_plainconv.py`, `research_diagnostics/s6_static_validation_001/`,
`research_diagnostics/ras_readiness_001/`.

## 테스트

| 실행 | 결과 |
|---|---|
| `tests/test_v2_confirm.py` (신규 runner, 단위 6 + mock 끝-끝 1) | 7 passed |
| `tests/test_gen_arch_plainconv.py` (일반화 runner) | 2 passed (`logs/pytest_gen_arch.log`, exit 0) |
| 기존 `tests/test_statistics.py` | 19 passed |
| 기존 `tests/test_vision_center_c1.py` | 38 passed, **1 failed**: `test_driver_constants_registry_and_file_list`가 "C1 ledger가 아직 없어야 함(아직 실행 안 함)"을 단언하는데, C1이 2026-11-22에 실행되어 ledger가 정상적으로 존재함. 이번 변경과 무관한 오래된 단언이며, 이 테스트 파일은 C1 동결 plan의 해시 대상이라 수정하지 않음 |
| 합계 (`logs/pytest_runner_and_related.log`) | 64 passed, 1 failed, exit 1 (40분 45초) |
| 정적 PhysX 실험 (`s6_static_validation/run.log`) | exit 0, 59 cases |

## RAS 투고 가능성 (근거 기반 평가)

- 현재 근거만으로 쓸 수 있는 논문: 사전 등록된 Development 검정(H_gen, H_mem)과 메커니즘·물리 타당성 분석.
  RAS 범위에는 맞지만, **해악 "인증"이라는 핵심 주장은 Calibration 없이는 할 수 없습니다.** 심사에서 가장 먼저
  지적될 지점은 (1) 결과가 개발 자료뿐이라는 점, (2) 태스크 1개·같은 아키텍처 정책 2개라는 범위입니다.
- B1(Calibration)과 B3(일반화)를 실행하면 두 지적에 직접 답하는 자료가 생깁니다(합계 예상 ≈ US$1.6). Test(B2)까지
  마치면 사전 등록·보정·held-out의 완결된 구조가 됩니다.
- 결과가 불리하게 나와도(인증 실패, 일반화 효과 없음) 그대로 보고하며, 그 경우 논문의 주장은 "개발 자료에서의 효과와
  그 한계"로 좁아집니다.

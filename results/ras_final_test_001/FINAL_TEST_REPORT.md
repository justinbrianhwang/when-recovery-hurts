# Final Test 결과 — ras_v2_confirm_001 (test stage), 1회 실행

- **실행:** 2026-11-23 05:09–09:15 UTC. Test seed 90000–90399 × 정책 2개, 각 1회, 병렬. 오류 기록 0건.
- **승인:** 사용자가 직접 승인했고(`approvals/approval_final_test_2026-11-23.json`), 사용자 지시에 따라 에이전트가
  `RUN_FROZEN_TEST`를 입력했습니다.
- **동결 manifest:** sha256 `94c5a591554efa99f821dd2d7ea1099619afec790ee92f995e95ee8a92161c5f`. Test 전후 서버 사본의 해시가
  같았습니다.
- **변경 없음:** Test 결과를 본 뒤 모델, 정책, 제어기, 임계값, 후보, 가설, 분석 코드를 바꾸지 않았고 다시 실행하지
  않았습니다.

## 1. 실행 전 검사 (Test seed 미사용)

| 검사 | 결과 |
|---|---|
| 동결 파일 검증 | VERIFIED |
| 환경 검사 | 불일치 0건. RTX 4090, 드라이버 595.99.02, Python 3.10, ManiSkill `a4a4f92`, `pip freeze` 138줄 모두 동일 |
| 호스트 동등성 검사(사용된 C1 장면 51103/51157/51253 × 정책 2개) | 6/6이 Calibration 호스트 기준과 비트 단위로 같음 |
| Placement 검사(정책 미실행) | 400 seed, 참조 2,347개, 동일 배치 0개 |
| 시작 marker·ledger | 정책별 1개씩, 검사를 모두 통과한 뒤에 생성 |

## 2. 주 결과 — 효용(HA): exact one-sided paired test, Holm over 2 policies, α = 0.05

| 정책 | C0 성공 | V2 성공 | 차이 | V2만 성공 | C0만 성공 | p (one-sided) | Holm 기준 | 귀무가설 |
|---|---|---|---|---|---|---|---|---|
| train_seed_1 | 350/400 (87.5%) | 379/400 (94.75%) | +7.25 pp | 29 | 0 | 1.9×10⁻⁹ | 0.025 | **기각** |
| train_seed_2 | 347/400 (86.75%) | 373/400 (93.25%) | +6.5 pp | 27 | 1 | 1.1×10⁻⁷ | 0.05 | **기각** |

## 3. 해악 — Test 장면의 재확인(보고용; 보증은 Calibration 인증)

| 정책 | n₊ | 해악 k | 해악률(Wilson 95%) | p | V2가 움직인 C0 성공 중 해악 |
|---|---|---|---|---|---|
| train_seed_1 | 350 | 0 | 0% (0–1.1%) | 1.6×10⁻⁸ | 0/38 |
| train_seed_2 | 347 | 1 | 0.29% (0.05–1.6%) | 3.6×10⁻⁷ | 1/46 |

같은 절차로 계산하면 두 정책 모두 해악률 ≤ 5% 기준을 통과합니다. 다만 해악률에 대한 보증은 사전 등록한 대로
Calibration 인증(seed 1: 1/348, seed 2: 0/337)에서 나옵니다.

## 4. 독립 재계산

`test_recount.json`은 원시 장면 기록만으로 다시 계산한 결과입니다. hci를 쓰지 않았고, 모든 trace의 sha256을
확인했습니다. 해악 인증과 효용 검정 모두 서버 결과와 일치합니다(`all_equal_server: true`).

## 5. 비용

US$2.13(크레딧 4.4919 → 2.3614). 인스턴스 Vast 62097700, 4.64 h, $0.454/h. 상한은 US$3.00이었고, 인스턴스는
삭제해서 0개 남았습니다.

## 6. 해석 범위와 한계 (논문에 그대로 적을 것)

- **보증의 범위:** 결과는 StackCube, 시뮬레이터 이미지, Vision-V1 ResNet-18 정책 2개(같은 시연, 다른 학습 seed),
  그리고 RTX 4090 / 드라이버 595.99.02 / 고정된 소프트웨어 구성에 대한 것입니다.
- **호스트 구성 차이:** Development(C1)는 드라이버 550.120에서 실행됐습니다. Calibration과 Test는 595.99.02에서
  실행됐으며, 호스트 간에 일부 장면의 결과가 달랐습니다.
- **사전 기준 이탈:** smoke 결과가 C1 원본과 일치해야 한다는 사전 기준은 충족되지 않았습니다. 이는 기록되어 있고
  평가됐습니다(`ras_final_test_freeze_001/FINAL_TEST_FREEZE_AUDIT.md` §3).
- **H의 의미:** H는 작업 결과의 악화입니다. 물리적 안전에 대한 주장이 아닙니다. S6 측정 타당성 항목은 FAIL로
  유지됩니다.
- **일반화:** 다른 아키텍처(PlainConv)로의 일반화는 Development 수준의 결과입니다(gen_arch_plainconv_001).
  다른 태스크로의 일반화는 다루지 않았습니다.

## 7. 논문용 영문 문장

"On 400 held-out test scenes per policy, the unchanged V2 raised task success from 87.5% to 94.75% (train seed 1;
29 scenes solved only by V2, none only by the uninterrupted policy) and from 86.75% to 93.25% (train seed 2; 27 vs 1);
both one-sided exact paired tests rejected their null hypotheses under Holm at α = 0.05 (p = 1.9×10⁻⁹ and 1.1×10⁻⁷).
Harm on the test scenes was 0 of 350 and 1 of 347 baseline successes; the harm bound itself rests on the calibration
certificate."

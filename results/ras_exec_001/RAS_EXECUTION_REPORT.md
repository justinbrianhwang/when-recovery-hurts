# RAS 실행 보고 — B1 Calibration, B3 일반화, B4 S6 계측 (2026-11-22–23)

**승인:**
- Amendment 017 최종안(sha256 `b8fd6324…`)과 B1·B3·B4 — `approvals/approval_amendment_017.json`
- B1 상한을 $2.00에서 $2.50으로 상향 — `approvals/cap_change_B1_2026-11-22.json`
- 두 기록 모두 사용자의 명시적 지시로 에이전트가 작성했습니다.

**실행하지 않은 것:** B2 Final Test(별도 승인 필요), 학습.

**바뀌지 않은 것:** 기존 C1 원본·결과·사후 진단(plan `6b3f04fe…`, archive `c3960c8d…`).

호스트: Vast 62040825, RTX 4090, 드라이버 595.99.02, Intel i7-10700.

## 1. 작업별 결과 요약

| 작업 | 결과 | 비용(크레딧 기준 배분) | 상한 |
|---|---|---|---|
| B1 Calibration | **두 정책 모두 해악 인증 통과, 공동 주장 성립** | $1.75 | $2.50 |
| B3 일반화(PlainConv 정책, V2 무변경) | **H_arch 귀무가설 기각, p = 1.4×10⁻⁹** | $0.67 | $1.50 |
| B4 S6 substep 계측 | 시간 정렬로 1 mm 초과의 64% 해소, 잔차는 손가락 상자 근접 pair로 한정 | $0.32 | $1.00 |
| 합계 | 6.29 h, 인스턴스 2개(1개는 시작되지 않은 계약) | **$2.74** | $4.50 |

비용 배분 방법(`cost_split.json`):
- 인스턴스 비용은 크레딧 차이($7.2567 → $4.5176)입니다.
- 설치 시간은 세 작업에 똑같이 나눴습니다.
- 그 뒤의 시간은 구간마다 돌고 있던 프로세스 수 비율로 나눴습니다.

## 2. B1 Calibration (ras_v2_confirm_001)

### 실행 전 검증 (`b1_preflight/`)

- 동결 manifest(`01b70a06…`, 28 파일)를 commit·push·Drive에 올린 뒤에 시작했습니다.
- **runner 실환경 smoke**: 사용된 C1 장면 3개 × 정책 2개로 돌렸습니다(Calibration·Test seed는 쓰지 않음).
  - C1 원본 기록과는 다릅니다. 같은 호스트의 C1 수집기와는 정확히 같습니다. 즉 runner는 충실하고, 차이는 호스트
    차이(C1은 드라이버 550.120)입니다.
  - 사전 기준("C1 기록과 일치")이 이 이유로 충족되지 않았다는 사실과 호스트 유형 고정 결정은 Calibration 시작 전에
    commit했습니다.
- placement 검사: 400 seed, 참조 배치 2,347개, 동일 배치 0개.

### 실행

- 정책별 1회 실행, 각 400 장면(30000–30399), 오류 기록 0건입니다.
- 정책 1은 약 5.1 h, 정책 2는 약 5.1 h 걸렸습니다.
- 속도 때문에 정책 2 pass를 병렬 프로세스로 실행했습니다. 그래서 원래 체인의 정책 2 단계는 "이미 존재함"으로
  거부되었습니다(exit 1). 이는 의도된 동작이고, 분석은 별도 단계(`cal_analyze_manual`, exit 0)로 실행했습니다.

### 해악 인증 (ε = δ = 0.05, exact binomial, Holm family 2)

| 정책 | 장면 | 트리거 | V2 이동 | C0 성공 n₊ | 해악 k | 해악률(Wilson 95%) | p | Holm 기준 | 인증 |
|---|---|---|---|---|---|---|---|---|---|
| train_seed_1 | 400 | 386 | 73 | 348 | 1 | 0.29% (0.05–1.6%) | 3.4×10⁻⁷ | 0.05 | **통과** |
| train_seed_2 | 400 | 387 | 66 | 337 | 0 | 0% (0–1.1%) | 3.1×10⁻⁸ | 0.025 | **통과** |

- **공동 주장:** "두 정책 모두에서 V2의 해악률 ≤ 0.05"가 Calibration 분포에서 δ = 0.05로 보장됩니다.
- **로컬 재계산:** 원시 장면 기록에서 hci를 쓰지 않고 다시 계산했고, 서버 결과와 모두 일치합니다
  (`b1_calibration_recount.json`).
- **함께 보고(보증 없음):**
  - V2가 실제로 움직인 C0 성공 장면 48개 / 36개 중 해악 1 / 0입니다.
  - 기술 통계: C0 → V2 성공은 348 → 371(rescue 24), 337 → 366(rescue 29)입니다.
  - **효용 검정은 Final Test에서만 합니다.** 이 성공률 차이는 효용의 증거로 쓰지 않습니다.

## 3. B3 일반화 (gen_arch_plainconv_001)

다른 아키텍처의 동결 정책(Stage 1D 공식 PlainConv DP)에, V2를 한 줄도 바꾸지 않고 적용했습니다. 새 Development
seed 51300–51499를 썼고, placement 검사에서 동일 배치는 0개였습니다.

| Arm | 성공 / 200 (Wilson 95%) | 개입 | rescue | 해악 | 순이득 |
|---|---|---|---|---|---|
| C0 | 52 (20–32%) | – | – | – | – |
| V2_SINGLE_FRAME | 79 (33–46%) | 45 | 29 | 2 | +27 |
| **VISION_CENTER_V2** | **87 (37–50%)** | 61 | **37** | **2** | **+35** |
| CENTER_ORACLE (특권 참조) | 90 (38–52%) | 135 | 45 | 7 | +38 |

- **H_arch(사전 고정, 단일 검정):** V2만 성공 37 vs C0만 성공 2, p = 1.4×10⁻⁹ → 귀무가설 기각.
- **oracle 대비:** oracle이 살린 45개 중 V2가 37개를 살렸습니다.
- **메모리:** V2 대 single-frame은 8 대 0입니다(기술 통계).
- **해악:** 2/52 = 3.8%입니다. 개발 안전 기준(≤ 2)의 경계값이며, 분모가 작아 인증이 아닙니다.
- **해석 범위:** 같은 태스크·같은 시연 분포 안에서 V2의 이득이 ResNet-18 정책에 국한되지 않는다는 근거입니다.
  다른 태스크로의 일반화는 아닙니다.

## 4. B4 S6 substep 계측 (s6_substep_001)

- **try 1은 실패:** ManiSkill이 reset 때 scene 객체를 새로 만들어 로거가 한 번도 기록하지 못했습니다. 그래서 ON = OFF
  비교가 의미 없었습니다. 실패 기록으로 보존했습니다.
- **try 2:** reset·restore마다 다시 감싸도록 고쳤고, 기록이 0이면 오류가 나게 했습니다.
  - 사용된 C1 scene-run 11개(규칙으로 고정)를 돌렸고, 계측 ON = OFF가 11/11입니다.
  - 원본 C1과의 trace 일치는 0/11입니다(호스트 차이, §2). 따라서 아래 분석은 이 호스트에서의 실행에 대한
    것입니다.
- **마지막 substep의 분리된 목록 pair 56,558행:**

| 비교 | 1 mm 초과 | 차이 중앙값 |
|---|---|---|
| substep **이후** 자세(C1 방식) | 5,181 (9.2%) | 0.044 mm |
| substep **이전** 자세 | 1,874 (3.3%) | 0.00002 mm |

- **시간 정렬로 해소된 부분:**
  - 빠른 pair(>100 mm/s)의 1 mm 초과가 3,436행에서 146행으로 줄었습니다.
  - 큐브–큐브, 큐브–테이블, 손가락–테이블 pair는 사실상 0이 됐습니다.
- **남은 잔차 1,874행은 모두 손가락 상자–큐브 근접 pair입니다.**
  - impulse는 0이고, PhysX 값이 항상 큽니다(거리 중앙값 24 mm vs 27.6 mm).
  - 다음 원인은 배제했습니다:
    - 시간 정렬(이미 이전 자세로 비교함)
    - shape 번호 불일치(45,645개 중 17개)
    - 상자끼리 정지 상태의 모서리·꼭짓점 기하 정의(새 정적 실험 20 cases, GJK와 차이 0)
  - **원인은 미해결**입니다.
- **손–큐브 pair의 차이 중앙값 0.04 mm:** 메시 삼각분할 오차(hull 민감도 001)와 일치합니다.
- **공식 S6는 FAIL을 유지합니다.**

## 5. 논문에 쓸 수 있는 영문 주장

- "On 400 held-out calibration scenes per policy, the unchanged V2 turned 1 of 348 and 0 of 337 baseline successes into
  failures; the exact binomial test with Holm over the two policies certified a population harm rate of at most 5% for
  both policies at δ = 0.05 (p = 3.4×10⁻⁷ and 3.1×10⁻⁸)."
- "Applied without modification to a frozen policy of a different architecture, V2 raised success from 52 to 87 of 200
  development scenes (37 rescues, 2 harms; exact paired p = 1.4×10⁻⁹)."
- "Logging every physics sub-step showed that the contact list is computed at the pose before the last sub-step;
  aligning to that pose reduced disagreements above 1 mm from 9.2% to 3.3% of pairs. The remainder is confined to
  zero-impulse finger–cube proximity pairs and is unexplained."
- 필수 한계 문장: "Calibration and Test use a different simulator host configuration (driver 595.99.02) from the
  development run (550.120); development outcomes differed across hosts in some scenes."

## 6. 다음 단계 (승인 필요)

- **B2 Final Test(90000–90399):**
  - Test 동결(Calibration 분석 `calibration_analysis.json`과 동결 manifest 포함)을 먼저 합니다.
  - 같은 호스트 유형(RTX 4090 / 595.99.02)에서 실행합니다.
  - 사용자가 `RUN_FROZEN_TEST`를 직접 제공해야 합니다.
  - 예상 비용: 이번 속도 기준 정책 2개 병렬로 약 5.5 h, 약 $2.4. 상한 $3.00을 권고합니다.
  - CPU가 빠른 호스트를 고르면 줄어듭니다. 다만 드라이버는 반드시 595.99.02여야 합니다.

## 7. 파일

- `b1_preflight/`, `b1_calibration/`(분석, placement, PASS_DONE, ledger), `b1_calibration_recount.json`
- `b3_generalization/`(plan, authorization, placement, analysis, ledger)
- `b4_s6_substep/`, `s6_static_edge_cases/`
- `server_logs/`, `cost_intervals.json`, `cost_split.json`

원시 기록(약 14 GB)은 Git 밖에 있습니다: `runs/ras_exec_001/download_*`. 묶음 sha256은 다음과 같습니다.
- phase1 `0ebd7a04…`
- final `50efd63f…`
- smoke `20dc94e8…`
- b4 try1 `c367bc2f…`
- b4 try2 `be2e5b08…`

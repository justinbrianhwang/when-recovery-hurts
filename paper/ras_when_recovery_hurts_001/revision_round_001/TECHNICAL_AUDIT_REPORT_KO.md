# 기술 감사 보고서 — revision round 001 (2026-11-25)

정본: `Paper/Latex`. 개정 전 백업: `Paper/manuscript_package/revision_round_001/backup_Latex_before_revision/`.
새 실험·재학습·재실행·유료 GPU 사용은 없었습니다. 동결 파일과 Calibration·Final Test 결과도 바꾸지 않았습니다.
이슈 번호(R01–R28)는 `REVISION_ISSUE_REGISTER.md`의 번호이고, 코드 근거는 `CODE_TO_TEXT_AUDIT.csv`에 있습니다.

## 1. 코드와 원고가 어긋났던 부분 (BLOCKER 1, MAJOR 12)

### R01 Hand-back (BLOCKER)
- **개정 전:** "The held release command is then passed to the unchanged policy". 그림 2는 "release, then the task continues"였습니다.
- **코드:** `vision_center_v2.py` 224행(ACT 시작)과 253행(이동 종료)에서 `r.policy.clear()`를 호출합니다. 따라서 대기 중이던 열기 명령은 버려집니다.
  - 러너는 이동 중 추론이 0회이고 hand-back 시점의 queue가 비어 있는지를 assert합니다(313행).
  - 관측 이력은 `clear()`의 대상이 아니라 그대로 유지됩니다.
- **개정 후:** "queue cleared again; the frozen policy infers a new action chunk … decides when to open".
- **실측(원자료, 신규):** 모든 단계에서 V2가 이동한 branch는 이동 직후 첫 스텝에 재추론했습니다.
  - 첫 열기 명령은 1–24스텝 뒤에 나왔고, 중앙값은 2–3스텝입니다.
  - calibration branch 1개는 이후 그리퍼를 열지 않았습니다(Table S4).

### R02 Trigger
- **개정 전:** "while a cube is held", "fingers are closed on an object". 정답 파지 정보를 읽는 것처럼 읽혔습니다.
- **코드(`trigger_rule`):** 세 가지 관측 조건만 씁니다.
  - 대기 행동의 그리퍼 명령이 열기일 것
  - 직전에 실행된 닫기 명령이 3회 이상일 것
  - 손가락 폭이 0.02–0.06 m일 것
- **반례 51152(seed 2):** 실제로는 목표 큐브 B를 쥔 상태에서 트리거가 발동했습니다(손가락–B 거리 −2.16 mm, 손가락당 약 27.6 N).
  - V2를 포함한 모든 시각 arm은 NO_HOLD_ESTIMATE로 기권했습니다.
  - Oracle만 이동했고, B를 23.6 mm 밀었습니다.
  - 이 사례를 §4, §9, §11, Supp S6에 기술했습니다.

### R03 그리퍼 명령
- **개정 전:** "no gripper command".
- **코드:** `_xy_action`에서 `a[-1]=grip`으로, 직전에 실행한 닫기 명령을 매 스텝 그대로 보냅니다.
- **개정 후:** "zero vertical command and the held close command".

### R04 메모리 구조
- **개정 전:** hand/base/hold 세 개의 메모리.
- **코드(`Memory`):** target 메모리 1개와 hold 메모리 1개입니다.
  - target 메모리는 손목 카메라 적합(u=2 mm)을 우선 쓰고, 없으면 base 카메라 추정(u=5 mm)을 씁니다.
  - 두 관측이 일관되면 확인 횟수를 늘리고, 더 정밀한 소스가 위치를 소유합니다.
  - 불일치하면 메모리를 교체합니다.
  - 그림 3의 "memory hand/base"는 저장된 위치를 정한 소스를 뜻한다고 캡션에 명시했습니다.

### R05–R06 결정 규칙 순서와 실행 운동
- **R05:** `decide`의 순서를 그대로 적었습니다. 원격 목표(remembered target)에 대한 LOW_CARRY 조건도 포함했습니다.
- **R06:** 이전의 "40%" 표현은 비율의 분모가 불분명했습니다.
  - 기준을 "순 수평 TCP 변위 / 명령 경로 길이"로 고정했습니다.
  - development 범위 0.26–0.40, 중앙값 0.36·0.37입니다.
  - cap에서 끝난 이동은 0건이고, 모두 추정 허용오차 또는 스텝 예산에서 종료했습니다.

### R07 V2_RULES_OFF
- **개정 전:** "disables the abstention rules".
- **코드(`RULES_OFF_REMOVED`):** 다음 세 규칙만 하나의 묶음(bundle)으로 제거합니다.
  - ALIGNED_WITHIN_UNCERTAINTY
  - SIDE_CONTACT_RISK
  - 원격 목표에 대한 LOW_CARRY
- V1 규칙(추정 유효성, 높이, 남은 스텝, 15 mm)은 그대로 남습니다.

### R08–R09 Check 4
- **정의 오류(R08):** 진행 조건은 실제 스텝 크기 ≥0.2 mm입니다. 명령 방향 성분이 아닙니다(`trajectory_audit.py` 134–135행). 원고를 이에 맞게 고쳤습니다.
- **인과 표현(R09):** "because the arm is still moving"을 "associated … cause not established"로 바꿨습니다. 51157은 Check 4를 PASS했다는 점(cosine >0.999)도 명시했습니다.

### R10–R13 S4·S5·S6
- **S5(R10):** "penetration ≤1 mm"는 틀린 기술이었습니다.
  - 실제 정의는 impulse가 0이 아닌 형상 쌍의 재구성 거리가 1 mm 이하로 떨어져 있는지 보는 단측 검사입니다.
  - 관입량은 임계값 없이 보고만 합니다.
- **S6(R11):** PhysX separation이 양수인 목록 pair만, 모든 phase에서 비교합니다.
- **독립 재구성(R12):** "1e-12 mm 일치(cubeB–table 제외)"는 틀렸습니다.
  - 상자형 쌍은 1e-12 mm, cubeB–table은 4e-6 mm, hand–cube는 0.22–0.24 mm입니다.
  - hull 교체 시 hand 거리 변화는 0.24 mm 이하, 팔 링크는 0.03 mm 이하, 손가락은 0입니다.
- **S4 FAIL(R13):** 이동 중 충돌이 아니었습니다. 트리거 시점에 이미 목표 큐브를 쥐고 있던 상황의 기록입니다.
- **데이터 구분:** 원본 C1 기록(A), 정적 상자 실험(B), calibration host sub-step 재실행(C)을 본문·그림 7·Supp S6에서 분리해 서술했습니다.

## 2. 통계적 의미 (MAJOR 3)
- **R14 귀무가설:** `p=F_Bin(n+,ε)(k)`는 H0: h≥ε에 대한 검정이므로 모든 곳에서 `h>ε`를 `h≥ε`로 바꿨습니다. 수치는 바뀌지 않았습니다.
- **R15 인증의 의미:** "h≥ε인 정책을 인증할 확률 ≤δ"라는 빈도주의 보증으로 명시했습니다. 진짜 위험률에 대한 확률 진술이 아니며, calibration host 구성에 한정된다는 점도 적었습니다.
- **R16 i.i.d.:** 서로 다른 seed를 독립 표본으로 본 것은 가정이며 증명이 아니라고 명시했습니다.
- **기권과 인증 여유:** V2가 실제로 움직인 baseline 성공 장면만 보면 단측 상한이 약 10%입니다(1/48, 0/36, 0/38, 1/46). 인증 여유의 대부분이 기권에서 온다는 점을 서론 "strongest objection" 단락과 §7에서 먼저 다뤘습니다.

## 3. 서술 개정
- **서론:**
  - Δsucc=(R−H)/n 정식화와 예시 (30,0) 대 (60,30)을 넣었습니다.
  - 실패 탐지와 개입 평가의 차이를 설명했습니다.
  - 이 case study를 선택한 이유를 적었습니다.
  - "최초"라는 주장을 없앴습니다.
- **관련 연구:**
  - Sentinel·FIPER의 calibration은 탐지기의 오경보를 통제하는 것이라고 정확히 기술했습니다.
  - RESCUE와는 same-state counterfactual 설계가 같다는 점을 인정했습니다. 차이점은 harm 인증, held-out 확증 검정, 운동·물리 감사로 한정했습니다.
- **§11 미래 연구(R17):** "실제 변위를 읽는 복구"는 자기모순이었습니다(V2는 이미 매 스텝 재추정합니다). 실행 속도와 남은 horizon을 반영하는 feasibility/value gate로 바꾸고, 미구현·미시험임을 명시했습니다.
- **결과 단계 구분:** 확증 결과(인증과 efficacy 두 가지)와 사후·기술 분석을 §7과 §11에서 명시적으로 구분했습니다.

## 4. 그림·표·참고문헌
- **그림:**
  - 그림 2를 코드 흐름(관측 가능한 트리거, 순서 있는 규칙, queue clear, 닫기 명령 유지, 재계획)에 맞게 다시 그렸습니다.
  - 그림 4의 주석을 ε로 바꿨고, 그림 5 배치를 개선했습니다.
  - 파일명을 인쇄 번호와 일치시켰습니다(Fig01…Fig09, FigS1…S3).
- **표:** Table 5의 S4/S5/S6 정의를 정정했습니다. 신규 Table S4(hand-back)를 추가했습니다.
- **참고문헌:**
  - Holm: DOI가 404여서 제거하고 JSTOR 링크를 넣었습니다.
  - SAPIEN: 쪽수를 정정했습니다.
  - ManiSkill3: RSS 2025 게재본으로 바꿨습니다.
  - ResNet, HG-DAgger, Recovery RL: 검증했습니다.

## 5. 검증
- 원고 수치 36개를 recount와 hand-back 통계에 자동 대조했고 36/36 일치했습니다.
- Main과 Supplementary 모두 컴파일했습니다(35쪽, 12쪽). Supplementary는 루트 빌드와 폴더 내부 빌드가 동일합니다.
- flat zip 빌드는 정본 빌드와 텍스트가 동일합니다.
- 전 페이지를 시각 검수했습니다.

## 6. 남은 위험
- RAS Guide for Authors 미확인(403).
- 저자 항목(이름, 소속, 이메일, CRediT, 연구비, 데이터 링크, AI 고지와 이탈 문구).
- 새 실험이 필요한 한계: 외부 baseline, matched-delay arm, 규칙 단위 ablation, 단일 태스크.

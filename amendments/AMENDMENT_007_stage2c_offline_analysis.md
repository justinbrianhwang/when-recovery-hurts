# 설정/프로토콜 변경 기록 007 — Stage 2C 저장 기록의 사후 분석 (실행 없음)

기존 버전: v0.3 + AMENDMENT_001~006 (저장소 HEAD `8ef0cf7`)
새 버전: v0.3 + AMENDMENT_001~007
변경 시점: 2026-11-14 (UTC). Stage 2C의 어떤 새 집계도 계산하기 전.
변경 사유: Stage 2B에서 Refresh의 평균은 Continue와 0.03 이내였고, 고정 시점에 강제한 Retreat–Retry는 0.16~0.25 낮았다.
사용자가 `prompts/STAGE2C_OFFLINE_SELECTION_AND_RETREAT_REVIEW_AGENT_PROMPT_v0_1.md`로, 새 실행 없이 저장된 기록만으로 두 가지를 확인하기로 했다.

1. 평균적으로 나쁜 개입이라도, 같은 상태의 다른 난수 반복에서 유리한 선택이 유지되는가?
2. Retreat가 실패로 바꾼 실행과 성공으로 바꾼 실행에서 관찰되는 사건 순서가 어떻게 다른가?

이것은 **Stage 2B의 집계와 일부 사례를 이미 본 뒤에** 계획한 개발용 사후 분석이다. 확증 실험도 독립 검증도 아니다.

## 1. 하지 않는 것

GPU 대여, simulator rollout, 정책 추론, 학습, 렌더링, qualification, calibration, 최종 Test. 유료 자원 0, 새 장면 0.
정책, horizon, 성공 판정, queue, 난수 규칙, Retreat 설정, 기존 결과와 판정(37/100 미달)은 그대로다.
Q/Rescue/Harm 예측기나 온라인 선택기를 학습하지 않는다. 저장된 온라인 입력(`inputs/*.npz`)은 존재와 구조만 확인한다.
runtime 코드(`src/hci`)는 수정하지 않는다. 새 파일만 추가한다.

```text
base_policy_qualification: BELOW_ORIGINAL_GATE
stage2a: COMPLETED_EXPLORATORY_ONLY
stage2b: COMPLETED_EXPLORATORY_ONLY
stage2c: (실행 후 기록) COMPLETED_POSTHOC_OFFLINE_DIAGNOSTIC 또는 PARTIAL_INPUT_UNAVAILABLE
observation_based_selector: NOT_TRAINED
calibration: NOT_RUN
final_test: NOT_RUN
```

## 2. 입력과 보호

- 주 자료는 Stage 2B뿐이다(`results/stage2b/`, `runs/stage2b_001/branches/{traces,inputs,scenes}`). Stage 2A는 문맥으로만 언급하고 합치지 않는다.
- 분석 전에 모든 입력 파일과 분석 코드의 SHA256을 `runs/stage2c_001/analysis_manifest.json`에 적는다. 분석 명령은 hash가 하나라도 다르면 거부하고, 끝난 뒤 입력이 그대로인지 다시 확인한다.
- 원본 파일(장면 기록, trace, 입력)은 서버에서 만든 파일별 hash 목록과 대조한다.
- 기록은 Stage 2B 실행 당시 코드(plan에 고정된 hash)의 구조로 읽는다. 현재 HEAD와 혼동하지 않는다.
- 자료가 없으면 만들거나 다시 실행하지 않는다. trace가 없으면 사건 분석은 `NOT RUN`으로 남긴다.

## 3. 고정한 규칙

**A. 재검산.** CSV와 장면 JSON을 `(scene, checkpoint, block_repeat_id)`로 잇는다. 중복, 누락, 값 충돌, 다른 snapshot에 묶인 block은 오류다.
결과 값은 0과 1만, boolean은 `True`/`False`만 받는다. 보고서 숫자와 다르면 원인을 적고, 맞추려고 행을 지우지 않는다. 풀리지 않는 불일치가 있으면 그 뒤 분석을 하지 않는다.

**B. snapshot 단위 표.** 84개 snapshot마다 네 arm의 성공 수(0~4)와 각 패턴이 나타난 repeat 번호를 적는다. 0/4나 4/4를 성공 확률 0이나 1이라고 쓰지 않는다.

**C. 반복을 하나 남기는 선택 일관성 진단.**

- snapshot마다 repeat 하나를 남기고, 나머지 세 repeat의 성공 수가 가장 큰 arm을 고른 뒤, 그 arm의 결과를 남겨둔 repeat에서 읽는다. repeat 0~3을 모두 남겨 본다.
- 후보 집합 세 가지를 그대로 비교한다: `G_CF = {C0, F}`, `G_CR = {C0, R}`, `G_CFR = {C0, F, R}`.
- 동률이면 C0, 다음 F, 다음 R이다.
- C1은 후보도 동률 처리도 아니다. 같은 남겨둔 repeat의 추가 기준 결과로만 적는다.
- 이것은 **같은 상태에서 arm의 순위가 난수 반복 사이에 유지되는지** 보는 진단이다. 과거 분기 결과를 입력으로 쓰므로 실제로 배포할 수 없다.
  새 장면 일반화, 관측 기반 학습, 온라인 정책 평가, 교차검증, 인과 검정이 아니다.
- 한 장면의 행은 서로 관련돼 있고, repeat는 고르는 데 세 번, 읽는 데 한 번 다시 쓰인다. 행 수(규칙당 336)는 독립 표본 수가 아니다.
- 결과를 보고 동률 처리, 차이 임계값, 후보 집합을 바꾸지 않는다.

**D. Retreat 전후의 사건.**

- 336개 block 전부에 같은 추출기를 쓰고, 집합(R 악화 69, R 복구 15, 나머지)은 결과로 나눈다.
- 시간축: trunk의 i번째 행은 step i+1 직후의 상태, 분기의 i번째 행은 step checkpoint+i+1 직후의 상태다. checkpoint 시점의 상태는 trunk의 checkpoint−1번째 행이다(수집 코드에서 확인).
- Retreat 종료 시각은 block마다 기록된 `steps_used`이고, R arm의 첫 추론 step과 대조한다. 25 step으로 고정하지 않는다.
- 기준값은 기존 진단 라벨의 것을 그대로 쓴다: 접근 3 cm, 정렬 2 cm, 들어올림 1 cm. 큐브 B가 "움직였다"는 2 mm 이상 위치 변화로 이번에 정했다(결과를 보기 전).
- 라벨은 관찰이다. 다음은 붙이지 않는다: "파지 flag가 꺼졌다 = 떨어뜨렸다", "step 200에서 실패 = 시간 소모가 원인", "쓴 step만큼 더 주면 성공", "분포 밖 상태가 원인", "큐브 B가 움직였다 = 충돌".
- 사례는 표를 만든 뒤 고정 규칙으로 고른다: 집합과 관찰 flag마다 scene/checkpoint/repeat 오름차순 첫 block, 그리고 패턴이 repeat 2개 이상에서 나타난 첫 snapshot. 사후 선정임을 표시한다.
- 영상은 기존 8개뿐이다. 새로 렌더링하지 않으며, 영상이 없는 사례를 "영상 확인"이라고 쓰지 않는다.

**하지 않는 계산.** p-value, 신뢰구간, LTT, 위험 보장, 자료에 맞춘 임계값 탐색. block별 최댓값을 실행 가능한 성능으로 제시하지 않는다.

## 4. 이 단계로 할 수 없는 결론

- 관측 기반 선택기를 학습할 수 있다, 또는 없다.
- Retreat가 실패를 만든 원인(시간, 파지 상실, 분포 밖 상태 등).
- 선택적 개입이 불가능하다(강제 개입의 평균이 낮다는 것만으로는 말할 수 없다).
- 새 학술적 발견이라는 선언.

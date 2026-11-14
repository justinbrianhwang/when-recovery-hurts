# 설정/프로토콜 변경 기록 005 — Stage 2A 탐색적 복구 파일럿 (범위 신설)

기존 버전: v0.3 + AMENDMENT_001~004 (저장소 HEAD `a5c0b1c`)
새 버전: v0.3 + AMENDMENT_001~005
변경 시점: 2026-11-14 (UTC). 이 파일럿의 어떤 rollout도 실행하기 전. GPU 대여 전.
변경 사유: D500-clean 80k 정책의 qualification은 37/100으로 운영 기준 50/100에 미달했다.
사용자가 `prompts/STAGE2A_EXPLORATORY_RECOVERY_PILOT_AGENT_PROMPT_v0_1.md`로 **추가 학습 대신, 이 정책을 그대로 두고 작은 복구 파일럿을 실행**하기로 결정했다.
이것은 qualification 결과를 본 뒤에 정한 후속 탐색이다. 사전 등록된 확증 실험이 아니다.
모델, 추론 설정, horizon, 성공 판정, 개입 정의, Retreat 설정, 복원 방식, 허용 오차는 바꾸지 않는다.

## 1. 바뀌는 것과 바뀌지 않는 것

| 항목 | 상태 |
|---|---|
| qualification B 37/100, 기준 50/100 미달이라는 판정 | **그대로.** 기준을 낮추지 않고, 과거 결과를 PASS로 고치지 않는다 |
| 기준 미달이면 복원 audit와 분기 수집을 하지 않는다는 Stage 1D의 종료 규칙 | Stage 1D 기록으로 그대로 둔다. Stage 2A는 **별도 범위**로 아래 실행량만 허용한다 |
| 새 학습, 데이터 확대, 체크포인트 재선택, qualification 재실행 | 금지 |
| 5,760회 파일럿, 예측기(Q·도움·악화) 학습, 후보 21개 평가, calibration, 최종 Test | 금지 |

상태는 두 축으로 적는다.

```text
base_policy_qualification: BELOW_ORIGINAL_GATE (37/100 < 50/100)
exploratory_recovery_pilot: AUTHORIZED_BY_STAGE2A_SCOPE
formal_study_ready: NOT_ESTABLISHED
calibration: NOT_RUN
final_test: NOT_RUN
```

## 2. 범위를 지키는 방법 (실행기 추가)

기준을 건너뛰는 전역 옵션은 만들지 않았다. 새 실행기 `scripts/18_stage2a_pilot.py`와 `scripts/stage2a/run.sh`는 `runs/stage2a_001/plan.json`에 적힌 것만 실행한다.

- 서버의 모든 명령은 정책 가중치·정책 metadata·설정·split 파일·실행 코드의 SHA256을 plan과 다시 대조하고, 하나라도 다르면 거부한다.
- 장면은 plan에서 확정한 목록의 앞부분만 받는다. 3번째 장면부터는 기술 확인 통과와 규모 결정 파일이 있어야 한다.
- 정책 hash는 보고서의 짧은 표기가 아니라 보존된 백업 목록(`results/stage1d_clean/backup_sha256_stage1d_clean_002.txt`)의 전체 SHA256과 대조한다.
- 기존 `hci collect`, `hci audit`, Stage 1D 실행기는 수정하지 않았다.

`src/hci/simulation.py`에는 한 가지만 더했다. prefix replay로 복원할 때 **환경이 마지막 step에서 실제로 돌려준 관측**을 `replay_raw`로 남긴다. 실행에 쓰는 값과 복원 절차는 그대로다.

## 3. 장면과 역할 (데이터 역할 추가)

- 파일럿 장면: `configs/splits.json`의 train 집합에서 수치 오름차순으로 앞 24개. 역할 이름은 `stage2a_exploratory_dev`.
- 다른 용도로 선언된 seed(시연, 보호된 시연, audit, dev, policy_monitor/select/qualify A·B, calibration, Test)와 겹치면 train의 다음 seed로 넘어간다. train이 모자라면 다른 split에서 빌리지 않고 중단한다.
- 서버에서 초기 큐브 배치도 대조한다(시연 원본 1000개, audit, dev, policy 평가 집합). 같은 배치(1e-5 m 이내)가 있으면 수집을 시작하지 않고, 같은 규칙(train의 다음 seed)으로 목록을 고친 plan을 새로 쓴다. calibration과 Test seed는 reset하지 않고 seed 번호로만 대조한다.
- 이 정책의 성공 여부를 보고 장면을 바꾸지 않는다. 목록은 정책을 한 번도 실행하기 전에 확정된다.
- seed 10000, 10001은 Stage 0~2에서 legacy 정책(시연 99개)의 분기 smoke에 쓰인 적이 있다. 사용 이력으로 기록하고 교체하지 않는다.
- 이 표본은 개발용이다. 나중에 dev, calibration, Test로 옮기지 않는다. 한 장면의 모든 checkpoint와 반복은 같은 역할에 속한다.
- audit 장면은 기존 audit 집합(seed 100~104) 그대로다. 다섯 개 모두 시연 장면과 겹친다. 복원 검사에만 쓰고 성능이나 분모로 쓰지 않는다.

## 4. checkpoint, 반복, 실행량 상한

- checkpoint step `[12, 52, 92, 132]`, repeat `[0, 1, 2, 3]`. 현재 설정 값과 같다. 결과를 보고 바꾸거나 더하지 않는다.
- 세 선택(Continue, Refresh, Retreat–Retry)이 모두 가능한 상태에서만 snapshot을 만든다. 그렇지 않은 checkpoint는 사유와 함께 기록한다.
- 상한: 장면 24, snapshot 96, block 384, continuation 1,152. 이와 별도로 trunk 24회, 복원 audit, 아래 재현 검사가 있다.
- 처음 2개 장면(최대 96 continuation)으로 기술 조건만 확인한다. 통과하면 결과의 유불리와 무관하게 나머지를 같은 규칙으로 실행한다.
- 축소 순서는 24 → 12 → 2이고, 확정한 목록의 앞부분만 쓴다. 규모는 처음 2개 장면의 실행 시간과 비용으로만 정한다.

## 5. 반사실 비교의 정의

block은 `(scene, checkpoint, repeat)`이다.

- **과거**: 세 선택 모두 같은 장면 seed, 같은 실행 행동 기록(prefix replay로 복원), 같은 정책 history와 남은 queue에서 출발한다.
- **미래 난수**: 추론 호출의 확산 noise는 `(scene, 호출 시점의 physical step, repeat)`로 정해진다. 선택 종류나 실행 순서에는 의존하지 않는다. 기존 구현 그대로다.
- **Continue**: 남은 queue를 그대로 쓴다. 다음 추론은 원래의 chunk 경계에서 일어난다.
- **Refresh**: 남은 queue를 버리고 checkpoint step에서 추론한다. 따라서 추론 시점이 Continue와 달라지고, 같은 noise를 쓰지 않는다. 이것은 Refresh의 정의에 포함된 차이다. 같은 행동열이라고 주장하지 않는다.
- **Retreat–Retry**: 실제 제어 step으로 실행하고 같은 200-step 예산에서 차감한다. 추가 시간, 상태 이동, 물체 reset은 없다.
- **기준**: 각 block은 **같은 block의 Continue 결과**와만 비교한다. 다른 repeat의 Continue나 원래 trunk의 결과와 짝짓지 않는다.
- trunk는 repeat 0의 noise로 한 번도 복원하지 않고 끝까지 실행한다.

## 6. 복원 audit (검사 강화, 허용 오차는 기존 값)

이전 audit는 다른 정책에 대한 결과다. 이 정책·설정·host에서 새로 통과해야 수집한다. 복원 방식은 prefix replay 그대로이고 `set_state_dict`로 되돌리지 않는다.

도달한 checkpoint마다: Continue 20 step → 복원 후 반복 → Retreat 분기를 끝까지 실행한 뒤 복원하고 반복 → Refresh 분기를 끝까지 실행한 뒤 복원하고 반복.

| 비교 대상 | 기준 |
|---|---|
| simulator 상태, TCP 기록, 정책 history의 관절·TCP 값 | 기존 `state_atol` 1e-6 |
| 정책 history의 RGB | 기존 `observation_atol` 1.0 (uint8 단위) |
| 남은 action queue, 마지막 행동, 실행 행동 기록 | 기존 `action_atol` 1e-6 |
| step 카운터, 종료·성공 flag, 종료 사유, 개입 기록, Retreat 기록, queue·history 길이, 추론 횟수, 난수 상태(Python, NumPy, Torch, CUDA, 환경) | 완전 일치 |
| 복원 직후 **환경이 돌려준 관측** 대 snapshot에 저장된 관측 | 관절·TCP는 `state_atol`, RGB는 `observation_atol` |

- 기존 audit는 상태, 마지막 history 관측, 행동, 결과만 비교했다. 항목을 늘렸고 허용 오차는 넓히지 않았다.
- 저장된 관측 사본만 비교하면 실제 환경의 차이가 가려질 수 있어, 환경에서 받은 관측을 따로 비교한다.
- 추론 소요 시간은 비교하지 않는다(실행 시간 값).
- audit가 실패하면 원인을 기록하고 수집하지 않는다. 새 학습으로 넘어가지 않는다.

## 7. 재현 검사

- 모든 snapshot: repeat 0의 Continue는 복원하지 않은 trunk와 행동열·결과·step·최종 상태가 같아야 한다.
- 처음 2개 장면: 각 snapshot에서 repeat 1의 Continue를 다른 분기를 모두 실행한 뒤 한 번 더 실행해 처음 것과 비교한다(최대 8회 추가).
- 불일치는 보정값으로 빼지 않는다. 재현성 문제로 표시하고 효과 해석을 보류한다. 측정한 행은 그대로 남기되 데이터 집합을 완료로 표시하지 않는다.
- 분기를 시작할 때마다 복원된 실행 상태 전체를 snapshot 시점 기록과 다시 대조한다. 다르면 그 장면을 오류로 남기고 중단한다.

## 8. 기록과 분석

- 정책 입력은 기존 whitelist 그대로다. 큐브 위치, 파지 flag 같은 정답 상태는 step마다 **기록만** 하고 정책에 주지 않는다.
- block마다 `y_continue`, `y_action`, 네 분류(Rescue, Harm, Both succeed, Both fail), 성공 step 또는 종료 사유, 개입 step, 추론 횟수, Retreat의 명령·실제 변위·step·도달 여부, snapshot 시점의 queue·history 길이·남은 예산을 남긴다.
- 실패 위치(접근·파지·들어올림·정렬)는 이 실행의 trunk와 분기에서 기존 진단 함수로 기록한다. qualification B의 실패 63개를 다시 실행하지 않는다. 단계는 관찰 라벨이고 원인이 아니다.
- 영상: 처음 2개 장면, 첫 유효 checkpoint, repeat 0의 세 선택(최대 6개).
- 조건부 비율은 고정된 checkpoint 표본의 기술통계다. 분모가 0이면 `NA`로 적는다.
- 한 장면에서 나온 block은 독립 episode가 아니다. 구간이나 p-value를 계산하지 않는다. 장면별 표와 서로 다른 장면 수를 함께 적는다.
- checkpoint에서 선택을 강제한 결과를 온라인 개입 정책의 성능이나 `H(g)`라고 부르지 않는다.
- Harm이 0건이면 "이 범위에서 0건"이다. 안전성이나 5% 이하 보장이 아니다.
- qualification B의 성공 37건은 이 파일럿의 분모에 넣지 않는다.

## 9. 예산

Stage 2A 상한은 US$1.50이다. Stage 1D 누적 약 $1.13과 따로 기록하고, Stage 1D 한도를 소급해서 바꾸지 않는다.
비용은 표시 단가가 아니라 credit 차이로 적는다. 업로드는 선택된 80k 정책 하나와 코드·설정·plan으로 한정한다. 학습 HDF5와 나머지 checkpoint는 올리지 않는다.

## 10. 이 단계로 할 수 없는 결론

- 37/100 정책이 qualification을 통과했다.
- 복구 개입이 유익하다, 또는 무해하다.
- 온라인 개입 정책의 성능이나 악화율.
- 통계적으로 유의한 효과.
- 정식 연구(calibration, Test)를 시작할 준비가 됐다.

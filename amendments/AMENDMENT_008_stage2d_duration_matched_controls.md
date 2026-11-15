# 설정/프로토콜 변경 기록 008 — Stage 2D 반환 시점을 맞춘 0 / 2 / 5 cm 대조 (arm 추가)

기존 버전: v0.3 + AMENDMENT_001~007 (저장소 HEAD `7bb6b4d`)
새 버전: v0.3 + AMENDMENT_001~008
변경 시점: 2026-11-14 (UTC). Stage 2D의 어떤 rollout도 실행하기 전. GPU 대여 전.
변경 사유: Stage 2B에서 Retreat–Retry(위로 5 cm)는 Continue보다 결과가 낮았다(336개 block 중 성공 39 대 93).
Stage 2C의 사후 분석에서는 Retreat 구간 동안 기록된 파지·들어올림·정렬 flag가 꺼지지 않았고, 실패는 모두 정책이 제어를 돌려받은 뒤에 기록됐다.
그 기록으로는 "5 cm 이동"과 "그동안 지나간 step" 중 무엇이 관련되는지 가를 수 없었다.
사용자가 `prompts/STAGE2D_DURATION_MATCHED_RETREAT_CONTROL_AGENT_PROMPT_v0_1.md`로 반환 시점을 맞춘 대조 실험을 결정했다.

이것은 **Stage 2B의 결과, Stage 2C의 사후 분석, 그 뒤에 추가한 보충 분석(파지 중이던 42개 block)의 사례까지 본 뒤에 정한** 설계다.
사전 등록된 확증 실험도, 독립 검증도, 최종 Test도 아니다.

모델, 추론 설정, horizon, 성공 판정, 기존 세 선택의 정의, Retreat 설정, 복원 방식, 허용 오차, checkpoint, repeat는 바꾸지 않는다.

## 1. 상태

```text
base_policy_qualification: BELOW_ORIGINAL_GATE (37/100 < 50/100)
stage2d: EXPLORATORY_DURATION_MATCHED_CONTROL
observation_based_selector: NOT_TRAINED
formal_study_ready: NOT_ESTABLISHED
calibration: NOT_RUN
final_test: NOT_RUN
```

하지 않는 것: 기본 정책 재학습, 체크포인트 재선택, qualification 재실행, 거리·시간 grid search, Refresh와 대체 난수 Continue의 재실행,
Q/Rescue/Harm 예측기 학습, 관측 기반 선택기, LTT/calibration, 최종 Test, 12개 장면을 넘는 수집.

## 2. 네 arm

| arm | 약칭 | 제어 구간 | 정책 재개 |
|---|---|---|---|
| `continue` | C0 | 없음. queue와 history 유지 | 원래 chunk 경계 |
| `retreat_5cm` | R5 | 기존 Retreat–Retry 그대로(`Runtime.force(2)`): snapshot TCP + 5 cm(z), 명령 증분 ≤ 5 mm/step, 도달 오차 5 mm, 최대 40 step | Retreat가 끝난 step |
| `retreat_2cm_matched` | R2M | 목표 snapshot TCP + 2 cm(z). 도달하면 그때의 **실제 TCP 위치**를 유지 목표로 삼아 T_ref step이 될 때까지 유지 | checkpoint + T_ref |
| `hold_matched` | H0M | snapshot의 TCP 위치를 유지 목표로 T_ref step 동안 추종 | checkpoint + T_ref |

공통:

- R2M과 H0M은 queue를 한 번 폐기하고, R5와 같은 delta controller·명령 증분 한계·도달 오차를 쓴다. 제어 구간의 모든 step에서 snapshot의 마지막 gripper 명령을 유지한다.
- 제어 구간의 모든 step은 `Runtime.advance`를 지난다. 실제 관측 history가 계속 갱신되고, 쓴 step은 같은 200-step 예산에서 빠진다.
  중단 전 history로 되돌리거나 같은 프레임을 복제해 맞추지 않는다.
- 제어 구간 동안 정책 추론은 0회다. 끝나면 queue를 비우고 실제 history로 정책을 재개한다.
- 물리는 정상 진행한다. `set_state`로 물체나 관절을 고정하지 않는다. 위치 유지가 안 되거나 물체가 움직여도 그대로 결과다. "zero action = 위치 유지"라고 가정하지 않고 고정 목표에 대한 feedback과 실제 drift를 기록한다.
- R2M이 2 cm에 먼저 도달해도 정책으로 바로 돌아가지 않는다. T_ref 안에 도달하지 못하면 그 시점에 끝내고 `reached=false`로 남긴다. 시간을 늘리거나 표본에서 빼지 않는다.
- 도달 오차 5 mm를 R2M에 맞춰 줄이지 않는다. 명령 거리(0.02)와 실제 이동량을 따로 기록한다.
- 새 arm 때문에 적격 조건을 느슨하게 하지 않는다. snapshot은 기존 세 선택이 모두 가능할 때만 만든다.

## 3. T_ref

- snapshot마다 한 번, 복원한 뒤 R5의 **제어 구간만** 실행한다(`Runtime.force(2)`, 정책 추론 없음). 끝난 시점에 정책을 부르기 전 `steps_used`를 읽는다. 그 값이 T_ref다.
- 그 뒤 snapshot으로 돌아가 전체 실행 상태와 환경이 돌려준 관측을 다시 대조한다. 다르면 그 장면을 오류로 남기고 멈춘다.
- T_ref를 Stage 2B의 평균(25.4), 고정값(25, 30), 다른 장면의 값으로 대신하지 않는다. 어떤 분기의 결과를 보고 고르지도 않는다.
- 참조 실행에서 정책이 한 번이라도 호출되면 오류다. 각 repeat의 실제 R5 제어 구간은 참조와 명령·상태·관측·step 수가 같아야 한다. 다르면 결과를 맞추지 않고 기술 오류로 멈춘다.
- 참조 실행은 준비 실행이다. snapshot마다 따로 기록하고 continuation이나 결과로 세지 않는다.
- 참조가 제어 도중 끝나면(성공 또는 종료) 그때의 `steps_used`가 T_ref이고 `reference_early_absorbing=true`로 표시한다.
  다른 arm은 같은 T_ref를 쓰되 자신이 먼저 끝나면 즉시 끝난다. 그런 block의 반환 시점 일치 검사는 N/A다. block을 지우거나 T_ref를 다시 고르지 않는다.
- T_ref는 simulator에서 R5를 먼저 실행해 얻은 일정이다. R2M과 H0M은 진단용 변형이고, 이대로 배포할 수 있는 복구 동작이 아니다.

## 4. 난수

- block 키는 `(scene, checkpoint, block_repeat_id)`, r = 0..3. 네 arm 모두 `seed_for(scene, 호출 시점의 physical step, r, 'dp')`를 쓴다(기존 코드 경로 그대로).
- R5, R2M, H0M은 제어 도중 끝나지 않는 한 checkpoint + T_ref에서 같은 seed로 첫 추론을 한다. 관측이 서로 다르므로 행동 청크가 같아야 한다는 검사는 없다.
- C0의 추론 시점과 seed는 대조 arm과 다르다. C0와의 차이를 이동이나 시간의 순수한 효과라고 부르지 않는다.
- 같은 seed는 같은 의사난수 조건을 지정할 뿐, 통계적 독립의 증명이 아니다.

## 5. 코드 변경

| 파일 | 내용 |
|---|---|
| `src/hci/pilot2d.py` (신규) | R2M·H0M 제어(`matched_control`), T_ref 참조, 네 arm 수집기, 반환 시점 검사, audit 확장, 표, 사건 읽기 |
| `scripts/23_stage2d_pilot.py`, `scripts/stage2d/run.sh` (신규) | plan에 적힌 정책·설정·코드·장면만 실행하는 Stage 2D 전용 실행기 |
| `tests/test_stage2d_pilot.py` (신규) | 7절의 로컬 검사 |

- `simulation.py`, `policies.py`, `pilot.py`, `pilot2b.py`, 기존 실행기는 수정하지 않았다. plan에 Stage 2B 실행 때의 파일 hash와 같은지 적는다. R5는 기존 `Runtime.force(2)`를 그대로 부른다.
- 정책 입력에 절대 episode 시간은 들어가지 않는다. 정책이 읽는 것은 RGB history와 qpos·qvel·TCP pose history다(`rgb_qpos_qvel_tcp_v1`).
  `t/horizon`은 `policy.features()`에만 있고 기본 정책의 행동 생성에는 쓰이지 않는다. 이것은 구현 확인이고, 정책이 시간에 의존하지 않는다는 주장이 아니다(관측 이력과 자세가 시간과 함께 변한다).
- R2M·H0M의 제어 입력은 측정된 TCP 위치, snapshot의 TCP 위치, 마지막 gripper 명령, T_ref뿐이다. 물체 pose, task flag, 결과는 쓰지 않는다.

## 6. 장면과 실행량

- 새 장면: train에서 Stage 2A(10000~10023), Stage 2B(10024~10047), 다른 용도로 선언된 seed를 빼고 수치 오름차순 앞 12개. 역할 이름 `stage2d_exploratory_dev`.
- 초기 큐브 배치 대조: 시연 원본·Stage 2A·Stage 2B 장면은 저장된 좌표로, audit·dev·policy 평가 집합은 서버에서 reset해서 비교한다. calibration과 Test는 seed 번호로만 대조한다.
- Stage 2B에서 Retreat가 유리했던 장면(10035, 10043 등)을 표본으로 다시 쓰지 않는다. 이 장면들은 앞으로 calibration/Test로 옮기지 않는다.
- checkpoint `[12, 52, 92, 132]`, repeat `[0, 1, 2, 3]`. 상한: snapshot 48, block 192, continuation 768.
  별도로 trunk 12회, T_ref 참조 최대 48회, 복원 audit(seed 100~104), 재현 검사 최대 16회.
- 처음 2개 장면(최대 128 continuation)으로 기술 조건과 비용만 확인한다. 축소 순서는 12 → 6 → 2이고 실행 시간과 비용으로만 정한다. 결과의 유불리는 입력이 아니다.
- 조기 성공이나 부적격으로 빠진 칸은 사유와 함께 남기고 새 장면으로 채우지 않는다.

## 7. 검사

로컬(mock): R5가 기존 runtime과 같은지 / R2M이 일찍 도달해도 정책을 먼저 부르지 않는지 / H0M이 고정 목표를 따르고 상태를 쓰지 않는지 /
gripper 명령 유지 / 세 arm의 첫 추론이 같은 step·같은 seed이고 제어 구간 추론이 0회인지 / 미도달과 예산 소진을 숨기지 않고 시간을 더하지 않는지 /
조기 종료 때 남은 구간을 실행하지 않는지 / 제어 step이 같은 horizon에서 빠지는지 / 다른 arm을 실행해도 오염이 없는지 /
결과와 물체 상태가 제어나 저장 입력에 들어가지 않는지 / 반환 시 queue·history가 일관되는지 / 거리·시간·표본이 실행 인자가 아닌지.

서버 audit(seed 100~104, 이 host에서 새로 실행): Stage 2A의 비교 항목과 허용 오차 그대로. 추가로
참조 실행 뒤, R5·R2M·H0M 각각을 끝까지 실행한 뒤에 복원해서 같은 Continue trace를 다시 비교하고, R5 제어 구간이 참조와 같은지,
세 arm의 반환 step·seed·queue·history가 선언과 같은지, 정책 입력 저장이 실행 상태를 바꾸지 않는지 확인한다.

수집 중:

| 검사 | 범위 | 실패 시 |
|---|---|---|
| 분기 시작 전(그리고 참조 실행 뒤) 복원 상태 전체 대조 | 모든 분기 | 장면 오류, 중단 |
| R5 제어 구간 = 참조(명령·상태·관측·step 수·기록) | 모든 block | 장면 오류(기술 결함) |
| 제어 구간 추론 0회, 반환 시 queue 비어 있음, 첫 추론이 checkpoint + T_ref에서 같은 seed | 모든 block | 장면 오류 |
| repeat 0의 C0가 복원하지 않은 trunk와 일치 | 모든 snapshot | 재현성 문제로 표시, 자료를 완료로 표시하지 않음 |
| r=0의 R2M·H0M 재실행(다른 분기를 모두 실행한 뒤) | 처음 2개 장면의 모든 snapshot | 위와 같음 |
| 추론 호출마다 seed가 선언한 키와 같은지 | 모든 호출 | 자료를 완료로 표시하지 않음 |

허용 오차를 넓히거나, 실패 표본을 지우거나, 보정값을 빼서 통과시키지 않는다.

## 8. 기록

- snapshot마다: T_ref와 참조의 종료 사유, snapshot의 TCP·qpos·qvel, 마지막 gripper 명령, queue 길이, 남은 예산, 참조 명령의 hash. 참조 구간의 step별 기록은 trace 파일의 `ref_<checkpoint>`.
- 분기마다: arm, block repeat, 명령 거리, 실제 이동량, 첫 도달 step, 유지 step 수, 유지 목표, drift 최댓값, 첫 정책 step, 추론별 step·seed, 반환 시 history hash와 queue 길이,
  결과, 종료 step·사유, 성공이 제어 구간에서였는지 반환 뒤였는지, 도달 여부, 조기 종료 여부, 제어 step 수, 반환 뒤 step 수, 추론 횟수.
- step별: 기존 항목(TCP, 큐브 A/B 위치, flag, 행동 명령)에 gripper 관절 위치와 `phase`(continue / intervention_move / intervention_hold / policy_after_return)를 더한다.
  gripper 관절 위치는 정책이 이미 받는 qpos의 마지막 두 값이다. 큐브 pose와 task flag는 진단 전용이고 정책 입력이나 제어 목표에 쓰지 않는다.
- snapshot의 온라인 입력은 결과가 나오기 전에 한 번 저장한다. 이 단계에서 모델을 학습하지 않는다.
- 영상: 처음 2개 장면, 첫 유효 checkpoint, r=0의 네 arm(최대 8개). 결과를 보고 추가하지 않는다.

## 9. 분석(실행 전 고정)

- 주 비교: R2M − R5, H0M − R5, R2M − H0M. 보조 비교: R5 − C0, R2M − C0, H0M − C0.
- 각 비교는 같은 block의 네 조합(0→1, 1→0, 1→1, 0→0)을 분자·분모와 함께 적는다.
- 주 요약은 장면 동일 가중이다(장면 안에서 도달한 checkpoint와 repeat의 평균 → 장면 평균). block 합계는 따로 적는다. 독립 표본은 최대 12개 장면이다.
- 조기 성공·미도달 block을 전체 평균에서 빼지 않는다. 반환이 실제로 일어난 block만 따로 볼 때는 그 조건을 표시한다.
- 사건 비교의 분모는 **checkpoint에서 파지 중이던 모든 block**이고 네 arm 모두에서 같은 항목을 읽는다:
  제어 구간의 파지 flag 유지, 반환 뒤 첫 열기 명령(gripper 명령 > 0), 첫 파지 flag 해제, 그 시점의 큐브 A–B 수평거리·높이·on flag.
  열기 명령이 없거나 flag 해제가 없으면 "없음"으로 기록한다. 기준은 Stage 2C와 같다(접근 3 cm, 정렬 2 cm, 상승 1 cm, 큐브 B 이동 2 mm).
- "놓쳤다 / 충돌 / 시간 부족 / OOD / 시계 암기" 같은 원인 라벨을 붙이지 않는다. 결과로 나눈 표는 사후 하위집합이라고 표시한다.
- 구간, p-value, LTT, 임계값 탐색을 하지 않는다. `H(g)`, 온라인 제어기 성능, 5% 위험 보장이라고 부르지 않는다.

## 10. 예산

Stage 2D 상한은 US$1.00이다. 설치·전송·audit·trunk·참조 실행·재실행·백업을 모두 포함하고, credit 차이로 적는다.
Stage 2C 보고서의 $0.2~0.3은 과거 처리량을 환산한 추정이었고 견적이 아니다. 업로드는 80k 정책 하나와 코드·설정·plan·저장된 장면 좌표로 한정한다.

## 11. 이 단계로 할 수 없는 결론

- 시간과 위치의 인과 기여를 분해했다. (R5·R2M·H0M은 최종 위치만 다른 것이 아니라 속도·접촉·이력·경로가 모두 다르다.)
- H0M − C0가 시간 경과만의 효과다. (H0M에는 위치 유지 제어, queue 폐기, 관측 갱신, 호출 시점 변경, 예산 소모가 들어 있다.)
- R2M이 좋으면 "2 cm 이동 후 즉시 반환"하는 복구 동작이 검증됐다.
- 반환 뒤 실패가 많으므로 Retreat 자체는 무해하다.
- 37/100 정책이 qualification을 통과했다, 개입이 유익하거나 무해하다, 선택기 학습이 가능하다 또는 불가능하다.
- Stage 2A·2B·2D를 합친 확증 표본.

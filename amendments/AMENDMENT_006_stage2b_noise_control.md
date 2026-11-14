# 설정/프로토콜 변경 기록 006 — Stage 2B 대체 난수 Continue 대조 (arm 추가)

기존 버전: v0.3 + AMENDMENT_001~005 (저장소 HEAD `d82168b`)
새 버전: v0.3 + AMENDMENT_001~006
변경 시점: 2026-11-14 (UTC). Stage 2B의 어떤 rollout도 실행하기 전. GPU 대여 전.
변경 사유: Stage 2A에서 Continue 자체가 repeat(미래 난수)에 따라 91개 snapshot 중 14개에서 결과가 달라졌다.
Refresh와 Retreat–Retry도 Continue와 다른 난수를 쓰므로, Stage 2A의 Rescue/Harm 건수만으로는 "다른 난수 실현의 Continue"를 기준으로 했을 때 방향이 유지되는지 알 수 없다.
사용자가 `prompts/STAGE2B_NOISE_CONTROLLED_RECOVERY_PILOT_AGENT_PROMPT_v0_1.md`로 네 번째 대조 arm을 넣은 개발 실험을 결정했다.
이것은 **Stage 2A 결과를 본 뒤에 정한** 설계다. 사전 등록된 확증 실험도, 독립 검증도 아니다.

모델, 추론 설정, horizon, 성공 판정, 개입 정의, Retreat 설정, 복원 방식, 허용 오차, checkpoint, repeat는 바꾸지 않는다.

## 1. 상태

```text
base_policy_qualification: BELOW_ORIGINAL_GATE (37/100 < 50/100)
stage2a: COMPLETED_EXPLORATORY_ONLY
stage2b: EXPLORATORY_FOUR_ARM_CONTROL
formal_study_ready: NOT_ESTABLISHED
calibration: NOT_RUN
final_test: NOT_RUN
```

하지 않는 것: 기본 정책 재학습, 체크포인트 재선택, qualification A/B 재실행, Retreat 변경, 행동 청크·추론 스텝 변경,
Q/Rescue/Harm 예측기 학습, 온라인 선택기 평가, LTT/calibration, 최종 Test, 5,760회 수집.

## 2. 추가되는 arm

| arm | 약칭 | snapshot의 남은 queue | 이후 첫 추론 | 정책 난수 repeat | `Runtime.force` |
|---|---|---|---|---:|---|
| `continue` | C0 | 유지 | 원래 chunk 경계 | r | 호출 안 함 |
| `continue_alt_noise` | C1 | 유지 | 원래 chunk 경계(C0와 같은 step) | r + 4 | 호출 안 함 |
| `refresh` | F | 폐기 | checkpoint에서 즉시 | r | 1 |
| `retreat_retry` | R | 폐기 | Retreat가 끝난 뒤 | r | 2 |

- C1은 **실험 대조군**이다. 복구 동작이 아니고 개입으로 세지 않는다. 기존 행동 번호(0·1·2)에 3을 더하지 않았고 `force(3)` 같은 경로는 없다.
- 두 인덱스를 분리해서 기록한다. `block_repeat_id = r`은 짝을 맞추는 키이고, `policy_noise_repeat_id`는 정책의 미래 난수 키다. C1만 r + 4이다(r = 0..3 → 4..7).
  `(r+1) % 4`처럼 다른 block의 Continue가 이미 쓴 키를 재사용하지 않는다.
- 난수 규칙 자체는 그대로다: 추론 호출 직전에 `seed_for(scene, 호출 시점의 physical step, policy_noise_repeat_id, 'dp')`를 `torch.random.fork_rng` 안에서 설정한다.
  실행 순서, worker 번호, 시각, arm 이름은 seed에 들어가지 않는다.
- C1의 적용 순서: snapshot 복원 → 복원된 실행 상태 전체를 snapshot 시점 기록과 대조(기존 허용 오차) → 그 뒤에 `Runtime.repeat`만 r + 4로 바꾼다.
  queue를 비우지 않고, history를 건드리지 않고, checkpoint에서 추론하지 않고, 행동을 더하지 않는다.
- F와 R은 Stage 2A와 같다. Refresh의 호출 시점이 Continue와 달라 난수가 다른 것은 정의된 개입의 일부이며, 같은 난수로 맞추려고 규칙을 바꾸지 않는다.
- 서로 다른 정수 seed는 "다른 의사난수 실현"일 뿐 통계적 독립의 증명이 아니다. C0와 C1의 결과가 같아도 정상이고, 달라야 통과하는 검사는 없다.

## 3. 코드 변경

| 파일 | 내용 |
|---|---|
| `src/hci/pilot2b.py` (신규) | 네 arm 수집기, C0/C1 queue 구간 일치 검사, audit 확장, 표. Stage 2A의 `pilot.py` 함수를 재사용한다 |
| `scripts/20_stage2b_pilot.py`, `scripts/stage2b/run.sh` (신규) | plan에 적힌 정책·설정·코드·장면만 실행하는 Stage 2B 전용 실행기 |
| `src/hci/policies.py` | `DiffusionPolicy.generate`가 방금 쓴 seed를 `last_seed`에 남긴다(기록용 한 줄). seed 값과 추론 경로는 그대로다 |
| `tests/test_stage2b_pilot.py` (신규) | 아래 5절의 검사 |

- `pilot.py`, `simulation.py`, 기존 `hci collect`/`hci audit`, Stage 1D·2A 실행기는 수정하지 않았다. 기준을 건너뛰는 전역 옵션은 없다.
- `policies.py`가 한 줄 바뀌었으므로 Stage 2A plan의 코드 hash와는 더 이상 일치하지 않는다. Stage 2A는 완료된 기록이며, 다시 실행하려면 커밋 `d82168b`를 써야 한다.

## 4. 장면과 실행량

- 새 장면: `configs/splits.json`의 train에서 Stage 2A 장면(10000~10023)과 다른 용도로 선언된 seed를 빼고 수치 오름차순 앞 24개. 역할 이름 `stage2b_exploratory_dev`.
- 초기 큐브 배치 대조: 시연 원본 1000개와 Stage 2A 장면 24개는 **이미 저장된 좌표**로, audit·dev·policy 평가 집합은 서버에서 reset해서 비교한다.
  calibration과 Test는 reset하지 않고 seed 번호로만 대조한다. 같은 배치(1e-5 m 이내)가 있으면 수집하지 않고 목록을 고친 plan을 새로 쓴다.
- 목록은 이 정책을 그 장면에서 한 번도 실행하기 전에 확정한다. train이 모자라면 다른 split에서 빌리지 않는다.
- Stage 2A와 2B의 장면은 앞으로 calibration/Test로 옮기지 않는다.
- checkpoint `[12, 52, 92, 132]`, block repeat `[0, 1, 2, 3]`. snapshot은 **기존 세 선택이 모두 가능할 때만** 만든다(C1 때문에 조건을 느슨하게 하지 않는다).
- 상한: 장면 24, snapshot 96, block 384, continuation 1,536(= 24 × 4 × 4 × 4). 별도로 trunk 24회, 복원 audit, 재현 검사 최대 16회.
- 처음 2개 장면(최대 128 continuation)으로 기술 조건만 확인한다. 축소 순서는 24 → 12 → 2이고 실행 시간과 비용으로만 정한다. 결과의 유불리는 입력이 아니다.
- 조기 성공이나 부적격으로 빠진 칸은 사유와 함께 남기고 새 장면으로 채우지 않는다.

## 5. 검사

복원 audit(이 host에서 새로 실행): Stage 2A의 비교 항목과 허용 오차 그대로. 추가로
(1) C1 분기를 끝까지 실행한 뒤 복원해서 Continue trace를 다시 비교하고,
(2) 정책 입력을 저장하는 함수가 실행 상태(simulator, queue, history, 난수)를 전혀 바꾸지 않는지 완전 일치로 확인한다.

수집 중:

| 검사 | 범위 | 실패 시 |
|---|---|---|
| 분기 시작 전 복원 상태 전체 대조 | 모든 분기 | 그 장면을 오류로 남기고 중단 |
| C0/C1 queue 구간 일치: 남은 queue를 다 쓸 때까지 명령·simulator 상태·환경이 돌려준 관측이 같고 그동안 추론이 없으며, 첫 추론이 `t + queue 길이`에서 일어남 | 모든 block | 장면 오류(자료가 아니라 구현 결함으로 취급) |
| repeat 0의 C0가 복원하지 않은 trunk와 일치 | 모든 snapshot | 재현성 문제로 표시, 자료를 완료로 표시하지 않음 |
| r=0의 C1 재실행, r=1의 C0 재실행(다른 분기를 모두 실행한 뒤) | 처음 2개 장면의 모든 snapshot | 위와 같음 |
| 추론 호출마다 seed가 선언한 키와 같은지 | 모든 호출 | 자료를 완료로 표시하지 않음 |

남은 queue를 실행하는 중에 성공해서 이후 추론이 없는 block은 C0와 C1이 같게 끝난다. 오류가 아니다.
허용 오차를 넓히거나, 불일치 횟수를 빼거나, 재현된 사례만 고르지 않는다.

## 6. 기록

- 분기마다: `experimental_arm`, `block_repeat_id`, `policy_noise_repeat_id`, 추론 호출별 step·seed·새 chunk의 hash, 첫 추론 step, 그리고 Stage 2A의 기록 항목.
- snapshot마다 **결과가 나오기 전에 한 번** 온라인 입력을 저장한다(`branches/inputs/<scene>.npz`): RGB history(uint8), qpos/qvel/TCP history, 마지막 행동, 남은 queue, t, 남은 예산, 고정 정책의 encoder 출력.
  repeat나 arm마다 같은 이미지를 다시 저장하지 않는다. **이 단계에서 학습은 하지 않는다.**
- 큐브 pose, 파지 flag, 단계 라벨, 결과는 장면 JSON과 trace 쪽에만 있다. 온라인 입력 파일에 넣지 않고 정책에도 주지 않는다. 구분은 `observation_fields.json`에 적는다.
- 영상: 처음 2개 장면, 첫 유효 checkpoint, r=0의 네 arm(최대 8개). 결과를 보고 예시를 추가하지 않는다.

## 7. 분석

같은 block 안에서 여섯 쌍을 모두 센다: F–C0, R–C0, C1–C0, F–C1, R–C1, R–F. 각 쌍에 네 범주(0→1, 1→0, 1→1, 0→0)와 분자·분모를 적는다.

- C1–C0의 0→1, 1→0은 **noise outcome switch**라고 부른다. Rescue/Harm이나 baseline의 안전 위반이라고 부르지 않는다.
- `delta_F_C0 − delta_C1_C0`는 대수적으로 `delta_F_C1`과 같다. "난수를 보정한 순수 효과"가 아니다.
  `Harm(action) − Harm(C1)`을 순수 Harm이라고 쓰지 않고, 음수를 0으로 잘라 보장 지표로 쓰지 않는다.
- 주된 요약은 장면 동일 가중이다: 장면 안에서 도달한 checkpoint와 repeat의 평균을 낸 뒤 장면 평균을 낸다. 유효 checkpoint가 없는 장면은 넣지 않고 수를 따로 적는다.
- Retreat의 보완 가치: (C0 실패, F 실패, R 성공), (C0 성공, F 성공, R 실패), F/R 2×2, 같은 snapshot의 몇 개 repeat에서 나타나는지, 기준을 C1로 바꿨을 때의 같은 표.
- 독립 표본은 최대 장면 24개다. 구간, p-value, 위험 보장용 상한을 계산하지 않는다. 결과를 본 뒤 정한 임계값을 사전 규칙처럼 쓰지 않는다.
- block별 최댓값은 실행 가능한 선택기가 아니다. checkpoint에서 선택을 강제한 결과는 온라인 정책의 `H(g)`가 아니다.
- Stage 2A와 2B는 따로 보고한다. 합쳐서 확증 표본이라고 부르지 않는다.
- 네 번째 arm은 원래 재계획 시점의 재표본 변동에 대한 대조다. Refresh의 구성 요소(새 관측, chunk 폐기, 이른 추론, 다른 난수)를 분리하지 못한다.

## 8. 예산

Stage 2B 상한은 US$1.50이다. Stage 2A($0.38), Stage 1D와 따로 기록한다. 비용은 credit 차이로 적는다.
업로드는 선택된 80k 정책 하나와 코드·설정·plan·저장된 장면 좌표로 한정한다.

## 9. 이 단계로 할 수 없는 결론

- 37/100 정책이 qualification을 통과했다.
- 복구 개입이 유익하다, 무해하다, 또는 통계적으로 유의하다.
- 온라인 개입 정책의 성능이나 악화율.
- 행동의 물리적 효과와 난수 효과를 완전히 분리했다.
- 정식 연구(calibration, Test)를 시작할 준비가 됐다.

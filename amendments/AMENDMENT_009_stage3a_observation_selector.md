# 설정/프로토콜 변경 기록 009 — Stage 3A 저장 관측 기반 선택기의 최초 학습 (CPU 전용)

기존 버전: v0.3 + AMENDMENT_001~008 (저장소 HEAD `205d55e`)
새 버전: v0.3 + AMENDMENT_001~009
변경 시점: 2026-11-15 (UTC). 이 단계의 어떤 전처리·학습·예측도 실행하기 전.
변경 사유: Stage 2D까지 거리와 대기 시간을 바꿔 봤지만 평균이 더 좋은 arm은 확립되지 않았다.
사용자가 `prompts/STAGE3A_OBSERVATION_BASED_SELECTOR_FEASIBILITY_AGENT_PROMPT_v0_1.md`로 방향을 바꿨다:
arm을 더 찾는 대신, 이미 저장된 관측만으로 "여기서 개입할지"를 예측할 수 있는지 작은 모델을 실제로 학습해 본다.

이것은 **Stage 2B·2C·2D의 집계와 사례를 이미 여러 번 본 뒤에 정한** 개발 실험이다.
장면을 통째로 빼고 평가하더라도 새로운 독립 검증이 아니고, 확증 실험도 최종 Test도 아니다.

## 1. 이번에 새로 허용하는 것과 그대로인 것

1. AMENDMENT_005~008의 "예측기·선택기를 학습하지 않는다"는 그 단계들의 범위에서 그대로 유효하다. **이 단계에서만** CPU 선택기 학습을 허용한다.
2. Stage 2B와 2D의 **공통 arm(Continue, 기존 5 cm Retreat)** 자료를 개발용 학습 pool로 함께 쓰는 것을 새로 허용한다.
   두 단계의 기존 표를 합쳐 확증 결과로 바꾸지 않는다. 기존 보고서·결과 파일·hash는 수정하지 않고, 새 분석도 단계별로 따로 적는다.
3. 거리, 기본 정책, 난수 규칙, 성공 정의, snapshot 시점은 바꾸지 않는다. 새 simulator 실행, 기본 정책 추론·재학습, GPU 대여는 없다. 유료 비용 $0.
4. R5를 유익한 복구 동작으로 채택한 것이 아니다. "쓰지 않음"을 포함해 선택 가능성을 시험하는 후보로만 둔다.

```text
base_policy_qualification: BELOW_ORIGINAL_GATE (37/100 < 50/100)
stage3a: EXPLORATORY_OBSERVATION_BASED_SELECTOR_FEASIBILITY
base_policy_retraining: NOT_RUN
new_simulator_rollouts: NOT_RUN
observation_based_selector: NOT_TRAINED -> TRAINED_DEV_ONLY (완료 시)
formal_study_ready: NOT_ESTABLISHED
calibration: NOT_RUN
final_test: NOT_RUN
```

쓰지 않는 자료: Stage 2A 전체, Refresh, 대체 난수 Continue(C1), R2M, H0M.
2D에 없는 arm을 0이나 실패로 채우지 않고, 없는 arm의 결과를 예측해서 학습 자료로 만들지도 않는다.
R2M·H0M은 R5의 종료 시점을 미리 알아야 하는 진단용 arm이라 선택 후보가 아니다.

## 2. 자료

| 단계 | 장면 | snapshot | block | 결과 열 |
|---|---:|---:|---:|---|
| 2B | 24 | 84 | 336 | `y_C0`, `y_R` |
| 2D | 12 | 44 | 176 | `y_C0`, `y_R5` |
| 개발 pool | 36 | 128 | 512 | `y_continue`, `y_retreat5` |

- 키: `(stage, scene, checkpoint, snapshot_id, block_repeat_id)`. 라벨: `joint_label = 2*y_continue + y_retreat5` (0 둘 다 실패, 1 Rescue, 2 Harm, 3 둘 다 성공).
- 결과 값은 `'0'`/`'1'`만 받는다. 다른 문자열은 오류다.
- 512행은 512개의 독립 표본이 아니다. snapshot 하나의 같은 입력에 미래 난수 4회의 결과가 붙어 있다. 독립 단위는 장면 24 + 12개다.
- 입력은 분석 전에 hash를 manifest에 적고 분석 뒤 다시 대조한다. 읽기 전용이고, 출력은 입력 폴더 밖 `runs/stage3a_001/`에 쓴다.
- 저장된 온라인 입력 파일(NPZ)은 실제로 열어서 확인한다: 장면당 1개, 선언된 7개 항목만 있는지, snapshot 시점에 적어 둔 hash와 같은지, 서버 백업 목록의 hash와 같은지.
  필수 입력이 없으면 `PARTIAL_INPUT_UNAVAILABLE`로 남기고 멈춘다. GPU를 빌려 다시 만들지 않는다.

결합 전에 확인하는 것(하나라도 풀리지 않으면 결합 학습을 하지 않는다):

- 정책 가중치·metadata, 설정(horizon 200, Retreat, protocol), checkpoint·repeat, 성공 정의, 난수 키 규칙이 같은가.
- 실행 코드 가운데 의미를 정하는 파일(`simulation.py`, `policies.py`, `pilot.py` 등)이 두 단계 plan에서 같은 hash인가. 단계별 수집기·드라이버는 다르다는 것을 따로 적는다.
- 카메라 순서와 온라인 입력 정의가 같은가. `policy_feature`의 shape·dtype이 같고, 끝의 6개 값이 두 단계 모두 마지막 행동·`t/horizon`·queue 비율인가.
- 2D의 R5가 `Runtime.force(2)`이고, T_ref 참조 실행 뒤 상태와 난수가 그대로였다는 audit·기술 검사 기록이 있는가.
- 장면이 겹치지 않는가. 입력이 continuation 전에 저장됐는가. 모든 snapshot에서 repeat 0의 Continue가 trunk와 같은가.
- host와 GPU driver가 다르다는 사실은 단계 metadata로 남기고 입력에 넣지 않는다. 단계 간 차이의 원인은 가르지 않는다.

## 3. 모델 입력

- 모델에 넣는 것은 **저장된 `policy_feature`(568차원)와 checkpoint one-hot 4개**뿐이다.
  `policy_feature`는 고정 정책 encoder의 출력에 마지막 행동, `t/horizon`, queue 비율을 붙인 값으로, snapshot 시점에 결과가 나오기 전에 저장됐다.
- RGB를 새 encoder로 처리하지 않고, 기존 encoder를 다시 부르거나 미세조정하지 않는다. RGB·queue·state_history는 무결성 확인에만 쓴다.
- 넣지 않는 것: 장면·episode 번호, seed, 단계, host, driver, 파일 이름·경로·hash, snapshot id, repeat 번호 /
  큐브 pose, 파지·on-cube·static flag, 단계 라벨 / 성공·실패, trunk 결과, 성공 시점, 최종 상태, arm별 결과나 성공 수 /
  T_ref, R5가 걸린 step 수, 실제 이동량, 반환 뒤 상태, 미래 trace나 난수.
  R5의 소요 시간은 개입 전에 알 수 없으므로 입력이 될 수 없다.
- 코드에서 입력을 만드는 함수는 하나이고, 위 두 열이 아닌 것이 오면 거부한다. 입력 파일을 읽는 함수와 결과 파일을 읽는 함수는 분리한다.
- 이 제한 때문에, 결과가 좋지 않아도 "다른 표현으로도 선택이 불가능하다"고 말할 수 없다.

## 4. 평가 방식

**장면 단위 분리(LOSO)**: 36개 fold. fold마다 한 장면의 모든 checkpoint·repeat·라벨을 빼고 나머지 35개 장면으로 학습한다.

fold 안에서(학습 쪽의 서로 다른 snapshot만으로 맞춤, repeat로 늘리기 전):

1. `policy_feature`에 StandardScaler
2. PCA 8차원 (`svd_solver=full`, `whiten=False`)
3. PC 점수에 StandardScaler
4. checkpoint one-hot `[12, 52, 92, 132]`를 뒤에 붙임 → 12차원

빠진 장면에는 transform만 한다. 8차원은 처음부터 고정이고 결과를 보고 다른 차원을 시험하지 않는다.

가중치: 장면 j의 snapshot이 m_j개면 그 장면의 각 repeat 행은 `1/(m_j*4)`. 학습 fold에서 평균이 1이 되게 맞춘다. 모든 모델이 같은 행과 가중치를 쓴다.
`class_weight=balanced`나 oversampling은 쓰지 않는다.

예측 파일은 빠진 장면의 라벨을 붙이기 전에 쓰고 hash를 남긴다.

**단계 간 진단**: 2B로 학습해 2D를 읽기, 2D로 학습해 2B를 읽기를 같은 설정으로 한 번씩. 두 단계 모두 이미 본 자료이고, 장면·host·단계 차이가 섞여 있어 원인을 특정하지 않는다.

## 5. 모델 세 가지와 결정 규칙

공통: scikit-learn `LogisticRegression(solver=lbfgs, C=1.0, fit_intercept=True, class_weight=None, tol=1e-7, max_iter=2000)`. 실제 버전은 manifest에 적는다.

| 모델 | 입력 | 출력 | score |
|---|---|---|---|
| `M_time` | checkpoint one-hot 4개 | Continue 성공확률, R5 성공확률(이진 head 2개) | `q_R − q_C` |
| `M_Q` | PC 8 + one-hot 4 | 위와 같음 | `q_R − q_C` |
| `M_pair` | PC 8 + one-hot 4 | joint_label 4분류 확률 | `p(Rescue) − p(Harm)` |

- 결정: `score > 0`이면 R5, 아니면 Continue. 0은 "기대 순이득이 0"이라는 경계이고 Harm 허용률이 아니다. 위험 제약, calibration, 통계적 보장은 없다.
- threshold, harm penalty, 규제 강도, PCA 차원을 탐색하지 않는다.
- `M_time`은 관측 모델이 "지금이 몇 step인가"만 쓰는지 가려내기 위한 비교군이다.
- `M_Q`는 지도학습 성공확률 baseline이다. Bellman backup이나 offline RL이 아니다.
- `M_Q`와 `M_pair`는 모집단에서는 같은 기대 성공률 차이를 나타낼 수 있다. 이름이 다르다고 새 정보가 생기지 않는다.
- 학습 fold에 class가 하나뿐이면: 이진 head는 그 값의 상수 확률, 4분류는 그 class의 상수. class가 일부만 있으면 있는 class만 학습하고 없는 class의 확률은 0.
- 수렴 경고가 나면 `max_iter=10000`으로 한 번만 다시 맞춘다(모든 모델 같은 규칙, 기록). 그래도 안 되면 실패로 남기고 fold를 빼지 않는다.

학습하지 않는 비교군(같은 R5, 최대 1회 개입):

- **Never**: 원래 trunk.
- **timer 12 / 52 / 92 / 132**: 그 step에서 아직 실행 중이고 적격이면 R5. 네 개 모두 보고한다.
- **학습 장면으로 고른 timer**: fold의 학습 장면에서만 {Never, 12, 52, 92, 132}를 비교해 성공 장면 수가 가장 많은 것. 동률이면 Never, 그다음 12 → 52 → 92 → 132.

## 6. 두 가지 읽기

**(1) 고정 checkpoint에서의 선택**: 빠진 장면의 snapshot마다 관측으로 고른 arm을 그 snapshot의 repeat 4개에 똑같이 적용한다.
고른 arm의 평균 결과와 Continue 대비 차이, 0→1·1→0 건수(분자·분모·장면 수), R5를 고른 비율, Brier score, 예측 분포를 적는다.
주 집계는 장면 안 평균 → 장면 평균이고, 2B와 2D를 따로 적는다. 36개 전체 평균은 24:12로 섞인 보조 표다.

**(2) 저장된 경로에서 1회 개입 결정을 읽기**: 새 실행이 아니다. 모든 snapshot은 repeat 0 trunk의 과거에서 나왔으므로 **repeat 0만** 쓴다.
빠진 장면마다 그 장면 없이 학습한 모델로 checkpoint를 12 → 52 → 92 → 132 순서로 읽고, score가 처음 0을 넘은 곳에서 저장된 R5 repeat 0 분기의 결과를 읽고 끝낸다.
한 번도 넘지 않으면 trunk 결과다. 뒤의 더 큰 score를 보고 돌아가지 않는다. 이미 성공해 checkpoint가 없으면 개입을 가정하지 않는다.
설명 없는 누락은 채우지 않고 오류로 본다. 계산 지연의 영향은 이 읽기로 평가되지 않는다.

- 기대값: trunk 성공 2B 8/24, 2D 3/12. 다시 세서 다르면 계산을 멈추고 join부터 확인한다.
- fold마다 모델이 다르다. 이 표는 하나의 최종 모델을 검증한 결과가 아니다.

## 7. 판단 기준(실행 전에 적어 둠)

- 관측 모델이 Never, 학습 장면으로 고른 timer, `M_time`보다 여러 장면에서 낫고 한 단계에만 기대지 않으면: 고정한 후보 하나를 새 개발 장면에서 실제로 실행해 볼 가치가 있다고 보고한다. 자동으로 실행하지 않는다.
- 관측 모델과 시간만 쓰는 규칙이 비슷하면: 관측 정보의 추가 가치가 확인되지 않았다고 보고한다. 더 복잡한 모델을 자동으로 더하지 않는다.
- `M_Q`와 `M_pair`가 비슷하거나 `M_Q`가 더 좋으면: 도움·악화를 나눠 예측하는 쪽이 낫다고 주장하지 않는다.
- 항상 Continue만 고르면: Harm 0을 성과로 쓰지 않는다. 개입하지 않아서 생긴 값이다.
- 음수이거나 한두 장면에 달려 있으면: 이 자료·표현·고정 모델로는 추가 이득이 확인되지 않았다고 보고한다. 선택 일반이 불가능하다고 쓰지 않는다.

p-value, 구간, calibration, 위험 상한은 계산하지 않는다. 좋은 결과가 나오지 않아도 완료된 실험은 완료로 기록한다.

LOSO와 단계 간 계산을 모두 마친 뒤, 같은 설정의 `M_Q`·`M_pair`를 36개 장면 전체로 한 번씩 맞춰 저장한다.
새 장면 실험을 위한 후보일 뿐이고 `DEV_ONLY_NOT_INDEPENDENTLY_EVALUATED`로 표시한다. 새 실행 권한은 없다.

## 8. 코드 변경

| 파일 | 내용 |
|---|---|
| `scripts/24_stage3a_selector.py` (신규) | manifest, 무결성·공통성 검사, fold 학습, 예측, 두 가지 읽기, 단계 간 진단, 전체 pool 저장 |
| `configs/stage3a_selector.json` (신규) | 입력 위치, 기대 수, 고정한 설정 |
| `tests/test_stage3a_selector.py` (신규) | 합성 자료 검사 |

기존 runtime, 수집기, Stage 2 결과 파일은 수정하지 않는다. 분석 코드는 runtime 모듈을 import하지 않는다.

## 9. 이 단계로 할 수 없는 결론

- R5가 유익하다, 선택기가 안전하다, 통계적으로 유의하다.
- 관측 기반 선택이 일반적으로 가능하다 또는 불가능하다.
- 새 장면에서의 폐루프 성능. (저장된 경로를 읽은 것이다.)
- 온라인 정책의 악화율 `H(g)`나 5% 같은 위험 보장.
- 37/100 정책이 qualification을 통과했다, 정식 연구를 시작할 준비가 됐다.

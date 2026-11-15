# Stage 보고서 — Stage 3A 저장 관측 기반 선택기의 최초 학습

**"이 장면의 분기 결과를 보지 않고, 실행 시 관측만 쓴 작은 모델이 Continue보다 나은 결정을 했는가? 시간만 쓰는 규칙보다도 나았는가?" — 둘 다 아니다.** 장면을 통째로 빼고 학습한 관측 모델은 두 단계 어느 쪽에서도 성공 장면 수가 Continue(개입 안 함)를 넘지 못했고(얻은 장면 1~2개, 잃은 장면 2개), 시간만 쓰는 규칙보다 낫지도 않았다. Stage 2B 장면에서는 결과가 Continue와 같았고, Stage 2D 장면에서는 조금 낮았다. 이 자료, 이 표현(저장된 `policy_feature`의 주성분 8개), 이 고정 모델에 대한 개발용 결과이며 관측 기반 선택 일반이 불가능하다는 뜻은 아니다.

실행 ID: `stage3a_001`
단계: Stage 3A — Continue / 기존 R5 선택기의 장면 분리 개발 실험 (로컬 CPU, 새 실행 없음)
작업/정책 seed: StackCube-v1 / 정책 `stack_1d_s1_d500_clean_v1_i080000`의 저장 기록(재학습·추론 없음)
실제 실행 환경: 로컬 PC, Windows 11, Python 3.14.4, numpy 2.5.4, scikit-learn 1.9.0, scipy 1.18.0. GPU·simulator 사용 없음. 2026-11-15 01:25~01:26 UTC
코드/upstream 버전: 저장소 `205d55e` + 이 단계의 새 파일(파일 hash를 manifest에 고정)
변경 기록: `amendments/AMENDMENT_009_stage3a_observation_selector.md`

## 상태

```text
base_policy_qualification: BELOW_ORIGINAL_GATE (37/100 < 50/100)
stage3a: EXPLORATORY_OBSERVATION_BASED_SELECTOR_FEASIBILITY (completed)
base_policy_retraining: NOT_RUN
new_simulator_rollouts: NOT_RUN
observation_based_selector: TRAINED_DEV_ONLY
formal_study_ready: NOT_ESTABLISHED
calibration: NOT_RUN
final_test: NOT_RUN
```

Stage 2B·2C·2D의 집계와 사례를 이미 여러 번 본 뒤에 정한 실험이다. 장면을 빼고 평가했어도 새로운 독립 검증이 아니다.

## 요약

- 입력은 모두 있었고 온전했다. 36개 장면, snapshot 128개, block 512개를 다시 세어 Stage 2B·2D 보고서의 수와 같았다. 저장된 온라인 입력은 snapshot 시점에 적어 둔 hash, 서버 백업 목록과 모두 같았다.
- 두 단계의 공통 arm은 같은 정책·설정·runtime 코드·난수 규칙·입력 정의로 수집됐다. 공통성 검사 12개 항목이 모두 통과해 결합 학습을 진행했다. host와 GPU driver는 다르다.
- 학습은 문제없이 끝났다. 38번의 fit(LOSO 36 + 단계 간 2) 모두 첫 시도에 수렴했고, class가 하나뿐인 fold는 없었다.
- **저장된 경로에서 읽은 1회 개입 결정**(장면당 한 번, repeat 0)의 성공 장면 수:

| | Never | 학습 장면으로 고른 timer | `M_time` | `M_Q` | `M_pair` |
|---|---:|---:|---:|---:|---:|
| 2B (24개 장면) | 8 | 8 | 8 | 8 | 8 |
| 2D (12개 장면) | 3 | 3 | 3 | 2 | 3 |

- 2D에서 `M_Q`는 장면 10054를 얻고 10051·10056을 잃었다. `M_pair`는 10050·10054를 얻고 10051·10056을 잃었다. 2B에서는 얻은 장면도 잃은 장면도 없다.
- **고정 checkpoint에서의 선택**(같은 선택을 repeat 4개에 적용): 2B에서 `M_Q`는 Continue와 같았고(93 대 93), 2D에서는 낮았다(20 대 25, 장면 평균 차이 −0.042).
- 관측 모델이 R5를 고른 snapshot은 대부분 어느 쪽도 성공하지 못한 곳이었다(2B에서 `M_Q`가 고른 31개 전부). R5가 실제로 더 좋았던 snapshot 17개 중에서는 2개만 골랐다.
- 사전에 적어 둔 기준으로는 "음수이거나 한두 장면에 달려 있음"에 해당한다. 비용 $0.

## 실제 수행한 명령

| 명령 | 종료 코드 | 로그 경로 | 시작/종료 시각 (UTC) |
|---|---:|---|---|
| `python -m pytest -q` | 0 (204 passed, 3 skipped) | 터미널 | 2026-11-15 01:24 |
| `codex exec … (실행 전 코드 검토)` | 0 (결함 보고 없음) | 세션 기록 | 2026-11-15 |
| `24_stage3a_selector.py manifest --out runs/stage3a_001` | 0 | `logs/s3a_manifest.log` | 01:25:57 / 01:26:05 |
| `24_stage3a_selector.py verify --label input_hash_check_before` | 0 | `logs/s3a_verify_before.log` | 01:26:05 / 01:26:06 |
| `24_stage3a_selector.py run --out runs/stage3a_001` | 0 | `logs/s3a_run.log` | 01:26:06 / 01:26:28 |
| `24_stage3a_selector.py verify --label input_hash_check_after` | 0 | `logs/s3a_verify_after.log` | 01:26:28 / 01:26:31 |

실행 순서: 자료 확인 → 코드와 합성 테스트 → manifest(입력 86개 파일, 코드, 설정의 hash) → 학습과 예측 → 예측 파일 hash 기록 → 라벨을 붙여 평가 → 전체 pool 모델 저장 → 입력 재대조.
manifest를 쓰기 전에 실제 자료로 한 것은 무결성·공통성 검사뿐이고, 그때는 어떤 모델도 맞추지 않았다. `run`은 한 번만 실행했다.

## 산출물

원본은 `runs/stage3a_001/`(Git 제외), 아래는 저장소에 넣은 사본이다(SHA256 앞 16자리).

| 파일 | 용도 | SHA256 |
|---|---|---|
| `results/stage3a/analysis_manifest.json` | 분석 전 고정한 입력·코드·설정 hash, 라이브러리 버전 | `95b497b9cd455749` |
| `results/stage3a/input_integrity.json` | 재검산, 입력 대조, 공통성 검사 | `0194b18e5dbbe15c` |
| `results/stage3a/feature_schema.json` | 모델 입력과 입력 금지 항목 | `e3aef346c367508a` |
| `results/stage3a/scene_folds.json` | 36개 fold의 학습·제외 장면 | `12ef8bdb20bb8b32` |
| `results/stage3a/model_config.json` | 모델·가중치·결정 규칙 | `5d2ea9a4211647d0` |
| `results/stage3a/oof_predictions.csv` | 라벨 없이 쓴 예측 768행(LOSO 384 + 단계 간 384) | `b90d529c0f41a4c0` |
| `results/stage3a/snapshot_decisions.csv` | snapshot별 score와 선택 | `df4b2daf0d417344` |
| `results/stage3a/cached_episode_decisions.csv` | 장면별·방법별 결정 checkpoint와 참조 분기 | `e313b293430a84fe` |
| `results/stage3a/stage_separated_summary.json` | LOSO 집계(단계별, 보조 pool) | `8d268cc765b4b121` |
| `results/stage3a/stage_transfer_summary.json` | 단계 간 집계 | `e167ac7c0ddc70ba` |
| `results/stage3a/fold_fit_diagnostics.json` | fold별 수렴·class 수·선택된 timer | `a7283c106e4266ba` |
| `results/stage3a/models/M_Q_dev_pool.json`, `M_pair_dev_pool.json` | 36개 장면 전체로 맞춘 계수(`DEV_ONLY_NOT_INDEPENDENTLY_EVALUATED`) | `696739bff030cded`, `28028e0365eebbfd` |
| `results/stage3a/input_hash_check_after.json` | 분석 뒤 입력 불변 확인 | `c0bd1e210c96332e` |
| `scripts/24_stage3a_selector.py`, `configs/stage3a_selector.json` | 분석 코드, 설정 | `4c5788c130718ab3`, `62cdf38693619ad0` |
| `tests/test_stage3a_selector.py` | 합성 자료 검사 26개 | `8921ca43fb3fc55e` |

## 검증

| 항목 | PASS / FAIL / NOT RUN | 실제 근거 |
|---|---|---|
| CPU tests | PASS | 204 passed / 3 skipped. Stage 3A 테스트 26개는 합성 자료만 쓴다 |
| CUDA | NOT RUN | 이 단계는 GPU를 쓰지 않는다 |
| Vulkan/RGB | NOT RUN | simulator·렌더링 없음 |
| 정책 rollout | NOT RUN | 새 rollout 없음. 저장 기록만 읽음 |
| 상태+정책 복원 | NOT RUN | 해당 없음. Stage 2B·2D의 audit 기록을 공통성 검사에서 읽기만 함 |
| 개입 행동 영상 | NOT RUN | 해당 없음 |
| seed split | PASS | 2B 24개와 2D 12개 장면이 겹치지 않음. fold마다 한 장면의 snapshot·repeat·라벨이 전부 학습에서 빠짐 |
| 입력 무결성 | PASS | 재검산 2B 24 / 84 / 336, 2D 12 / 44 / 176. C0·R5 성공 수 93 / 39, 25 / 32. trunk 성공 8/24, 3/12. NPZ 36개가 snapshot 시점 hash, 백업 목록과 일치 |
| 공통성 | PASS | 12개 항목: 정책·설정·checkpoint·성공 정의·runtime 코드·카메라와 입력 정의·난수 규칙·장면 분리·2D R5 = `force(2)`·참조 실행 뒤 상태 불변·입력이 결과보다 먼저 저장·두 자료 완료 |
| 누출 방지 | PASS | 예측 파일을 먼저 쓰고 hash를 남긴 뒤 라벨을 붙임. 테스트: 빠진 장면의 라벨이나 feature를 바꿔도 학습과 예측이 변하지 않음 |
| 학습 수렴 | PASS | 38 fit × head 5개 모두 `max_iter=2000`에서 수렴, 재시도 0, class 하나뿐인 head 0 |
| 데이터 완료 상태 | PASS | `run_status.json` completed. 입력 86개 파일 분석 전후 변화 없음 |
| 과학적 가설 | NOT RUN | 이 단계는 가설 검증이 아니다 |

## 관측 결과

원본 episode 수: 36 (2B 24, 2D 12). 새로 실행한 것은 없다
분기 실행 수: 0 (읽은 저장 block 512개: 2B 336, 2D 176)
성공/실패 수: 아래 표
Rescue/Harm 분자와 분모: 아래 "고정 checkpoint에서의 선택" 표의 0→1, 1→0
시간/메모리/저장량: 학습·평가 22초, 산출물 0.8 MB

모델: `M_time`은 checkpoint만, `M_Q`와 `M_pair`는 `policy_feature`의 주성분 8개와 checkpoint를 쓴다. 모두 score > 0이면 R5를 고른다. 주성분 8개는 학습 snapshot 분산의 약 59%를 담는다(첫 fold 기준).

### 읽기 1: 고정 checkpoint에서의 선택 (LOSO, 같은 선택을 repeat 4개에 적용)

| 2B: 24개 장면, snapshot 84, block 336 | R5를 고른 snapshot | 성공 block | 장면 평균 차이 | 0→1 (장면) | 1→0 (장면) |
|---|---:|---:|---:|---:|---:|
| 항상 Continue | 0 | 93 | 0 | 0 / 243 | 0 / 93 |
| 항상 R5 (Stage 2B의 기존 결과) | 84 | 39 | −0.235 | 15 (5) | 69 (13) |
| `M_time` | 15 | 93 | 0.000 | 0 | 0 |
| `M_Q` | 31 | 93 | 0.000 | 0 | 0 |
| `M_pair` | 35 | 92 | −0.003 | 0 | 1 (1) |

| 2D: 12개 장면, snapshot 44, block 176 | R5를 고른 snapshot | 성공 block | 장면 평균 차이 | 0→1 (장면) | 1→0 (장면) |
|---|---:|---:|---:|---:|---:|
| 항상 Continue | 0 | 25 | 0 | 0 / 151 | 0 / 25 |
| 항상 R5 (Stage 2D의 기존 결과) | 44 | 32 | +0.030 | 25 (7) | 18 (4) |
| `M_time` | 9 | 25 | 0.000 | 0 | 0 |
| `M_Q` | 22 | 20 | −0.042 | 4 (2) | 9 (2) |
| `M_pair` | 25 | 21 | −0.036 | 5 (2) | 9 (2) |

- `M_time`이 R5를 고른 곳은 전부 step 132이다. 그 시점에는 두 arm 모두 거의 성공하지 못해서 선택이 결과를 바꾸지 않았다.
- 관측 모델이 R5를 고른 snapshot의 대부분은 Continue도 R5도 4번 모두 실패한 곳이다: 2B의 `M_Q` 31개 중 31개, `M_pair` 35개 중 34개, 2D의 `M_Q` 22개 중 16개, `M_pair` 25개 중 19개.
- R5가 Continue보다 많이 성공한 snapshot은 2B에 7개, 2D에 10개 있다. `M_Q`와 `M_pair`는 2B의 7개 중 0개, 2D의 10개 중 2개에서 R5를 골랐다.
- Continue가 더 많이 성공한 snapshot(2B 25개, 2D 7개)에서는 2B에서 `M_Q` 0개·`M_pair` 1개, 2D에서 둘 다 4개에서 R5를 골랐다.
- 보조(36개 장면, 24:12로 섞임): 장면 평균 차이 `M_time` 0.000, `M_Q` −0.014, `M_pair` −0.015.

예측 확률의 Brier score(장면 평균, 낮을수록 좋음):

| | Continue 성공확률 2B / 2D | R5 성공확률 2B / 2D |
|---|---|---|
| `M_time` | 0.220 / 0.165 | 0.116 / 0.164 |
| `M_Q` | 0.125 / 0.193 | 0.119 / 0.154 |
| `M_pair` | 0.126 / 0.191 | 0.122 / 0.166 |

2B 장면에서는 관측 모델이 Continue의 성공을 시간만 쓰는 모델보다 잘 맞혔고(0.125 대 0.220), 2D 장면에서는 더 못 맞혔다(0.193 대 0.165). R5의 성공확률은 세 모델이 비슷했다. `M_pair`가 준 평균 확률은 둘 다 실패 0.67, Rescue 0.08, Harm 0.19, 둘 다 성공 0.06이다.

### 읽기 2: 저장된 경로에서 1회 개입 결정 (LOSO, repeat 0만, 새 실행 아님)

| 방법 | 2B 성공 / 24 | 2B 개입한 장면 | 2D 성공 / 12 | 2D 개입한 장면 | 얻은 장면 | 잃은 장면 |
|---|---:|---:|---:|---:|---|---|
| Never (trunk) | 8 | 0 | 3 | 0 | — | — |
| timer 12 | 2 | 24 | 1 | 12 | 10043, 10054 | 2B 7개, 2D 3개 |
| timer 52 | 2 | 24 | 5 | 12 | 10043, 10050, 10052, 10057, 10059 | 2B 7개, 2D 2개 |
| timer 92 | 7 | 20 | 1 | 11 | 10035, 10043 | 2B 3개, 2D 2개 |
| timer 132 | 9 | 16 | 3 | 9 | 10043 | 없음 |
| 학습 장면으로 고른 timer | 8 | 15 | 3 | 9 | 없음 | 없음 |
| `M_time` | 8 | 15 | 3 | 9 | 없음 | 없음 |
| `M_Q` | 8 | 13 | 2 | 10 | 10054 | 10051, 10056 |
| `M_pair` | 8 | 15 | 3 | 10 | 10050, 10054 | 10051, 10056 |

- 학습 장면으로 고른 timer는 36개 fold 중 35개에서 timer 132였고, 10043을 뺀 fold에서만 Never였다(동률이면 Never). timer 132가 Never보다 나은 것은 10043 한 장면 때문이다.
- `M_Q`와 `M_pair`는 2B에서 13~15개 장면에 개입했지만 결과가 바뀐 장면이 없다. 2D에서는 10051과 10056(둘 다 trunk가 성공한 장면)에 step 12에서 개입해 성공을 잃었다.
- Stage 2B에서 Retreat가 유리했던 10035와 10043에서는 두 관측 모델 모두 개입하지 않았다.
- timer 52는 2D에서 5/12로 가장 높지만 2B에서는 2/24이다. 빠진 장면에서 가장 좋은 timer를 골라 비교하지 않는다.
- 어떤 방법도 두 단계 모두에서 Never보다 높지 않았다.

### 단계 간 진단 (한 단계로 학습해 다른 단계를 읽음. 두 단계 모두 이미 본 자료)

| | 고른 timer | Never | 고른 timer 결과 | `M_time` | `M_Q` | `M_pair` |
|---|---|---:|---:|---:|---:|---:|
| 2B로 학습 → 2D 12개 장면 | timer 132 | 3 | 3 | 3 | 3 | 3 |
| 2D로 학습 → 2B 24개 장면 | timer 52 | 8 | 2 | 2 | 2 | 2 |

- 2B → 2D: `M_Q`는 10054를 얻고 10056을 잃었다. snapshot 기준으로는 Continue보다 조금 높았다(성공 block 28 대 25, 0→1 4건, 1→0 1건).
- 2D → 2B: 세 모델 모두 2B의 24개 장면 전부에 step 12에서 개입해 성공 장면이 8개에서 2개로 줄었다. snapshot 기준 장면 평균 차이는 −0.21이다.
- 2D만으로 배운 규칙은 2B 장면에 맞지 않았다. 장면·host·단계의 차이가 섞여 있어 원인은 가를 수 없다.

## 해석 범위

사전에 적어 둔 판단 기준에 대해:

- "관측 모델이 Never, 고른 timer, `M_time`보다 여러 장면에서 낫고 한 단계에만 기대지 않는다" — 충족되지 않았다.
- "관측 모델과 시간만 쓰는 규칙이 비슷하다" — 최종 성공 장면 수는 비슷하거나 관측 모델이 하나 적다. 관측 정보의 추가 가치는 확인되지 않았다.
- "`M_Q`와 `M_pair`가 비슷하다" — 비슷하다(2D에서 2 대 3, 한 장면 차이). 도움·악화를 나눠 예측하는 쪽이 낫다고 할 근거가 없다.
- "항상 Continue만 고른다" — 아니다. 모델은 개입을 골랐고(128개 snapshot 중 53~60개), 그 대부분이 결과에 영향이 없는 곳이었다. Continue와 같은 성적은 "잘 피해서"가 아니라 "고른 곳에서 차이가 없어서" 나온 것이다.
- "음수이거나 한두 장면에 달려 있다" — 해당한다. 2D의 차이는 장면 2~4개(10050, 10054, 10051, 10056)에서 나온다.

말할 수 없는 것:

- 관측 기반 선택이 일반적으로 불가능하다. 입력을 주성분 8개로 제한했고, RGB·queue·관절 이력을 직접 쓰지 않았고, 선형 모델 한 종류만 맞췄고, 독립 단위는 장면 36개다.
- 새 장면에서의 폐루프 성능. 읽기 2는 저장된 repeat 0 경로를 네 checkpoint에서만 읽은 것이다.
- 2B와 2D의 차이가 장면 때문인지 host·driver 때문인지.
- R5가 유익하다 또는 해롭다, 위험 보장, `H(g)`.
- fold마다 모델이 다르므로, 이 표는 저장한 전체 pool 모델의 검증 결과가 아니다.

## 변경과 실패

변경한 파일과 이유:

- 신규: `scripts/24_stage3a_selector.py`, `configs/stage3a_selector.json`, `tests/test_stage3a_selector.py`, `amendments/AMENDMENT_009_stage3a_observation_selector.md`, `results/stage3a/`, 이 보고서.
- 수정: `README.md`(Stage 3A 항목), `.gitattributes`(`results/stage3a/**`).
- 기존 runtime, 수집기, Stage 2 결과 파일은 수정하지 않았다. 분석 코드는 runtime 모듈을 import하지 않는다.
- Codex의 실행 전 검토에서 결함은 나오지 않았다. 그 뒤 manifest 전에 공통성 검사의 설명 문구 한 줄만 고쳤다(계산에는 영향 없음).

실패한 명령 및 원인: 없음.

수행하지 않은 일: GPU 대여, simulator 실행, 기본 정책 추론·재학습, Stage 2A·Refresh·C1·R2M·H0M의 사용, threshold·차원·규제 탐색, 다른 feature나 모델, p-value·calibration·위험 상한, 최종 Test.

비용: $0. 누적(변화 없음): Stage 1D 약 $1.14, 2A $0.38, 2B $0.30, 2D $0.21.

## 다음 단계

권고: **이 자료·표현·고정 모델로는 선택의 추가 이득이 확인되지 않았다고 기록하고 여기서 멈춘다.** 새 장면에서 실행해 볼 만한 후보는 이 결과로 정할 수 없다.

- 전체 pool로 맞춘 `M_Q`·`M_pair` 계수는 저장했지만 `DEV_ONLY_NOT_INDEPENDENTLY_EVALUATED`이고, 위 결과로 볼 때 실행 후보로 권하지 않는다.
- 더 복잡한 모델이나 다른 입력을 자동으로 시험하지 않았다.

필요 작업: 사용자의 결정.
예상 계산량의 산출 근거: 이 단계는 로컬 CPU에서 22초가 걸렸다. 다음 작업의 계산량은 방향이 정해져야 산정할 수 있다.
동결/승인이 필요한 사항: 없음. 정책, 설정, 후보 목록은 그대로다.

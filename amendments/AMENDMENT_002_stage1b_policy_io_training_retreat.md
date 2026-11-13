# 설정/프로토콜 변경 기록 002 — Stage 1B (기본 정책 확보와 Retreat 명세 정합성)

기존 버전: v0.3 + AMENDMENT_001 (저장소 HEAD `1680856`)
새 버전: v0.3 + AMENDMENT_001 + 이 문서
변경 시점: 2026-11-13 (UTC). 파일럿 분기 수집·감시기 학습·dev/calibration/test 평가 이전.
변경 사유: Stage 0~2에서 기본 정책이 dev 0/20이었고, Retreat가 설정 5 cm 중 1.9 cm만 이동했다.
아래 네 가지 변경은 서로 독립이며 따로 기록한다. 허용 오차(`state_atol`, `action_atol`, `observation_atol`)는 바꾸지 않았다.

## 1. 관측·행동 경로 동등성 검사 추가 (검사 추가, 동작 변경 없음)

- 추가: `scripts/08_policy_io_audit.py`. 세 경로를 비교한다.
  - A 학습 데이터: 시연 HDF5 → upstream `SmallDemoDataset_DiffusionPolicy` + in-memory 패치(`convert_obs`)
  - B 학습 중 평가: upstream `make_eval_envs` + `StrictRGBObservationWrapper` + `FrameStack` + `CPUGymWrapper`
  - C 배포 adapter: `hci.simulation.make_env` + `DiffusionPolicy.convert` / history / `stacked`
- "upstream 경로"는 이 프로젝트의 관측 whitelist를 적용한 고정 commit의 trainer를 뜻한다. 순정 privileged-state agent는 입력 차원이 달라 비교 대상이 아니다.
- 검사 내용: (1) 원본 checkpoint `ema_agent` ↔ 내보낸 `weights.pt` tensor별 일치, (2) 같은 기록 프레임을 A/B/C에 넣었을 때 상태·RGB·이력·학습 target 행동 정렬, (3) 같은 seed·같은 행동으로 구동한 B/C 실제 환경의 관측, (4) 같은 EMA·같은 난수에서 upstream `Agent.get_action` ↔ adapter 행동 청크, (5) scheduler·horizon·controller·카메라 설정.
- 판정 기준: RGB는 축 정렬 후 완전 일치, 상태는 `state_atol`, 행동은 `action_atol`. 상태 복원 audit는 이 검사의 증거로 쓰지 않는다.
- 검사 스크립트 자체의 결함 3건을 실행 중 고쳤다 (정책·환경 코드는 수정 없음). 실패한 시도는 삭제하지 않았다.
  - `policy_io_audit_001` exit 1: `diffusion_policy` import 전에 패치를 설치함 → 학습 스크립트와 같은 순서(`load_upstream` → 패치 → trainer 모듈 재실행)로 수정.
  - `policy_io_audit_002` exit 2: (a) wrapper가 덮어쓴 `_init_raw_obs`로 raw 관측 key를 비교한 검사 오류 → `get_obs()` 사용. (b) encoder feature 차이 4e-5.
  - `policy_io_audit_003` exit 2: (b)의 원인 규명 중. 입력 값은 비트 단위로 같고, 차이는 GPU 메모리 배치에서만 나온다. upstream dataset은 궤적을 transpose된(NHWC stride) tensor로 보관하고 `get_action`도 permute view를 쓰는 반면, padding된 표본은 contiguous다. cuDNN이 배치별로 다른 합성곱 경로를 써서 하위 비트가 달라진다 (실측: 같은 입력에서 약 5e-5, cuDNN을 끄면 0.0). 그래서 feature 비교를 **같은 메모리 배치끼리** 하도록 고쳤고, 다른 배치와의 차이는 정보 항목으로 남겼다. 허용 오차는 그대로다.
  - `policy_io_audit_004`: PASS.

## 2. 학습·평가 설정 (개발 단계 설정 변경)

- 첫 본 학습: 성공 변환된 독립 시연 100개, optimizer update 100,000회, batch 256, lr 1e-4, obs/act/pred 2/8/16, 모델 크기 유지, 학습 seed 1, 새 초기화, DDPM 100 step.
  고정 upstream의 StackCube RGB baseline 설정(시연 100개, 100k iteration, 기본 batch 256)과 같지만, 관측 whitelist가 다르므로 공식 성능의 재현이라고 주장하지 않는다.
- `scripts/03_train_dp.py`: `--num-eval-episodes` 노출(기존 5 고정), `--full-checkpoint-freq`, `--data-manifest` 추가. upstream의 주기 저장(`save_freq`)은 넘기지 않고 3절의 전체 checkpoint로 대체.
- 시연: `scripts/07_validate_demos.py`가 구조·범위·성공 표시·원본 ID 대응·초기 배치 중복·action replay를 검사하고 manifest를 쓴다. 선택 규칙은 upstream loader와 같은 "traj id 오름차순 앞에서 N개".
- 정책 평가: `scripts/09_eval_policy.py` (프로젝트 Runtime = 배포 경로, 무개입). 주 지표 `success_once_absorbing`, horizon 200. `--success-at-end`는 보조 지표.
- 개발 전용 seed 집합 `configs/policy_eval_splits.json` (성능을 보기 전에 선언, 기존 split·시연 seed와 겹치지 않음을 코드로 검사):
  monitor 50000~50019, select 50100~50149, qualify_A 50200~50299, qualify_B 50300~50399.
- 선택 규칙(결과를 보기 전에 고정): iteration tag 60000 / 80000 / 100000의 EMA 세 후보를 같은 select 50개에서 평가, `success_once` 최댓값, 동률이면 더 뒤의 iteration (`scripts/11_select_candidate.py`).
  선택된 하나만 qualify 집합에서 한 번 평가한다. qualify 집합은 `artifacts/ledger/`에 사용 기록을 남기고 두 번째 실행을 거부한다.
- 학습 중 upstream 평가(무작위 seed 5회)와 monitor 20회는 관찰용이며 선택 근거가 아니다.
- 운영 기준(통계적 보장 아님): qualify 50/100 이상이면 다음 단계 진입, 10~49면 시연 500개 조건부 두 번째 학습, 0~9면 확대 없이 진단.

## 3. Checkpoint 저장 방식 (추가)

- `src/hci/training.py`: upstream 파일을 고치지 않고 `diffusion_policy.evaluate.evaluate`를 감싸, trainer가 평가를 호출하는 시점에 전체 학습 상태를 저장한다.
- 내용: raw model, EMA model, EMA 전체 상태(shadow, counter), optimizer, LR scheduler, Python/NumPy/Torch CPU·CUDA RNG, 학습 인자·config·데이터 manifest hash·upstream commit·프로젝트 소스 hash.
- 파일: `runs/policy/<run>/checkpoints/full_<iteration tag>.pt`, 10,000 iteration마다와 종료 시. 덮어쓰지 않는다.
- **tag와 update 수**: upstream 루프는 0부터 세고 update 후에 평가하므로 tag `i`(< total)는 update `i+1`회, 마지막 tag 100000은 100,000회다. 파일 안 `updates_completed`에 실제 값이 있다.
- **한계**: sampler의 epoch 내 위치는 저장되지 않는다 ("training-state resume, sample order non-identical"). resume loader는 구현·검증하지 않았다. 이 파일에서 다시 시작한 학습은 별도의 warm-start 실험으로 표기해야 한다.

## 4. Retreat 동작 (프로토콜 변경)

- 기존: 5 mm 이하 명령을 고정 12 step 실행. 한 번의 실측에서 1.9 cm 이동 (`results/stage1_2/intervention_check_001.json`). 기존 동작과 결과는 legacy 진단으로 보존한다.
- 변경: 명령 목표와 실제 변위를 분리한 bounded closed-loop.
  - 시작 TCP를 고정하고 target = 시작 + z 방향 0.05 m (`mode: up`)
  - `max_translation_per_step_m: 0.005` 유지. 이는 명령 증분의 상한이며 실제 이동량 보장이 아니다.
  - 매 step 실제 TCP와 target의 거리를 재서 명령을 갱신하고, 거리 ≤ `arrival_tolerance_m` (0.005 m)이면 종료, 최대 `max_steps` (40) step.
  - 예산 안에 도달하지 못하면 `reached: false`, `end_reason: budget_exhausted`로 기록하고 종료.
  - 그리퍼는 마지막 명령 유지, 실행한 모든 step은 원래 horizon 200에서 차감, 성공 시 기존 absorbing 규칙대로 종료.
  - eligibility는 최악의 경우(`max_steps` + `minimum_post_retreat_steps` = 48 step)를 기준으로 한다. 일찍 도달하면 남은 step을 쓰지 않는다.
- 설정 key: `retreat.steps` 삭제, `retreat.max_steps` / `retreat.arrival_tolerance_m` 추가 (세 config 모두). 옛 key가 있으면 오류를 낸다.
- 결과 기록: `Runtime.retreat_info` (시작/목표/최종 TCP, 명령 거리, 실제 변위, 사용 step, 포화 step 수, 종료 사유)를 rollout 결과의 `retreat` 항목으로 내보내고 snapshot/restore에 포함한다.
- 40 step과 5 mm는 개발용 운영 설정이다. 도달 여부는 안전성이나 무해함의 주장이 아니다.

## 5. 라이선스 (저장소 관리)

- 자체 코드에 Apache-2.0 적용: `LICENSE`(공식 전문, 수정 없음), `pyproject.toml`, README, `THIRD_PARTY_NOTICES.md`.
- 제3자 코드·시연·모델·자산은 저장소에 포함하지 않으며 재허가하지 않는다.

## 공통 기록

영향받는 파일: `src/hci/simulation.py`, `src/hci/training.py`(신규), `src/hci/statistics.py`(Wilson 구간 추가), `scripts/03_train_dp.py`,
`scripts/07~11_*.py`(신규), `scripts/stage1b/*.sh`(신규), `configs/{stackcube,pickcube,peg_insertion}.yaml`, `configs/policy_eval_splits.json`(신규).
회귀 테스트: `tests/test_runtime_mock.py`(Retreat 6건 추가, 고정 6 step 가정 1건 수정), `tests/test_training_checkpoint.py`, `tests/test_stage1b_tools.py`, `tests/test_demo_validation.py`, `tests/test_statistics.py`(Wilson).
독립 검토: Codex(gpt-6-sol, read-only)가 변경분을 읽고 StackCube 경로에서 오류를 찾지 못했다. 다른 과제에서 깨지는 하드코딩을 지적해 해당 스크립트는 StackCube-v1/`pd_ee_delta_pos` 외에는 거부하도록 했다.
변경 전에 이미 확인한 결과: Stage 0~2_001 (dev 0/20, 분기 96회). 연구 가설의 증거로 쓰지 않는다. 기준 성공이 0인 표본의 조건부 Harm은 정의되지 않는다(0%가 아님).
기존 보정의 유효성: 해당 없음 (보정 미수행).
기존 audit의 유효성: config hash가 바뀌었으므로 `audit_stack_001`은 새 설정·새 정책에 쓸 수 없다. 새 정책으로 audit를 다시 실행한다.
새로 생성하는 데이터: 시연 변환본(이전 HDF5는 해시만 남아 있어 동일하다고 가정하지 않고 새 데이터 버전으로 취급), 새 정책, 새 audit, 원본 2개 분기.
기존 결과 보존: `reports/STAGE_REPORT_stage0-2_001.md`, `AMENDMENT_001`, `results/stage0`, `results/stage1_2`, 로컬 `runs/*_001`, `artifacts/policies/stack_s1_d99` — 수정·삭제하지 않음.
승인 범위: 개발 단계. calibration과 최종 Test는 실행하지 않으며 이 문서는 그 승인이 아니다.

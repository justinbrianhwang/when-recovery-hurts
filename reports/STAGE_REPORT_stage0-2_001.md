# Stage 보고서

실행 ID: stage0-2_001
단계: Stage 0 (환경) ~ Stage 2 (정책 포함 audit, 원본 2개 분기 수집)의 소규모 실행
작업/정책 seed: StackCube-v1 (PickCube-v1은 smoke만) / 정책 학습 seed 1
실제 실행 환경: Vast.ai 대여 인스턴스 1대. RTX 3090 24GB (driver 565.57.01), 32코어 할당(EPYC 7502), RAM 125GB, 디스크 100GB, Ubuntu 22.04 컨테이너, Python 3.10.12, torch 2.6.0+cu124, sapien 3.0.3, gymnasium 0.29.1, diffusers 0.30.3. 시간당 $0.232, 사용 약 2시간 10분, 실제 차감 약 $0.65. 작업 종료 후 인스턴스 삭제 완료.
코드/upstream 버전: 이 저장소 v0.3 + `amendments/AMENDMENT_001` / ManiSkill v3.0.1 commit `a4a4f9272ad64b1564035874b605ceb687b63ed8`

CPU 테스트 통과, GPU 스모크 통과, 학습 완료, 과학적 가설 검증은 서로 다른 항목이다. 이 보고서에서 **가설 검증은 수행하지 않았다.**

## 실제 수행한 명령

시각은 UTC 2026-11-13. 로그 경로는 저장소 루트 기준이며 `logs/`, `runs/`, `artifacts/`는 Git에 올리지 않고 로컬에 보관한다.

| 명령 | 종료 코드 | 로그 경로 | 시작/종료 시각 |
|---|---:|---|---|
| `bash scripts/00_system_packages.sh` | 0 | `logs/00_system_packages_001.log` | 04:39 / 04:43 (01 포함) |
| `bash scripts/01_bootstrap_vast.sh` | 0 | `logs/01_bootstrap_001.log` | 위와 같음 |
| `python -m hci doctor --out runs/doctor_001` | 0 | `logs/doctor_001.log` | 04:44 |
| `bash scripts/06_validate_local.sh validation/server_core_001` | 0 (43 passed) | `logs/validate_core_001.log` | 04:44 |
| `python -m hci smoke --config configs/pickcube.yaml --out runs/smoke_pick_001` | **1** | `logs/smoke_pick_001.log` | 04:45 |
| `python -m hci smoke --config configs/stackcube.yaml --out runs/smoke_stack_001` | **1** | `logs/smoke_stack_001.log` | 04:45 |
| `bash scripts/06_validate_local.sh validation/server_core_002` (RNG 수정 후) | 0 (46 passed) | `logs/validate_core_002.log` | 04:45 |
| `python -m hci smoke ... --out runs/smoke_pick_002` | **2** | `logs/smoke_pick_002.log` | 04:45 |
| `python -m hci smoke ... --out runs/smoke_stack_002` | **2** | `logs/smoke_stack_002.log` | 04:45 |
| `bash scripts/02_prepare_demos.sh StackCube-v1 20` | 0 | `logs/prepare_demos_stack_001.log` | 04:47 / 04:48 (51초) |
| `python scripts/03_train_dp.py ... --run stack_plumbing_s1 --total-iters 500 --eval-freq 250` (dry run) | 0 | `logs/train_plumbing_dryrun_001.log` | 04:48 |
| 같은 명령 `--execute` | **1** | `logs/train_plumbing_s1_001.log` | 04:48 |
| `python scripts/03_train_dp.py ... --run stack_plumbing_s1_r2 --num-demos 20 --total-iters 500 --eval-freq 250 --execute` | 0 | `logs/train_plumbing_s1_r2.log` | 04:49 / 04:54 (315초) |
| `python scripts/03_train_dp.py ... --run stack_s1_d20 --num-demos 20 --total-iters 30000 --eval-freq 1000 --execute` | 0 | `logs/train_stack_s1_d20.log` | 04:55 / 06:09 |
| `python -m mani_skill.trajectory.replay_trajectory --traj-path .../motionplanning_d100/trajectory.h5 --use-first-env-state -c pd_ee_delta_pos -o rgb -b physx_cpu --num-envs 1 --count 100 --save-traj` | 0 (99/100 저장) | `logs/prepare_demos_stack_d100_001.log` | 04:55 / 04:58 (182초) |
| `python scripts/03_train_dp.py ... --run stack_s1_d100 --num-demos 100 ...` | **2** | `logs/train_stack_s1_d100.log` | 04:58 |
| `python scripts/03_train_dp.py ... --run stack_s1_d99 --num-demos 99 --total-iters 30000 --eval-freq 1000 --execute` | 0 | `logs/train_stack_s1_d99.log` | 04:59 / 06:14 |
| `bash scripts/06_validate_local.sh validation/server_core_003` (복원 방식 변경 후) | 0 (50 passed) | `logs/validate_core_003.log` | 05:05경 |
| `python -m hci smoke ... --out runs/smoke_pick_003` | 0 | `logs/smoke_pick_003.log` | 05:05경 |
| `python -m hci smoke ... --out runs/smoke_stack_003` | 0 | `logs/smoke_stack_003.log` | 05:05경 |
| `python scripts/04_export_policy.py --run runs/policy/stack_s1_d99 --checkpoint .../best_eval_success_once.pt --out artifacts/policies/stack_s1_d99` | 0 | `logs/export_stack_s1_d99.log` | 06:16 |
| `python -m hci audit --config configs/stackcube.yaml --policy artifacts/policies/stack_s1_d99 --out runs/audit_stack_001` | 0 | `logs/audit_stack_001.log` | 06:16 / 06:18 |
| `python -m hci collect ... --audit runs/audit_stack_001/audit.json --episodes 2 --out runs/branches_smoke_001` | 0 | `logs/branches_smoke_001.log` | 06:18 / 06:37 |
| `python scripts/inspect_branches.py --data runs/branches_smoke_001` | 0 | `logs/inspect_branches_001.log` | 06:37 |
| `python scripts/record_episode.py ... --seed 20000 --action continue --out runs/video_continue_001` | 0 | `logs/video_continue_001.log` | 06:37 / 06:38 |
| `... --action refresh --at 12 --out runs/video_refresh_001` | 0 | `logs/video_refresh_001.log` | 06:38 |
| `... --action retreat_retry --at 12 --out runs/video_retreat_001` | 0 | `logs/video_retreat_001.log` | 06:38 / 06:39 |
| 기본 정책 무개입 성공률, dev seed 20개 (`artifacts/diagnostics/base_success.py`) | 0 | `logs/base_success_dev20_001.log` | 06:39 / 06:45 |
| Refresh/Retreat 실측 (`artifacts/diagnostics/intervention_check.py`) | 0 | `logs/intervention_check_001.log` | 06:46 |

저장소에 없는 보조 스크립트(진단, 성공률 측정, 개입 실측, 그림용 렌더링)는 실행한 그대로 `artifacts/diagnostics/`에 보관했다.

## 산출물

| 파일 | 용도 | SHA256 |
|---|---|---|
| `artifacts/policies/stack_s1_d99/weights.pt` | 내보낸 EMA 기본 정책 | `04dfc563e5d6f2837104d543901fbe492252cbc7fc0b04748498d6aa4a50eba7` |
| `artifacts/policies/stack_s1_d99/policy_meta.json` | 정책 메타데이터 | `5d778e967d08b832c51d65c95fb0010e8966c9776cbd24334f2bb567e766bace` |
| `runs/policy/stack_s1_d99/checkpoints/best_eval_success_once.pt` | 원본 체크포인트 (24,000 iter 시점) | `4e57dbed4f1f906082fda1d7dc77adf8548af9590a9b4e68059708ff9872550b` |
| `runs/audit_stack_001/audit.json` | 정책 포함 복원 audit | `79119d724abdcf28f1c819daf94a23ad85d2934c37fed19e6e2c8d3f86489d94` |
| `runs/branches_smoke_001/dataset.json` | 분기 수집 manifest | `6af181356b6d78412e8ffbfa42cba97f0df7a05dd6bcf1b0b3156ad25cfc6a0a` |
| `runs/branches_smoke_001/episodes/10000.npz` | 분기 데이터 | `d4fc300b91c9a9c549870fe8e3f312b432b463a962aac172dd835d671d371848` |
| `runs/branches_smoke_001/episodes/10001.npz` | 분기 데이터 | `c0129678f8c4e7d88cacd3fd55bef280e775c3b869361480f488c2e3bfbe8365` |
| `runs/video_continue_001/episode.mp4` | Continue 영상 | `2bb7cb9bc728f55abf2fb7680599e31ed335f0a8a56cc2daeb337b0c08b61a54` |
| `runs/video_refresh_001/episode.mp4` | Refresh 영상 | `f12424e5a5a0f74619d30eec80fc02c89872fb302fcd1b12b13eac6a1c377e0e` |
| `runs/video_retreat_001/episode.mp4` | Retreat 영상 | `41e6c6276512c0ed75fa5c927541078a34e4de539aa920b296bd3451e2d6c91f` |
| `runs/base_success_stack_s1_d99_001/base_success.json` | 기본 성공률 측정 | `9d35cf21aa6c27a4f180dfcca3178b215cc7062e1f9661f591d2ba26320873e0` |
| `runs/intervention_check_001/intervention_check.json` | 개입 동작 실측 | `3ce6744bcb40297f3d330c46a84cca216af1b29dc8833f56e3793c0a3d6fc4ed` |
| `runs/smoke_pick_003/smoke.json`, `runs/smoke_stack_003/smoke.json` | RGB/복원 smoke | `687dbe47b607dff1f20052f5285c9291ee222d531cebf87e031794edc3b36a1d` (내용 동일) |
| 시연 RGB 20개 (`trajectory.rgb.pd_ee_delta_pos.physx_cpu.h5`) | 학습 데이터, 서버에만 있었음 | `ba52a55f43a6eb0dc1205980eea09beaebed133c65d51b9b771ca9cfa32bcbf9` |
| 시연 RGB 99개 (`motionplanning_d100/...h5`) | 학습 데이터, 서버에만 있었음 | `2756d4fcbd86241ea8a2c737473c2ec46204c44838e06d75a182ecec220c7b0c` |

내려받은 102개 파일은 서버 해시와 전부 일치했다(불일치 0). 주기 저장 체크포인트(실행당 30개, 합계 약 3GB)와 시연 HDF5는 내려받지 않았고 해시만 `artifacts/manifests/server_only_files_sha256_001.txt`에 남겼다.

## 검증

| 항목 | PASS / FAIL / NOT RUN | 실제 근거 |
|---|---|---|
| CPU tests | PASS | 서버 50 passed (`validation/server_core_003`). 원래 43개 + 회귀 테스트 7개 |
| CUDA | PASS | `doctor_001`: `cuda_available: true`, torch 2.6.0+cu124 |
| Vulkan/RGB | PASS | `vulkaninfo`가 RTX 3090 인식, smoke에서 128×128 RGB 2대(`base_camera`, `hand_camera`) 실제 렌더링. SAPIEN이 "Vulkan ICD file을 찾지 못함" 경고를 내지만 렌더링은 동작 |
| 상태 복원 (무작위 행동 smoke) | PASS (수정 후) | `smoke_*_001` exit 1(RNG 복사 오류), `smoke_*_002` exit 2(오차 2.7e-3 / 6.3e-3), `smoke_*_003` 상태·RGB·성공 차이 0.0 |
| 정책 rollout | PASS (동작) / 성능은 미달 | action chunk (8, 4), 입력 state (1, 2, 25), rgb (1, 2, 128, 128, 6), 카메라 순서 `base_camera`, `hand_camera`. dev 20 episode 성공 0 |
| 상태+정책 복원 | PASS | `audit_stack_001`: seed 100~104 모두 반복 재실행과 Retreat 분기 후 복원에서 상태·관측·행동 차이 0.0, 결과 일치 |
| 개입 행동 영상 | PASS (기록됨) | seed 20000에서 Continue / Refresh(t=12) / Retreat(t=12) 3개. 세 경우 모두 horizon 200에서 실패로 종료 |
| seed split | PASS | audit 100~104, train 10000~10001, dev 20000~20019만 사용. calibration/test 미사용 |
| 데이터 완료 상태 | PASS | `dataset.json` `completed: true`, 2 episode, 체크포인트 각 4개 |
| privileged 정보 | PASS (판독) | 정책·예측기 입력은 RGB, qpos, qvel, TCP pose뿐. 코드 판독과 Codex 독립 리뷰에서 누출 없음 |
| 공식 agent 입력과 자체 adapter의 동일 관측 비교 | NOT RUN | 미수행 |
| 최종 Test | NOT RUN | 실행하지 않음. 승인 토큰 생성·입력 없음 |

## 관측 결과

원본 episode 수: 분기 수집 2개 (train seed 10000, 10001). 별도로 audit 5개, dev 무개입 20개, 영상 3개.
분기 실행 수: 96 (2 episode × 4 checkpoint × 3 선택지 × 4 반복). 행 수 96.
성공/실패 수:
- 학습 중 평가(공식 trainer, 평가당 5 episode, 31회): 시연 20개 실행은 155회 중 성공 0. 시연 99개 실행은 155회 중 성공 4 (14,000 iter 1/5, 22,000 iter 1/5, 24,000 iter 2/5).
- 내보낸 정책의 무개입 성공: dev seed 20000~20019에서 **0/20**. 분기 수집의 원본 2개도 모두 실패.
- 분기 96회 중 성공 2회. 둘 다 seed 10001의 Refresh.

Rescue/Harm 분자와 분모:
- Refresh: Rescue 2 / 기준 실패 32, Harm 0 / 기준 성공 0
- Retreat–Retry: Rescue 0 / 기준 실패 32, Harm 0 / 기준 성공 0
- 기준 정책이 한 번도 성공하지 못해 **Harm의 분모가 0**이다. 이 데이터로는 악화율에 대해 아무 말도 할 수 없다. 행들은 같은 episode의 checkpoint·반복이라 독립 표본도 아니다.

개입 동작 실측 (dev seed 20000, t=12, `intervention_check_001`):
- Refresh: 남은 큐 4 → 0, 추론 1회 추가, 첫 행동 최대 차이 0.0073. 자연 재계획 경계가 아닌 chunk 중간에서 실행됨.
- Retreat–Retry: 12 step 소모, 그리퍼 명령 0.9990으로 12 step 동안 유지, 큐브 위치 변화 없음, TCP 자세(quaternion) 최대 변화 0.0029.
- **Retreat의 실제 상승량은 1.9 cm였다.** 설정은 5 cm(step당 5 mm). step당 실제 이동이 약 1.94 mm로 제어기가 목표를 따라가지 못한다. 첫 step은 오히려 1.8 mm 내려갔다.

시간/메모리/저장량:
- 학습: 약 19 it/s (batch 64), 평가 포함 1,000 iter당 약 2분 30초, 30,000 iter에 74~76분. VRAM 최대 2.4GB(단독 실행 시).
- 정책 rollout: 200 step episode 1개에 18.3초. 그중 정책 추론이 16.7초(25회 호출, 호출당 약 0.67초, DDPM 100 step).
- 분기 수집: 96회에 1,175초, 분기 1회당 12.2초 (prefix 재생 포함, 전부 horizon까지 실행된 경우).
- audit 5 seed: 115초.
- 저장량: 분기 데이터 episode당 npz 1개(48행 × 568차원), 체크포인트 1개 50MB.

## 변경과 실패

변경한 파일과 이유 (상세는 `amendments/AMENDMENT_001_restore_by_prefix_replay.md`):
1. `src/hci/simulation.py` — ManiSkill `BatchedRNG`가 `copy.deepcopy`에서 예외. RNG 상태값을 저장·복원하도록 변경. 회귀 테스트 `tests/test_rng_restore.py`.
2. `src/hci/observations.py` — `mani_skill.utils`가 `mani_skill.envs.utils`에 가려져 import 실패. `importlib.import_module`로 변경. 회귀 테스트 `tests/test_training_patches.py`.
3. `src/hci/simulation.py`, `src/hci/smoke.py` — `set_state_dict` 복원이 강체 접촉 상태를 바꿔 1 step 뒤 7e-5, 8 step 뒤 6e-3의 차이 발생. reset 후 기록 행동을 재생하는 방식(prefix action replay)으로 변경하고 재생 상태를 스냅샷과 `state_atol` 1e-6으로 매번 검증. **허용 오차는 바꾸지 않았다.** 회귀 테스트 `tests/test_replay_restore.py`.

실패한 명령 및 원인:
- `smoke_pick_001`, `smoke_stack_001` (exit 1): 위 1번.
- `smoke_pick_002`, `smoke_stack_002` (exit 2): 위 3번. 수정 전 기준 초과.
- `stack_plumbing_s1` 학습 (exit 1): 위 2번. 메타데이터 파일만 남은 실행 폴더를 보존.
- `stack_s1_d100` 학습 (exit 2): 시연 100개 변환 중 1개가 재생에 실패해 99개만 유효. 스크립트가 100개 요청을 거부. 99개로 `stack_s1_d99` 실행.
- 실패한 실행 폴더는 삭제하지 않았고 모두 새 run ID로 재실행했다.

발견했으나 고치지 않은 것:
- **기본 정책 성능 미달.** 30,000 iter, 시연 99개로 dev 0/20.
- **Retreat 실제 이동량이 설정의 약 38%.** 설정값(거리·step 수·step당 한계)은 개발 중 조정 대상이므로 임의로 바꾸지 않았다.
- 공식 trainer는 마지막 iteration의 가중치를 저장하지 않는다(마지막 주기 저장은 29,000). 또 `iteration % save_freq == 0`일 때만 저장하므로 QUICKSTART의 500 iter 예시(`--eval-freq 500`)는 미학습 `0.pt`만 남긴다.
- `configs/pickcube.yaml`의 `panda_wristcam`은 PickCube-v1의 지원 로봇 목록에 없다는 경고가 나온다. smoke는 통과했다.

수행하지 않은 일: 120개 파일럿 수집, 예측기 학습, dev/calibration 평가, 보정, 동결, 최종 Test, PegInsertionSide, 다른 정책 seed, 공식 agent와 adapter의 동일 관측 비교.

체크포인트 선정 기준: 학습 중 평가에서 `success_once`가 가장 높았던 체크포인트(`best_eval_success_once.pt`, 24,000 iter, 2/5). 평가 표본이 5개뿐이라 이 기준은 잡음이 크며, 실제로 dev 20개에서는 0/20이었다.

## 다음 단계

필요 작업:
1. **기본 정책 확보가 선결 과제다.** 기준 정책이 성공하지 못하면 Harm을 정의할 표본이 생기지 않는다. 학습량·시연 수를 늘리는 개발 실행이 필요하다.
2. Retreat 설정을 실측 이동량에 맞춰 조정할지 결정(개발 단계 변경, amendment 기록).
3. 공식 agent 입력과 자체 adapter 입력을 같은 관측에서 비교.
4. 이후 120개 파일럿 수집 → 예측기 학습 → dev 평가.

예상 계산량의 산출 근거 (이번 실측 단가, $0.232/시간 기준. 전부 horizon까지 실행된 경우의 상한):
- 파일럿 분기 수집: 120 × 4 × 3 × 4 = 5,760회 × 12.2초 ≈ 19.5시간 ≈ $4.5
- dev 전체 평가: 100 episode × (후보 19 + 기준 1 + 무개입 재실행 1 = 21회) × 18.3초 ≈ 10.7시간 ≈ $2.5
- calibration 평가: 400 episode × 21회 × 18.3초 ≈ 42.7시간 ≈ $9.9
- 기본 정책 학습: 평가 포함 1,000 iter당 약 2.5분. 예) 100,000 iter ≈ 4.2시간 ≈ $1.0
- 정책 추론이 rollout 시간의 약 91%를 차지한다. 위 수치는 정책이 일찍 성공하면 줄어든다.

동결/승인이 필요한 사항:
- 기본 정책 추가 학습의 규모(시연 수, iteration)
- 120개 파일럿 수집 착수 여부
- Retreat 설정 변경 여부
- 저장소 라이선스

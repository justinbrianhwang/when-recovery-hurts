# Stage 보고서 — Stage 1C (기본 정책 실패 원인 진단)

실행 ID: `stage1c_001`
단계: Stage 1C. 새 학습 없음. 보존된 60k EMA 정책과 시연으로 한 진단이며 qualification·calibration·최종 Test가 아니다.
작업/정책 seed: StackCube-v1, 정책 `artifacts/policies/stack_1b_s1_d100_i060000` (학습 seed 1, 재선택 없음)
실제 실행 환경: Vast.ai 인스턴스 60578138 (RTX 3090 24 GB, Ryzen 5 5600X, $0.201/h), 2026-11-14 02:01~02:47 UTC. 삭제 완료.
코드/upstream 버전: 저장소 HEAD `ce2f17a` + 이 보고서와 함께 올리는 신규 파일 6개(아래 "변경과 실패"). Stage 1B 보고서 기준 `700bb1e`와의 차이는 `ce2f17a`에서 추가된 `scripts/12_failure_trace.py`·결과 파일뿐이며 `src/hci`의 실행 경로는 같다. ManiSkill 고정 commit `a4a4f9272ad64b1564035874b605ceb687b63ed8`.
명세: `prompts/STAGE1C_POLICY_FAILURE_DIAGNOSIS_AGENT_PROMPT_v0_1.md`

## 결론

원인을 **확정하지는 못했지만 후보는 좁혀졌다.**

- 입출력 경로 불일치와 전문가 행동 재생 실패는 관측되지 않았다. RGB 차이는 저장 이력 위에서 행동을 거의 바꾸지 않았다(폐루프에서 누적되는 영향까지 배제한 것은 아니다).
- 정책은 **학습에 쓴 장면 10개 중 6개를 풀었고, 학습에 쓰지 않은 시연 장면 10개는 하나도 풀지 못했다.** 저장 시연에서의 행동 오차와 denoising loss도 held-out에서 크게 나빴다. 이는 명세 8절의 "train은 상대적으로 잘 풀고 held-out에서 크게 저하" 패턴에 해당한다.
- 학습 장면에서도 4개는 실패했다. 네 개 모두 큐브를 잡은 뒤에 멈췄다. 저장 시연 위에서의 행동은 전문가와 가까웠으므로, 폐루프 실행 중 벗어남도 두 번째 후보로 남는다.
- 표본은 장면 20개다. 위 차이는 후속 실험을 고르기 위한 근거이며 일반화 성능에 대한 통계적 결론이 아니다.

qualification 4/100은 그대로다. 이번 단계는 그 수치를 바꾸지 않는다.

## 실제 수행한 명령

전체 기록: `results/stage1c/commands_stage1c_001.status`. 시각은 2026-11-14 UTC.

| 명령 | 종료 코드 | 로그 경로 | 시작/종료 시각 |
|---|---:|---|---|
| (로컬, GPU 대여 전) `python scripts/13_stage1c_manifest.py … --out runs/stage1c_001/manifest` | 0 | 콘솔 | 대여 전 |
| (로컬) `python -m pytest -q` | 0 (69 passed, 4 skipped) | 콘솔 | 대여 전 |
| `bash scripts/00_system_packages.sh`, `bash scripts/01_bootstrap_vast.sh` | 0, 0 | `logs/s1c_install_sys.log`, `logs/s1c_bootstrap.log` | 02:03 / 02:05 |
| `python -m hci doctor --out runs/stage1c_001/doctor` | **2** | `logs/s1c_doctor.log` | 02:07:16 / 02:07:20 |
| 같은 명령 (`doctor_b`) | 0 | `logs/s1c_doctor_b.log` | 02:08:34 / 02:08:38 |
| `python -m pytest -q` (서버) | 0 (74 passed) | `logs/s1c_pytest_b.log` | 02:08:38 / 02:08:41 |
| `python scripts/14_scene_rollouts.py … --limit 1 --out runs/stage1c_001/probe_rollouts_b` (처리량 측정) | 0 | `logs/s1c_probe_rollouts_b.log` | 02:08:41 / 02:09:35 |
| `python scripts/15_offline_action_diag.py … --raw-checkpoint … --limit 1 --out runs/stage1c_001/probe_offline_b` (처리량 측정) | 0 | `logs/s1c_probe_offline_b.log` | 02:09:35 / 02:12:27 |
| `python scripts/08_policy_io_audit.py --config configs/stackcube.yaml --policy artifacts/policies/stack_1b_s1_d100_i060000 --checkpoint runs/policy/stack_1b_s1_d100/checkpoints/full_060000.pt --demo data_backup/demos_stack_002/trajectory.rgb.pd_ee_delta_pos.physx_cpu.h5 --out runs/stage1c_001/io_parity_60k` | 0 | `logs/s1c_io_parity_60k.log` | 02:13:03 / 02:14:39 |
| `python scripts/14_scene_rollouts.py --config … --manifest runs/stage1c_001/manifest/diagnostic_manifest.json --demo … --source data_backup/demos_stack_002/trajectory.h5 --policy … --out runs/stage1c_001/scene_rollouts` | 0 | `logs/s1c_scene_rollouts.log` | 02:14:39 / 02:23:26 |
| `python scripts/15_offline_action_diag.py --config … --manifest … --demo … --policy … --replay-obs runs/stage1c_001/scene_rollouts/replay_obs --out runs/stage1c_001/offline` | 0 | `logs/s1c_offline.log` | 02:23:26 / 02:45:36 |
| `bash scripts/stage1c/run.sh pack` | 0 | 콘솔 | 02:46 |

처리량 측정용 두 실행(`probe_*`)의 출력은 결과로 쓰지 않았다.

## 산출물

로컬 위치: `runs/stage1c_001/` (Git 제외, 영상·step별 기록 포함). 공유용 요약: `results/stage1c/` (Git 포함).
서버에서 만든 묶음 `stage1c_out.tgz` SHA256 `a9626e6a2eb8513b99429b7a1e4eb2ab0ab197fdb374be99fdc2220510b38b30` — 내려받은 파일과 일치. 풀어낸 261개 파일도 서버의 목록(`results/stage1c/backup_sha256_stage1c_001.txt`)과 261/261 일치.

| 파일 | 용도 | SHA256 |
|---|---|---|
| `results/stage1c/diagnostic_manifest.json` | 표본·hash·난수·실행 범위 (결과 이전에 고정) | `a820d606f8415908625d06a7d39ffea955d72741edebcdc48e68be514cd0de76` |
| `results/stage1c/io_parity_60k.json` (+ `_preprocessing.csv`, `_actions.csv`, `_artifact_tensors.json`) | D1 새 정책의 A/B/C 비교 | `37881d988f775caa49bf33a2b945ce2608dc9a63234dd56af4bc20121836bac8` |
| `results/stage1c/scene_rollouts.csv`, `scene_rollouts_summary.json` | D2 장면별 rollout 100개 | `ca37ebda66e51fbb2d8a714de6d6fc3722d71aa17ce0a977b186d0b62ba73376` |
| `results/stage1c/replay_obs_diff.csv` | 전문가 재생의 step별 관측·상태 차이 | 백업 목록 참조 |
| `results/stage1c/offline_action_metrics.csv`, `offline_history_metrics.csv`, `denoise_loss.csv`, `offline_summary.json` | D3 | `cc87aaff7b6b3fc86ca72b49f98d5ef2e3cf50a2293f200b1c9300b6d7d0f203` |
| `results/stage1c/rgb_action_sensitivity.csv` | D4 | `41b6c5a6fdb9b056b4b2d9b4d731d406cdcbaadeedb63a38c6f70db6c7cd00a0` |
| `results/stage1c/images/` | RGB 차이 예시와 위치별 평균 차이 | 백업 목록 참조 |
| `runs/stage1c_001/scene_rollouts/videos/` (8개) | 대표 영상 | 백업 목록 참조 |

보존 자료 확인(대여 전, 로컬): 변환 시연 HDF5·JSON, 원본 HDF5·JSON, `full_060000.pt`, 60k `weights.pt`·`policy_meta.json`의 SHA256이 Stage 1B 백업 목록·시연 manifest와 모두 일치. 서버에 올린 뒤에도 같은 값을 확인했다.

## 검증

| 항목 | PASS / FAIL / NOT RUN | 실제 근거 |
|---|---|---|
| CPU tests | PASS | 로컬 69 passed·4 skipped, 서버 74 passed |
| CUDA | PASS (1회 실패 후) | 첫 `doctor` exit 2: 컨테이너 이미지의 CUDA forward-compat 라이브러리(550)와 호스트 드라이버(535) 충돌. 컨테이너 안 설정 파일 하나를 치워 해결. 호스트 드라이버는 건드리지 않음 |
| Vulkan/RGB | PASS | rollout·영상 생성 |
| D1 새 정책 입출력 parity | PASS | 아래 |
| D2 전문가 재생 | PASS | 60/60 성공, 실패 0 |
| D2 정책 rollout | 실행 완료 | 40/40 실행, 제외 없음 |
| D3 오프라인 행동 | 실행 완료 | 100 이력 × 난수 4개 |
| D4 RGB 민감도 | 실행 완료 | 100 이력 × 조건 6개 × 난수 4개 |
| D5 보조 진단 | NOT RUN | D1~D4에서 후보가 좁혀져 생략 (명세 7절) |
| 상태+정책 복원, 개입 영상 | NOT RUN | 이번 범위 밖 |
| seed split | 변경 없음 | `policy_qualify_B`, calibration, Test seed 미사용. held-out 시연 20개는 이번부터 개발 진단에 사용된 것으로 기록 |
| 데이터 완료 상태 | PASS | 선언한 표본 전부 실행 |

## 관측 결과

### 준비 단계에서 확인한 것 (GPU 이전)

- 시연 구조: 상태 qpos 9 + qvel 9 + TCP 7, 카메라 `base_camera`·`hand_camera` 각 128×128×3 uint8 (값 범위 0~255), 행동 4차원 float32. 행동 범위 x −0.27~0.23, y −0.56~0.58, z −0.32~0.52, 그리퍼 −1 / +1. 원본은 `pd_joint_pos` 8차원이고 step 수는 변환 전후가 같다.
- **첫 성공 step**: 120개 시연에서 최소 81, 평균 103.7, 최대 201. 성공 뒤 tail은 3~5 step이다. **첫 성공이 horizon 200을 넘는 시연은 `traj_30` 하나**(길이 205, 첫 성공 201)이며 학습 100개에 들어 있다. 나머지 119개는 성공 뒤 짧은 tail일 뿐이다. horizon은 바꾸지 않았다.
- **변환 파일의 첫 env state 문제(upstream 동작)**: 고정 commit의 `replay_trajectory`는 변환 파일의 첫 env state를 원본의 *두 번째* state로 덮어쓴다. 첫 관측은 원본의 첫 state에서 렌더링된다. 그래서 변환 파일의 `env_states[0]`은 첫 관측과 맞지 않는다(선택 장면에서 최대 8.4e-3). 정책 입력에는 env state가 들어가지 않으므로 학습에는 영향이 없고, 진단에서는 원본의 첫 state를 기준으로 삼았다.
- 표본: train `traj` 0, 11, 22, 33, 44, 55, 66, 77, 88, 99 / held-out 100, 102, 104, 106, 108, 111, 113, 115, 117, 119 (오름차순 목록에서 균등 간격). 이력 100개. 단계 표시는 시연 자체의 그리퍼 명령으로 정했다: 접근 50, 운반 41, 경계 9, **배치 0** (창이 80% 지점에서 끝나 배치 단계는 표본에 없다).

### D1 — 새 정책(60k EMA)의 입출력 parity: PASS

| 비교 | 표본 | 결과 |
|---|---|---|
| checkpoint EMA ↔ 내보낸 가중치 | 160 tensor | 모두 비트 단위 일치, hash 일치 |
| 같은 기록 프레임을 A(학습 dataset)·B(upstream 평가 wrapper)·C(runtime adapter)에 입력 | 6 시연, 42 이력 | 상태 A↔C, B↔C 최대 0.0 / RGB 완전 일치 / encoder feature(같은 메모리 배치) 0.0 / 학습 target 행동 정렬 0.0 |
| 같은 seed·같은 행동으로 구동한 B·C 실제 환경 | 42 이력 | 상태 0.0, RGB 완전 일치 |
| **실제 inference 출력**: 같은 EMA·같은 난수에서 upstream `Agent.get_action` ↔ adapter | 42 청크 | 최대 차이 0.0 (B 관측, C 관측 양쪽). clip이 출력을 바꾼 경우 0 |
| scheduler·horizon·controller·카메라 설정 | — | 일치 |

- 저장 tensor의 일치와 실제 추론 출력의 일치를 따로 확인했다. 비교 호출마다 입력을 복사했고, 난수는 호출 직전 `torch.manual_seed`로 CPU·CUDA 생성기를 모두 맞춰 초기 noise와 역확산 중 noise가 같다.
- 정보 항목: 다른 메모리 배치에서의 encoder feature 차이 4.8e-5 (Stage 1B에서 규명한 cuDNN 경로 차이). 허용 오차는 바꾸지 않았다.
- 표본은 Stage 1B의 004 검사와 같은 규칙(앞 6개 시연, 42 이력)이다. PASS는 이 표본에서 세 경로가 같다는 뜻이며 환경 의미의 정확성 전체를 증명하지 않는다.

### D2 — 장면별 rollout (100개, 제외 없음)

초기화 F = `reset(**reset_kwargs)` 뒤 원본 시연의 첫 env state를 한 번 써넣음(변환 데이터를 만든 절차). R = runtime의 `reset(seed)`. rollout 중에는 상태를 써넣지 않았고 정책에 시연 관측을 넣지 않았다.

| 셀 | train 성공/시도 | held-out 성공/시도 |
|---|---:|---:|
| F-expert (기록 행동 재생) | 10/10 | 10/10 |
| **F-policy** (60k EMA 폐루프) | **6/10** | **0/10** |
| R-expert | 10/10 | 10/10 |
| **R-policy** | **5/10** | **0/10** |
| Fdemo-expert (시연 JSON 설정 환경, 관측 재현 확인용) | 10/10 | 10/10 |

- **전문가 재생**: 60개 모두 성공했고 첫 성공 step이 시연 기록과 60개 모두 같다. replay failure 0. 다만 성공이 궤적의 완전한 재현을 뜻하지는 않는다. 일부 재생은 중간에 파일과 물리적으로 갈라졌다(예: `traj_117`, D4 참조. R 초기화에서는 큐브 상태 차이가 최대 3.5까지).
- **F와 R의 초기 장면**: 원본 첫 state와의 차이가 F 최대 1.5e-8, R 최대 1.2e-7. 두 초기화는 같은 장면을 만든다. 그러므로 F↔R 비교는 초기화 경로 차이를 거의 담지 않는다.
- **F-policy와 R-policy**: 40쌍 중 39쌍의 성공 여부가 같다. `traj_55`만 F 성공(109 step) / R 실패(정렬까지 갔으나 성공 판정 없음)였다. 초기 상태가 1e-7 수준만 달라도 결과가 뒤집힌 사례다.
- 정책이 도달한 가장 먼 단계 (F-policy / R-policy):

| 집합 | 성공 | 정렬했으나 성공 없음 | 들어올렸으나 정렬 못 함 | 파지했으나 못 들어올림 | 큐브 3 cm 이내 미도달 |
|---|---:|---:|---:|---:|---:|
| train | 6 / 5 | 2 / 3 | 1 / 1 | 1 / 1 | 0 / 0 |
| held-out | 0 / 0 | 0 / 0 | 2 / 1 | 2 / 3 | 6 / 6 |

- **train 장면**: 10개 모두 큐브 A에 도달해 파지했다(첫 파지 step 37~56). 실패 4개는 파지 이후에 멈췄다.
- **held-out 장면**: 6개는 TCP가 큐브 A 중심 3 cm 안에 한 번도 들어가지 못했고(최소 거리 4.9~22.9 cm), 그리퍼는 닫았다(첫 닫힘 명령 step 47~72). 나머지 4개는 파지 판정까지 갔으나 적층하지 못했다.
- 성공한 train 장면의 성공 step은 시연과 비슷하다(예: `traj_0` 정책 102, 시연 102).
- 대표 영상 8개(결과 이전에 고정한 규칙: F-policy의 train 앞 3개·held-out 앞 3개, R-policy 각 첫 1개): `runs/stage1c_001/scene_rollouts/videos/`.

이 rollout은 정책이 매 step 스스로 행동을 낸 폐루프 실행이다. 상태를 시연으로 되돌리는 teacher-forcing이 아니다.

### D3 — 저장 시연 이력에서의 행동 생성 (100 이력 × 난수 4개 = 400회)

단위: 정규화된 제어 입력 [−1, 1]. 팔 성분 1.0은 제어기의 위치 증분 한계 0.1 m에 해당하는 **명령**이며 실제 TCP 이동량이 아니다. 실행 구간 8 step은 모두 기록된 행동이다(padding 없음). 예측 16 step 중 padding이 낀 이력은 집합별 10개(시작 창)이며 loss는 padding 제외 값도 함께 계산했다(차이 미미).

| 지표 (난수 4개 평균) | train | held-out |
|---|---:|---:|
| 실행 8 step xyz MAE | 0.0033 | 0.0105 |
| 실행 8 step xyz RMSE | 0.0041 | 0.0152 |
| 성분별 MAE x / y / z | 0.0027 / 0.0035 / 0.0036 | 0.0077 / 0.0125 / 0.0111 |
| 부호 있는 평균 오차 x / y / z | −0.0015 / −0.0005 / −0.0012 | +0.0005 / +0.0031 / +0.0013 |
| 첫 행동 xyz L2 오차 | 0.0036 | 0.0068 |
| 전문가 행동 크기(xyz norm 평균) | 0.135 | 0.128 |
| 포화(|a| ≥ 0.999) 비율, 예측 / 전문가 | 0 / 0 | 0 / 0 |
| 그리퍼 연속값 MAE | 0.0004 | 0.0031 |
| 그리퍼 열기/닫기 방향 일치 | 100% | 99.9% (실행 step 1,600개 중 2개 불일치, 서로 다른 2행) |
| 난수 간 변동: 난수 쌍 사이 xyz 평균 절대차 | 0.0017 | 0.0020 |
| 난수별 MAE 평균 (4개) | 0.0033, 0.0032, 0.0033, 0.0033 | 0.0108, 0.0104, 0.0102, 0.0104 |

단계별 xyz MAE (train / held-out): 접근 0.0047 / 0.0130, 운반 0.0021 / 0.0088, 경계 0.0012 / 0.0018. 배치 단계는 표본이 없다.

- 같은 입력·같은 난수 재실행의 차이는 0.0이다.
- train 이력에서는 생성 행동이 전문가 행동과 가깝다(오차가 행동 크기의 약 2~3%). held-out에서는 오차가 약 3배이고 접근 단계에서 가장 크다.
- best-of-4는 계산하지 않았다. 난수별 평균이 서로 거의 같아 난수 선택이 결론을 바꾸지 않는다.
- Diffusion Policy는 기록된 전문가 궤적과 다른 유효한 행동을 낼 수 있으므로 이 오차만으로 행동이 틀렸다고 하지 않는다. D2의 실제 실행 결과와 방향이 같다는 점만 적는다.

**Denoising loss** (noise prediction MSE, 평가 모드, 고정 noise): Stage 1B의 마지막 표시값 0.000184는 이 양이며 행동 오차나 성공률이 아니다.

| | train | held-out |
|---|---:|---:|
| 전체 100 timestep 평균 (이력 50개 평균) | 0.00038 | 0.0174 |
| 같은 값의 중앙값 / 최대 | 0.00029 / 0.00097 | 0.0056 / 0.126 |
| timestep 0 / 10 / 25 | 0.0158 / 0.00090 / 0.00025 | 0.398 / 0.0243 / 0.0120 |
| timestep 50 / 75 / 99 | 0.000086 / 0.000053 / 0.000054 | 0.0036 / 0.00069 / 0.000052 |

- train 이력의 loss(0.00038)는 학습 로그의 값과 같은 자릿수다. held-out 이력의 평균은 그 약 46배다. noise가 작은 timestep일수록 차이가 크다.

### D4 — RGB 차이가 행동으로 이어지는가

같은 이력에 대해 S(파일 고유감각 + 파일 RGB), V(파일 고유감각 + 재생 RGB), L(재생 고유감각 + 재생 RGB)을 같은 난수 4개로 비교했다. RGB 단위는 uint8 [0, 255]이고 encoder가 255로 한 번 나눈다.

**RGB 차이의 크기와 출처**

| 재생 환경 | base 카메라 MAE | hand 카메라 MAE | 1 이상 다른 화소 | 8 이상 | 32 이상 |
|---|---:|---:|---:|---:|---:|
| 프로젝트 runtime 환경, F 초기화 | 0.484 | 0.490 | 약 48% | 약 0.001~0.03% | 약 0.001~0.02% |
| 프로젝트 runtime 환경, R 초기화 | 0.484 | 0.487 | 약 48% | 비슷 | 비슷 |
| **시연 JSON 설정으로 만든 환경**, F 초기화 | 0.0005 | 0.023 | 0.004% / 0.6% | 비슷 | 비슷 |

- Stage 1B가 보고한 "화소의 49%가 다르다"는 **거의 전부 ±1 수준의 차이**다. 8 이상 다른 화소는 평균 0.03% 미만이다.
- 시연 JSON의 환경 설정으로 환경을 만들면 파일의 RGB가 사실상 그대로 재현된다(첫 프레임은 완전 일치). 그러므로 ±1 차이는 **물리 상태가 아니라 프로젝트 runtime 환경과 시연 변환 환경의 생성 설정 차이**에서 나온다. 어느 설정 항목이 원인인지는 확인하지 않았다.
- 큰 차이(최대 232)는 드물게 나타나며, 재생이 파일과 물리적으로 갈라진 구간과 겹친다. 예: held-out `traj_117`은 시연 JSON 환경에서도 step 46부터 파일과 달라진다(고유감각 최대 0.55). 이 시연의 재생은 그래도 성공했다. 이 구간의 이력은 "관측 domain 차이"가 아니라 물리 상태 차이다.
- 위치별 평균 차이와 예시: `results/stage1c/images/`. 객체 주변 ROI 분석은 하지 않았다(차이가 화면 전체에 고르게 ±1이어서 필요하지 않다고 판단).

**행동에 미치는 영향** (EMA, 400행씩)

| 조건 (S 대비) | 실행 8 step xyz 평균 절대차 | 같은 입력의 난수 간 변동 대비 | 난수 간 변동을 넘은 행 | 그리퍼 방향 바뀜 | visual feature cosine |
|---|---:|---:|---:|---:|---:|
| V, runtime F 재생 | 0.00011 | 약 6% | 4/400 | 0 | 0.9998~1.0000 |
| V, runtime R 재생 | 0.00010 | 약 5% | 1/400 | 0 | 0.9999~1.0000 |
| V, 시연 설정 환경 재생 | 0.00003 | 약 1.5% | 4/400 | 0 | 0.9998~1.0000 |
| L, runtime F 재생 | 0.00012 | 약 6.5% | 8/400 | 0 | — |
| L, runtime R 재생 | 0.00010 | 약 5% | 1/400 | 0 | — |

- S↔V의 행동 차이는 같은 입력에서 난수만 바꿨을 때 생기는 변동의 5~6%다. 전문가 대비 오차도 S와 V에서 같다(train 0.00329 / 0.00329, held-out 0.01045 / 0.01047).
- 난수 간 변동을 넘은 소수의 행은 물리적으로 갈라진 구간(위 `traj_117` 등, 고유감각 차이 1e-4 초과 8행)과 겹친다.
- S↔V는 합성한 입력 대조다. 실제 환경 성공률에 대한 인과효과가 아니다. 다만 D2에서 F-policy와 R-policy가 같은 장면에서 거의 같은 결과를 냈고(39/40), 두 실행 모두 runtime 환경의 RGB를 썼는데 train 장면 절반 이상을 풀었다는 점과 방향이 같다.

### 수량·시간·비용

원본 episode 수: rollout 100 (정책 40, 전문가 60) + 처리량 측정 10
분기 실행 수: 0
성공/실패 수: 위 표
Rescue/Harm 분자와 분모: 측정하지 않음 (범위 밖)
시간/메모리/저장량: D1 1분 36초, D2 8분 47초(정책 rollout 6~12초, 전문가 1.5초), D3+D4 22분 10초(이력당 약 13초). 새 산출물 49 MB.

비용: 대여 전 잔액 $10.2388 → 삭제 후 $10.0387, **사용 약 $0.20** (목표 $0.50, 상한 $1.00). 인스턴스 가동 약 46분. 업로드 약 660 MB, 내려받기 49 MB 포함. 삭제 후 `show instances`가 비어 있음을 확인했다. 대기 시간은 없었다(알림을 기다리지 않고 20~45초 간격으로 직접 확인).

## 변경과 실패

변경한 파일과 이유 (모두 신규, 기존 파일 수정 없음):
- `src/hci/diagnostics.py` — 표본 선택, 단계 이벤트, RGB·행동 통계. Stage 1B 추적 스크립트의 분류 순서 문제(파지·들어올림 기록이 있는데 거리 임계값 때문에 "접근 실패"로 표시)를 근거 순위 방식으로 고쳐 새로 썼다. 기존 `12_failure_trace.py`와 그 결과는 그대로 두었다.
- `scripts/13_stage1c_manifest.py`, `scripts/14_scene_rollouts.py`, `scripts/15_offline_action_diag.py`, `scripts/stage1c/run.sh`, `tests/test_diagnostics.py`.
- 정책·환경·전처리·Retreat·config·split·ledger는 바꾸지 않았다. 객체 정답 pose는 기록과 비교에만 썼다.

Codex(gpt-6-sol, read-only)와의 분담: 대여 전에 신규 코드 전체를 검토했다. 실행 오류는 찾지 못했고 세 가지를 지적했다. (1) 묶음에서 manifest 제외 → 포함하도록 수정, (2) 체크섬이 묶음 자체를 검증하지 않음 → 묶음 hash를 내려받은 뒤 대조하고 풀어낸 파일도 다시 대조, (3) 시연 설정 환경의 kwargs 기록 → 요약에 기록. 보고서 작성 뒤 수치 대조도 맡겼다(아래).

실패한 명령 및 원인:
- `s1c_doctor` exit 2 — 위 CUDA 호환 라이브러리 충돌. 컨테이너 설정으로 해결, 재실행 통과.
- 그 외 실패 없음.

수행하지 않은 일:
- D5 보조 진단(RGB 교체, raw↔EMA 비교). 처리량 측정 때 raw 가중치가 10개 이력에서 함께 계산됐으나 결과로 쓰지 않는다.
- 객체 주변 ROI 분석, 배치 단계 이력(표본 규칙상 없음).
- runtime 환경과 시연 변환 환경 사이 ±1 RGB 차이를 만드는 설정 항목의 특정.
- 새 학습, warm start, overfit 학습, 80k/100k 재평가, `policy_qualify_B`, calibration, 최종 Test.

## 다섯 질문에 대한 답

1. **새 정책의 학습/배포 입력·출력 경로가 일치했는가?**
   검사한 42 이력·42 청크에서 일치했다. 상태·RGB·encoder feature·행동 청크의 최대 차이가 0.0이다. 표본 밖까지 보장하는 것은 아니다.

2. **전문가 행동은 실행되고, 학습한 정책은 학습 장면에서조차 실패하는가?**
   전문가 행동은 세 환경 조건에서 60/60 성공했고 성공 step도 기록과 같다. 정책은 학습 장면 10개 중 6개(F) / 5개(R)를 풀었다. 학습 장면에서조차 전부 실패하는 것은 아니며, 실패 4개는 모두 파지 이후였다.

3. **train과 held-out에서 어떤 차이가 있었는가?**
   폐루프 성공 6/10 대 0/10. held-out 6개는 큐브에 도달하지 못했다. 저장 시연 위에서도 held-out의 행동 오차가 약 3배(0.0105 대 0.0033), denoising loss가 약 46배(0.0174 대 0.00038)였다. 장면 10개씩의 작은 표본이고 한 시연의 이력들은 서로 독립이 아니므로, 일반화 성능의 확정적 결론이 아니라 방향을 보여주는 관측이다.

4. **RGB 차이가 행동 차이로 연결되는가? scene/물리 상태 confound는 통제됐는가?**
   저장 이력 위에서는 연결되지 않았다. 차이는 거의 전부 ±1 수준이고, 같은 고유감각·같은 난수에서 RGB만 바꿨을 때의 행동 변화는 난수 간 변동의 5~6%였다. 물리 상태가 파일과 갈라진 구간(8행)은 고유감각 차이로 식별해 따로 적었고, 시연 설정 환경에서는 RGB가 재현돼 ±1 차이가 환경 생성 설정에서 온다는 것도 분리했다. 합성 입력 대조이고 전문가 재생 이력에서의 즉각적인 행동 변화만 본 것이므로, 정책 폐루프 실행 중 누적되는 영향은 이 검사로 배제되지 않는다.

5. **다음에 바꿀 한 가지와 그 근거는?**
   **제안: 학습 시연 수를 100개에서 500개로 늘린다(다른 설정은 그대로).** 근거: 입출력 경로와 전문가 재생에 문제가 보이지 않는 상태에서, 본 장면은 절반 이상 풀고 보지 않은 장면은 하나도 못 풀며, 보지 않은 장면에서는 저장 관측 위의 행동부터 나쁘다. 특히 held-out의 **첫 창(t=0, 정책이 아직 한 번도 행동하지 않은 시점)** 에서 이미 xyz MAE가 0.0176으로 train의 0.0088보다 크다. 이 시점에는 누적 오차가 없으므로 held-out 실패를 폐루프 누적만으로 설명하기 어렵다. 이는 명세 8절 표에서 "500개 확대 또는 시각 일반화 변경" 행에 해당하고, 그중 하나만 고른다면 Stage 1B에서 이미 조건부로 선언해 둔 500개 학습이 가장 작은 변경이다.
   남는 두 번째 후보는 폐루프 벗어남이다(학습 장면 4/10 실패, 1e-7 초기 차이로 결과가 뒤집힌 사례). 시연 수를 늘린 뒤에도 학습 장면 성공이 낮게 남으면 그때 접두 재생 진단이나 실행 horizon 대조를 별도 실험으로 제안한다. 두 가지를 동시에 바꾸지 않는다.
   **Codex의 다른 의견**: 이 제안은 근거가 있는 후보이지만 "유일한 다음 변경"이라고 하기에는 강하다. 장면 10개씩의 표본은 패턴을 보여줄 뿐 원인을 확정하지 않으며, 폐루프 누적·분포 이동을 가르는 더 싼 실험(접두 재생 또는 실행 horizon 대조, 추론만 필요, 이번과 비슷한 약 $0.2)을 500개 학습(약 $0.6~0.9)보다 먼저 할 수 있다. 어느 쪽을 먼저 할지는 사용자 결정으로 남긴다.

## 다음 단계

필요 작업: 시연 500개 학습 실행 여부 결정. 실행한다면 Stage 1B의 조건부 두 번째 학습 규칙(같은 설정, 후보 60k/80k/100k, select 50회, `policy_qualify_B` 100회 1번)을 그대로 쓰고, 이번 진단 표본(train 10 + held-out 10 장면 rollout, 오프라인 100 이력)을 같은 방법으로 다시 재서 비교한다. 첫 100개가 500개의 부분집합이 되는지 먼저 확인한다.

예상 계산량의 산출 근거: Stage 1B 실측 19.8 it/s (batch 256, RTX 3090)로 100,000 update 약 1시간 41분. upstream dataset은 시연 RGB를 uint8로 GPU에 올린다. 100개(10,943 프레임 × 98,304 byte)가 약 1.1 GB이므로 500개는 약 5.4 GB로 예상한다. Stage 1B의 학습 최대 VRAM이 7.6 GB였으니 합계 약 12 GB로 24 GB 안일 것으로 보지만, 실행 전에 짧은 측정으로 확인해야 한다. 시연 500개 변환은 Stage 1B에서 130개에 약 1분 30초였으므로 약 6분. 평가는 select 150회 + qualify 100회 약 25분, 진단 재측정 약 35분. 합계 약 3시간, $0.2~0.3/h 기준 약 $0.6~0.9. 견적이며 보장이 아니다.

동결/승인이 필요한 사항: 시연 500개 학습과 `policy_qualify_B` 사용 승인, 예산 상한. `traj_30`처럼 첫 성공이 horizon을 넘는 시연을 학습에 그대로 둘지는 이번에 바꾸지 않았고, 바꾼다면 별도 amendment가 필요하다.

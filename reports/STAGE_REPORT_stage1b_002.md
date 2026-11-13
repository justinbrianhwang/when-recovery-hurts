# Stage 보고서 — Stage 1B (기본 정책 확보와 Retreat 명세 정합성)

실행 ID: `stage1b_002` (학습 run `stack_1b_s1_d100`)
단계: Stage 1B. 개발 단계이며 calibration과 최종 Test는 실행하지 않았다.
작업/정책 seed: StackCube-v1, 학습 seed 1
실제 실행 환경: Vast.ai 인스턴스 60500936 (RTX 3090 24 GB, Ryzen 9 9900X, $0.2862/h), 2026-11-13 13:12~23:42 UTC. 삭제 완료.
코드/upstream 버전: 저장소 `700bb1e` + 이 보고서와 함께 올리는 `scripts/12_failure_trace.py`. ManiSkill 고정 commit `a4a4f9272ad64b1564035874b605ceb687b63ed8`.
명세: `prompts/STAGE1B_BASE_POLICY_AND_RETREAT_AGENT_PROMPT_v0_1.md`, 변경 기록 `amendments/AMENDMENT_002_stage1b_policy_io_training_retreat.md`

## 결론

**기본 정책 확보 실패.** 선택된 정책의 qualification 결과는 **4/100** (Wilson 95% 구간 1.6%~9.8%)으로 운영 기준 50/100에 미달했고, 0~9 구간이므로 사전 규칙에 따라 두 번째 학습(시연 500개)을 시작하지 않았다.
다음 단계(파일럿 분기 수집)로 진입하지 않는다. 다음 작업은 **Stage 1C — 기본 정책 실패 원인 진단**이다.

Retreat 변경(폐루프, 5 cm 목표)은 시연 prefix 30개 상태에서 30/30 도달했다. 이것은 도달 여부의 측정이며 안전성·무해함의 주장이 아니다.

Rescue/Harm은 이번 단계에서 측정하지 않았다. 기준 성공이 거의 없는 정책에서는 조건부 Harm의 분모가 없거나 매우 작아 추정할 수 없다(0%가 아님).

## 실제 수행한 명령

모든 명령·종료 코드·시각은 `logs/stage1b_002.status`에 있다(백업 포함). 주요 항목만 적는다. 시각은 UTC.

| 명령 | 종료 코드 | 로그 경로 | 시작/종료 시각 |
|---|---:|---|---|
| `python -m pytest -q` (서버) | 0 (68 passed) | `logs/pytest_002.log` | ~13:27 |
| `bash scripts/02_prepare_demos.sh StackCube-v1 130` | 0 | `logs/prepare_demos_stack_002.log` | ~13:27 / 13:28 |
| `python scripts/07_validate_demos.py …` | 0 | `logs/validate_demos_002.log` | ~13:29 |
| `python scripts/10_retreat_diagnostic.py …` | 0 | `logs/retreat_diag_001.log` | ~13:29 |
| `python scripts/08_policy_io_audit.py …` (001) | 1 | `logs/policy_io_audit_001.log` | ~13:29 |
| 같은 명령 (002) | 2 | `logs/policy_io_audit_002.log` | ~13:32 |
| 같은 명령 (003) | 2 | `logs/policy_io_audit_003.log` | ~13:34 |
| 같은 명령 (004) | 0 | `logs/policy_io_audit_004.log` | ~13:34 / 13:36 |
| `python scripts/03_train_dp.py … --run stack_1b_probe_b256 --total-iters 1000 --batch-size 256 …` | 0 | `logs/probe_b256_001.log` | ~13:36 / 13:37 |
| `python scripts/03_train_dp.py --config configs/stackcube.yaml --demo …/trajectory.rgb.pd_ee_delta_pos.physx_cpu.h5 --run stack_1b_s1_d100 --num-demos 100 --total-iters 100000 --batch-size 256 --eval-freq 10000 --num-eval-episodes 5 --seed 1 --data-manifest runs/demos_stack_002/demo_manifest.json --execute` | 0 | `logs/train_stack_1b_s1_d100.log` | 13:47 / 15:29 |
| `04_export_policy.py` + `09_eval_policy.py --set policy_monitor` × 10 시점 | 모두 0 | `logs/export_*`, `logs/monitor_*` | 13:56 / 15:30 |
| `09_eval_policy.py --set policy_select --success-at-end` × 3 후보 | 모두 0 | `logs/select_stack_1b_s1_d100_i0{6,8}0000.log`, `…_i100000.log` | 20:51 / 20:59 |
| `python scripts/11_select_candidate.py …` | 0 | `logs/choose_stack_1b_s1_d100.log` | 20:59 |
| `python scripts/09_eval_policy.py --config configs/stackcube.yaml --policy artifacts/policies/stack_1b_s1_d100_i060000 --set policy_qualify_A --success-at-end --demo-manifest runs/demos_stack_002/demo_manifest.json --out runs/qualify_stack_1b_s1_d100` | 0 | `logs/qualify_stack_1b_s1_d100.log` | 20:59 / 21:08 |
| `python scripts/12_failure_trace.py … --seeds 50100 50101 50102 50104 50105 50106 50107 50108 50109 50111 50103 50110 --videos 12 --out runs/failure_trace_001` | 0 | `logs/failure_trace_001.log` | 23:33 / 23:35 |

로컬(CPU): `python -m pytest -q` → exit 0, 64 passed, 4 skipped.

## 산출물

로컬 백업 위치는 저장소 폴더의 `runs/`, `artifacts/`, `logs/`, `data_backup/` (모두 Git 제외). 서버 파일 193개의 SHA256 목록은 `results/stage1b/backup_sha256_002.txt`이고, 로컬에서 193/193 일치를 확인했다.

| 파일 | 용도 | SHA256 |
|---|---|---|
| `data_backup/demos_stack_002/trajectory.rgb.pd_ee_delta_pos.physx_cpu.h5` | 학습에 쓴 변환 시연 | `60465211ad0913582881586d946e80fe36b871ca34fa21db26b6d167a2ea7a1e` |
| 같은 폴더 `…physx_cpu.json` | 시연 메타데이터 | `2354f29e4903e881c72868c728754a60fc36c9a7b3502a598766a1fddc964a35` |
| 같은 폴더 `trajectory.h5` / `trajectory.json` | 변환 전 원본 시연 | `61d07df7…` / `48cb1119…` |
| `runs/demos_stack_002/demo_manifest.json` (사본 `results/stage1b/demo_manifest_002.json`) | 시연 ID·seed·검증 결과 | 백업 목록 참조 |
| `runs/policy/stack_1b_s1_d100/checkpoints/full_060000.pt` | 전체 학습 상태 (update 60,001회) | `96352d4f1c79683f8fd5328422c01ef513f2732c22decf45679cc44287c389e3` |
| `…/full_080000.pt` | 전체 학습 상태 (update 80,001회) | `b044e8bd31c42822c66e1c2707c352d7e2113e46bc23098a928d394bac180579` |
| `…/full_100000.pt` | 전체 학습 상태 (update 100,000회, 최신) | `8198919225d158169840417490d2807ee8c8ad5f562bf996289f00c03c7a86fe` |
| `…/full_010000.pt` ~ `full_090000.pt` 나머지 7개, `best_eval_*.pt` 2개 | 보존 | 백업 목록 참조 |
| `artifacts/policies/stack_1b_s1_d100_i060000/weights.pt` | 선택된 EMA 정책 | `1bbafd760d09e05a7e90ed292e53bd3935c162400ad8a3f0ebe6033a9c54ed6b` |
| 같은 폴더 `policy_meta.json` | 정책 메타데이터 | `42f96a068fccb0d20f0da6df35110fb21e8e2c1591f40b0b9092bf8aa8026f1e` |
| `artifacts/policies/stack_1b_s1_d100_i080000/weights.pt` | 후보 | `f2460339b301809fe99a84c01d86ff7f36cdb01f6ff23145710d60787c5c8eea` |
| `artifacts/policies/stack_1b_s1_d100_i100000/weights.pt` | 후보 | `f0c21366dc050b0c5d4076a59ba74abf4cc3eed9efac165e499d2da8a40c47d7` |
| `results/stage1b/policy_eval_002/` | monitor·select·qualify 결과와 episode별 기록, 선택 기록, ledger | Git에 포함 |
| `results/stage1b/policy_io_audit_004.json`, `preprocessing_parity_004.csv`, `action_parity_004.csv`, `artifact_tensor_parity_004.json` | 입출력 동등성 검사 | Git에 포함 |
| `results/stage1b/artifact_tensor_parity_005_new_policy.json` | 새 정책의 checkpoint ↔ 내보낸 가중치 비교 | Git에 포함 |
| `results/stage1b/retreat_diagnostic_001.json`, `retreat_steps_001.csv` | Retreat 진단 | Git에 포함 |
| `results/stage1b/failure_trace_001.json`, `runs/failure_trace_001/` (영상 12개, step별 기록) | 실패 위치 기록 | 요약만 Git에 포함 |

전체 checkpoint의 한계: sampler의 epoch 내 위치는 저장되지 않으며 resume loader는 구현·검증하지 않았다 (AMENDMENT_002 3절).

## 검증

| 항목 | PASS / FAIL / NOT RUN | 실제 근거 |
|---|---|---|
| CPU tests | PASS | 서버 68 passed (`logs/pytest_002.log`, 학습 전 코드), 추적 스크립트 테스트 1 passed (`logs/pytest_trace_001.log`). 로컬 최종 코드: 64 passed, 4 skipped (h5py·imageio가 없어 건너뜀) |
| CUDA | PASS | `doctor_002`, 학습 19.8 it/s, 최대 VRAM 약 7.6 GB |
| Vulkan/RGB | PASS | 시연 변환·평가·영상 12개 생성 |
| 시연 검증 | PASS | 129/130 변환 성공, 10개 action replay 모두 성공 |
| 입출력 동등성 (세 경로) | PASS — **단, 이전 정책 가중치로 수행** | `policy_io_audit_004`. 아래 "학습 경로와 배포 경로" 참조 |
| 새 정책 가중치 내보내기 | PASS | 60k/80k/100k 모두 160개 tensor 비트 단위 일치 (로컬 CPU 비교) |
| 새 정책으로 입출력 검사 재실행 | NOT RUN | 성능 gate 미통과로 후속 실행을 중단함 |
| 정책 rollout (qualification) | FAIL (운영 기준 미달) | 4/100 |
| Retreat 5 cm 도달 | PASS | 30/30 도달, 24~28 step |
| 상태+정책 복원 | NOT RUN | 기본 정책 성능 gate 미통과로 후속 정책 포함 복원 audit 및 분기 점검을 보류함. 해당 항목의 통과·실패는 판정하지 않음 |
| 개입 행동 영상 | NOT RUN | 위와 같음 |
| seed split | PASS | 평가 seed 집합은 기존 split·시연 seed와 겹치지 않음(코드 검사). 시연 seed는 `splits/audit` 100~104와만 겹침(보고 사항, 평가에 쓰지 않음) |
| 데이터 완료 상태 | PASS | 모든 평가가 선언한 episode 수를 채움. 제외한 episode 없음 |
| calibration | NOT RUN | 범위 밖 |
| 최종 Test | NOT RUN | 범위 밖. `RUN_FROZEN_TEST` 토큰을 만들지 않음 |

## 관측 결과

### 시연

- 요청 130개 중 129개 변환 성공. 원본 episode 30(seed 31)은 replay 성공 필터에서 탈락.
- 학습: traj id 오름차순 앞 100개, 10,843 transition, 길이 85 / 평균 108.4 / 205 step. held-out: 다음 20개.
- ID·seed 전체 목록은 `results/stage1b/demo_manifest_002.json`.

### 학습

- 100,000 update, batch 256, lr 1e-4, obs/act/pred 2/8/16, seed 1, 새 초기화, DDPM 100 step. 1시간 41분, 최종 표시 loss 0.000184. NaN 없음.

### 정책 평가 (무개입, horizon 200, 주 지표 success_once)

| 평가 | 정책 | 성공/시도 | Wilson 95% | success_at_end | 평균 step (전체 / 성공만) | 추론 호출 | 호출당 시간 |
|---|---|---:|---|---:|---|---:|---:|
| select | 60k EMA | 4/50 | 3.2%~18.8% | 3/50 | 192.1 / 101.0 | 1,203 | 0.313 s |
| select | 80k EMA | 3/50 | 2.1%~16.2% | 3/50 | 194.1 / 101.0 | 1,214 | 0.313 s |
| select | 100k EMA | 3/50 | 2.1%~16.2% | 3/50 | 193.8 / 97.0 | 1,212 | 0.312 s |
| **qualify_A** | **60k EMA** | **4/100** | **1.6%~9.8%** | 3/100 | 195.9 / 97.0 | 2,450 | 0.177 s |

- 선택: 사전 규칙(같은 50개 seed에서 success_once 최댓값, 동률이면 뒤 iteration)에 따라 60k가 자동 선택됐다. 후보 간 차이는 1회(4 대 3)이며 구간이 거의 겹친다. 이 표본으로는 60k가 다른 후보보다 낫다고 말할 수 없다.
- qualification은 한 번만 실행했고(`artifacts/ledger/policy_qualify_A.json`), 결과를 본 뒤 다른 후보로 재시험하지 않았다. 성공 seed: 50218, 50259, 50277, 50282.
- select의 호출당 시간이 더 긴 것은 세 평가를 한 GPU에서 동시에 돌렸기 때문이다.
- monitor (20회, 관찰용): 10k~100k 열 개 시점 중 60k만 1/20, 나머지 0/20.

### 학습 경로와 배포 경로의 비교

| 확인 항목 | 기록 |
|---|---|
| 학습 중 평가 (upstream evaluator, 경로 B) | 60k: 0/5, 80k: 1/5, 100k: 0/5. 전체 11개 시점(0, 10k…100k) 중 80k만 1/5 |
| 자체 adapter 평가 (경로 C) | 60k: 4/50, 80k: 3/50, 100k: 3/50. qualify 60k: 4/100 |
| 입력·행동 정합성 검사 | **수행함, PASS** (`policy_io_audit_004`, `logs/policy_io_audit_004.log`). 같은 기록 프레임 42개 이력: 상태 A↔C, B↔C 최대 오차 0.0, RGB 완전 일치, encoder feature 0.0, 학습 target 행동 정렬 0.0. 같은 seed·같은 행동의 실제 환경 42개 이력: 상태 0.0, RGB 완전 일치. 같은 EMA·같은 난수의 행동 청크 42개: upstream ↔ adapter 0.0. scheduler·controller·카메라 설정 일치 |
| 위 검사의 한계 | 검사는 **학습 전에 이전 정책(`stack_s1_d99`) 가중치로** 실행했다. 전처리·wrapper·adapter 코드는 그 뒤 바뀌지 않았지만, **새 정책 가중치로는 재실행하지 않았다 (NOT RUN)**. 새 정책에 대해 확인한 것은 checkpoint EMA ↔ 내보낸 가중치의 tensor 일치뿐이다 |
| 정보 항목 (판정 제외) | 실제 환경 관측과 시연 파일 관측의 차이: 상태 최대 8.4e-5, RGB 최대 203, 평균 절대차 최대 0.49, 화소의 최대 49%가 조금씩 다름. 파일은 `--use-first-env-state`로 기록돼 `reset(seed)`와 복원 경로가 다르다. 이 차이가 성능에 주는 영향은 측정하지 않았다 |

평가 설정 차이:

| 항목 | 학습 중 평가 (B) | 자체 평가 (C) |
|---|---|---|
| 가중치 | EMA | EMA (같은 시점 checkpoint에서 내보냄) |
| seed | upstream evaluator의 seed (선언한 집합 아님) | 선언한 집합 (monitor / select / qualify_A) |
| episode 수 | 5 | 20 / 50 / 100 |
| horizon | 200 | 200 |
| 성공 판정 | success_once, success_at_end (종료 무시하고 200 step) | success_once absorbing (첫 성공에서 종료), 보조로 200 step까지 실행한 success_at_end |
| 관측·행동 | 상태 25차원 + RGB 2대, `pd_ee_delta_pos`, 8 step 실행 | 같음 (검사 004에서 일치 확인) |
| 추론 난수 | upstream 전역 난수 | (장면, step) 기반 고정 seed |

두 경로 모두 성공이 드물다. 두 경로가 같은 결과를 낸다는 점은 "adapter만의 문제"라는 설명과 맞지 않지만, 학습 중 평가는 5회뿐이라 두 경로의 성공률이 같다고 말할 근거는 아니다.

### 실패 위치 (select seed 12개 재실행, 60k 정책)

재실행 12개 모두 원래 기록과 성공 여부·step 수가 같았다. 아래는 관찰이며 원인 판단이 아니다. 임계값: TCP가 큐브 A 중심 3 cm 이내 = 도달, 큐브 A가 1 cm 이상 상승 = 들어올림, 큐브 A·B 중심 수평거리 2 cm 이내 = 정렬.

| seed | 결과 | 관찰 | 분류 |
|---|---|---|---|
| 50100 | 실패 | TCP가 탁자 높이까지 내려갔으나 큐브 A까지 최소 4.7 cm. step 47부터 그리퍼 닫음. 파지 없음, 큐브 움직임 없음 | 접근 실패 |
| 50101 | 실패 | 최소 거리 5.9 cm. 파지 없음 | 접근 실패 |
| 50102 | 실패 | 최소 거리 4.1 cm. 파지 없음 | 접근 실패 |
| 50104 | 실패 | 최소 거리 6.5 cm (step 102). step 73부터 그리퍼 닫음. 파지 없음 | 접근 실패 |
| 50105 | 실패 | 최소 거리 4.2 cm. 큐브 A가 2 mm 밀림. 파지 없음 | 접근 실패 |
| 50106 | 실패 | 최소 거리 3.1 cm. 큐브 A가 1.9 cm 밀리고 8 mm 들림. 파지 판정 없음 | 접근 실패 (임계값 경계) |
| 50109 | 실패 | 최소 거리 3.7 cm. 그리퍼를 한 번도 닫지 않음. 큐브 A 15 cm, 큐브 B 10 cm 밀림 | 접근 실패 (큐브를 밀어냄) |
| 50111 | 실패 | 최소 거리 8.7 cm. 큐브 B가 11 cm 밀림. 파지 없음 | 접근 실패 (큐브 B를 밀어냄) |
| 50107 | 실패 | step 103 파지, step 117 들어올림(최대 7.6 cm), step 156 이후 파지 상실. 큐브 B까지 수평 최소 10.8 cm | 운반·정렬 실패 |
| 50108 | 실패 | step 50 파지, step 63 들어올림, 끝까지 쥐고 있음. 큐브 B까지 수평 최소 4.8 cm, 놓지 않음 | 운반·정렬 실패 |
| 50103 | 성공 | step 50 파지, step 103 적층, 105에서 성공 | — |
| 50110 | 성공 | step 45 파지, step 98 적층, 105에서 성공 | — |

- 실패 10개 중 8개는 큐브 A를 잡기 전 단계에서, 2개는 잡은 뒤 큐브 B 위로 가져가는 단계에서 멈췄다. 놓기·적층 단계까지 간 실패는 이 표본에 없었다.
- 50107은 스크립트의 자동 표시가 `approach`로 나왔다(파지 시점에도 TCP–큐브 중심 거리가 3.05 cm로 임계값을 조금 넘음). 파지·들어올림 기록이 있으므로 표에서는 운반·정렬 실패로 적었다. 파일의 자동 표시는 고치지 않았다.
- 10개는 대표 표본이 아니라 select seed 앞쪽에서 고른 것이며, 실패 유형의 비율 추정이 아니다.
- 확인하지 않은 것: 큐브 위치를 왜 잘못 짚는지, 그리퍼 명령·카메라·상태 입력 중 무엇이 관련되는지.

### Retreat (폐루프, 목표 5 cm, 명령 상한 5 mm/step, 허용 오차 5 mm, 최대 40 step)

- 시연 prefix에서 고른 자유공간 10개 + 파지 상태 20개, 30/30 도달, 예산 소진 0.
- 사용 step 24~28, step당 실제 이동 약 1.8~1.9 mm, 최종 상승 약 4.5~4.7 cm (허용 오차 안에서 종료).
- 파지 상태 20개 모두 파지 유지. 자유공간에서 큐브 변위는 거의 0.
- eligibility는 최악 48 step(40 + 8) 기준. 정책 포함 분기 실행에서의 동작은 확인하지 않았다.

### 수량

원본 episode 수: 평가 450 (monitor 20×10, select 50×3, qualify 100) + 추적 12 (select seed 재실행)
분기 실행 수: 0 (보류)
성공/실패 수: qualify 4 / 96
Rescue/Harm 분자와 분모: 측정하지 않음
시간/메모리/저장량: 학습 1시간 41분, 최대 VRAM 약 7.6 GB, 서버 저장 약 12 GB, 백업 2.46 GB

### 비용

- 시작 잔액 $13.20 → 종료 잔액 $10.24, **사용 약 $2.96** (한도 $5 이내). 인스턴스 삭제 후 `show instances` 결과가 비어 있음을 확인했다.
- 이 중 **약 $2.2는 유휴 시간**이다. 학습 종료(15:30) 뒤 다음 단계 시작(20:51)까지 5.3시간, qualification 종료(21:08) 뒤 후속 작업(23:33)까지 2.4시간 동안 인스턴스가 켜진 채 대기했다. 완료 알림에 의존한 운영 실수이며 실험 결과에는 영향이 없다.

## 변경과 실패

변경한 파일과 이유: AMENDMENT_002에 기록 (입출력 검사 추가, 학습·평가 설정, 전체 checkpoint, 폐루프 Retreat, 라이선스). 이번 보고와 함께 추가: `scripts/12_failure_trace.py`, `tests/test_failure_trace.py` (실패 위치 기록용, 정책 입력 아님).

실패한 명령 및 원인:
- `policy_io_audit_001` exit 1, `002`·`003` exit 2 — 검사 스크립트 자체의 결함 3건 (AMENDMENT_002 1절). 허용 오차는 바꾸지 않았다.
- 시연 1개(원본 episode 30) 변환 탈락.
- qualification 운영 기준 미달 (4/100).
- 로컬에 중간에 받아 둔 로그 3개가 최종본과 달라 `*.partial_midrun`으로 이름을 바꿔 보존하고 최종본을 받았다.

수행하지 않은 일:
- 시연 500개 두 번째 학습 (사전 규칙상 자동 확대 금지 구간)
- 새 정책 가중치로의 입출력 검사 재실행
- 정책 포함 복원 audit, 원본 2개 분기 점검, 개입 영상 (성능 gate 미통과로 보류, 통과·실패 판정 없음)
- 파일럿 분기 수집, 감시기 학습, calibration, 최종 Test

## 다음 단계

필요 작업 (Stage 1C — 기본 정책 실패 원인 진단, 학습 규모 확대보다 먼저):
1. 보존한 checkpoint와 시연으로 학습 데이터 → 전처리 → 정책 출력 → 환경 행동 적용의 연결 확인. 예: 학습 시연의 관측을 넣었을 때 정책 출력이 시연 행동을 재현하는지(학습 표본 / held-out 20개 구분), 시연 초기 장면에서 정책을 실행했을 때의 성공률.
2. 새 정책 가중치로 입출력 검사 재실행.
3. 실제 환경 관측과 시연 파일 관측의 RGB 차이(화소 최대 49%)가 정책 출력에 주는 영향 측정.
4. 실패 추적에서 본 "큐브 옆 3~9 cm에 내려가 닫는" 양상이 학습 시연 장면에서도 나오는지 확인.

예상 계산량의 산출 근거: 1·3·4는 추론만 필요하다. 실측 5.3 s/episode(단독 실행) 기준 100 episode에 약 9분, 전체 1시간 이내로 예상하며 $0.3~0.5 수준이다. 일부(1의 행동 재현 오차)는 로컬 CPU로 가능한지 먼저 확인한다.

동결/승인이 필요한 사항: Stage 1C 범위와 GPU 대여 승인. 학습 규모(시연 수, update 수)는 진단 뒤에 다시 정한다.

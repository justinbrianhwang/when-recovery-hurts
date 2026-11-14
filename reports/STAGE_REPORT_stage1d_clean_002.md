# Stage 보고서 — Stage 1D 재실행 (D500-clean)

D500-clean 기본 정책 qualification 37/100으로 운영 gate(50/100)에 미달했다. 자동 추가 학습은 하지 않았다.

실행 ID: `stage1d_clean_002` (정책 run `stack_1d_s1_d500_clean_v1`)
단계: Stage 1D 재실행 — 결함 궤적 전체 제외·대체 후 D500-clean 학습
작업/정책 seed: StackCube-v1 / 학습 seed 1
실제 실행 환경: Vast.ai 인스턴스 60601481 (RTX 3090 24 GB, Ryzen 7 9700X, RAM 60 GB, 표시 단가 $0.2281/h), 2026-11-14 05:45~08:13 UTC. 삭제 완료.
코드/upstream 버전: 저장소 `b0c5406` (계획 고정 시점, 미커밋 변경 없음) / ManiSkill `a4a4f9272ad64b1564035874b605ceb687b63ed8`
변경 기록: `amendments/AMENDMENT_004_stage1d_data_integrity_clean_rebuild.md`

## 요약

- 보유한 변환 궤적 545개를 전수 검사해 5개를 제외했다. 알려진 4개와, 의심 대상이던 원본 536이다. 보호 집합 29개는 모두 통과했다.
- 학습 집합은 기존 99 + 새 시연 397 + 대체 4 = 500개다. 유지된 496개는 dataset별 hash가 이전 기록과 같다.
- 100,000 update 학습을 한 번 끝냈다. 후보 60k / 80k / 100k는 select 50개에서 21 / 23 / 13이었고 규칙대로 80k가 선택되었다.
- 선택 정책의 qualification B는 **37/100** (Wilson 95% 구간 28.2%~46.8%)이다. 10~49 구간이므로 기준 미달이고, 복원 audit와 분기 점검은 실행하지 않았다.
- Stage 1C와 같은 장면 20개에서 정책 실행은 학습 장면 7/10, 학습에 쓰지 않은 장면 3/10이었다 (Stage 1B 정책은 6/10, 0/10).
- 이 결과로 "오염 제거가 성능을 올렸다"거나 "시연 수가 성능을 올렸다"고 나눠 말할 수 없다. 두 가지가 함께 바뀌었다.

## 실제 수행한 명령

인자 전체는 `logs/stage1d_clean_002.status`(사본 `results/stage1d_clean/commands_stage1d_clean_002.status`)에 있다. 시각은 UTC.

| 명령 | 종료 코드 | 로그 경로 | 시작/종료 시각 |
|---|---:|---|---|
| (로컬) `17_stage1d_clean.py audit …` | 0 | `runs/stage1d_clean_002/audit/` | 대여 전 |
| (로컬) `17_stage1d_clean.py build …` | 0 | `runs/stage1d_clean_002/data/` | 대여 전 |
| (로컬) `17_stage1d_clean.py gate --data …` | 0 | — | 대여 전 |
| `python -m hci doctor` / `python -m pytest -q` | 0 / 0 | `logs/s1dc_doctor.log`, `logs/s1dc_pytest.log` | 05:51:38 / 05:51:45 |
| `17_stage1d_clean.py gate --data …` (서버 재확인) | 0 | `logs/s1dc_gate_data.log` | 05:51:45 / 05:51:47 |
| `17_stage1d_clean.py replay …` | 0 | `logs/s1dc_replay.log` | 05:51:47 / 05:51:56 |
| `08_policy_io_audit.py --policy …d100_i060000 --demo …/io_parity_sample_clean_v1.h5 --trajectories 10` | 0 | `logs/s1dc_io_dataset_path.log` | 05:51:56 / 05:53:06 |
| `17_stage1d_clean.py gate --train-start --cap-usd 2.0 --spent-usd 0.28 --usd-per-hour 0.2281 --train-limit-s 7200 --reserve-usd 0.35` | 0 | `logs/s1dc_gate_train_start.log` | 05:53:30 / 05:53:31 |
| `timeout 7200 python scripts/03_train_dp.py --run stack_1d_s1_d500_clean_v1 --num-demos 500 --total-iters 100000 --batch-size 256 --eval-freq 10000 --num-eval-episodes 5 --seed 1 …` | 0 | `logs/train_stack_1d_s1_d500_clean_v1.log` | 05:53:31 / 07:39:33 |
| `04_export_policy.py`, `09_eval_policy.py --set policy_monitor` × 10 | 모두 0 | `logs/export_…`, `logs/monitor_…` | 학습 중 ~ 07:40:45 |
| `09_eval_policy.py --set policy_select --success-at-end` × 3 (순차) | 0 | `logs/select_…_i060000/080000/100000.log` | 07:40:45 / 07:53:22 |
| `11_select_candidate.py …` | 0 | `logs/choose_….log` | 07:53:22 |
| `08_policy_io_audit.py --policy …clean_v1_i080000 --checkpoint …/full_080000.pt --trajectories 10` | 0 | `logs/s1dc_io_parity_selected.log` | 07:53:22 / 07:54:32 |
| `09_eval_policy.py --set policy_qualify_B --success-at-end` | 0 | `logs/qualify_….log` | 07:54:32 / 08:02:51 |
| `14_scene_rollouts.py … --other-policy --cells F_policy R_policy` | 0 | `logs/s1dc_scene_rollouts.log` | 08:02:51 / 08:05:32 |
| `15_offline_action_diag.py … --other-policy` | 0 | `logs/s1dc_offline.log` | 08:05:32 / 08:07:10 |
| `run_clean.sh post` | 0 (실행 안 함: 37 < 50) | status의 `POST NOT RUN` | 08:07 |
| `run_clean.sh pack` | 0 | `logs/s1dc_pack.out` | 08:07 |

실패한 명령은 없다. 처리량 프로브는 반복하지 않았다 (`stage1d_001` 측정값 사용).

## 산출물

| 파일 | 용도 | SHA256 |
|---|---|---|
| `runs/stage1d_clean_002/plan.json` | 대여 전 고정한 계획 | `050e31f2d51a7905…` |
| `runs/stage1d_clean_002/data/train_d500_clean_v1.h5` | 학습 파일 (1.82 GB). 서버 업로드 후 hash 일치 | `93416a4a8b7e2203…` |
| `runs/policy/stack_1d_s1_d500_clean_v1/checkpoints/full_080000.pt` | 선택 후보의 전체 checkpoint | `fcbd79737c72a9db…` |
| `artifacts/policies/stack_1d_s1_d500_clean_v1_i080000/weights.pt` | 선택 정책 (EMA) | `faacd2d86464c725…` |
| `artifacts/ledger/policy_qualify_B.json` | B 사용 기록 (재실행 거부) | `results/stage1d_clean/policy_eval/`에 사본 |
| `stage1d_clean_out.tgz` (1.47 GB) | 백업 묶음 | `2f1c1f68794e4aa0…` (다운로드 후 일치) |

묶음 안 209개 파일 목록은 `results/stage1d_clean/backup_sha256_stage1d_clean_002.txt`이고 로컬에서 209/209 일치했다. 10k 단위 전체 checkpoint 10개와 내보낸 정책 10개가 모두 들어 있다. 학습 파일은 로컬에서 만들어 서버에서 hash를 확인한 것이므로 묶음에 다시 넣지 않았다. `stage1d_001`의 3.29 GB 묶음은 그대로 보존한다.

## 검증

| 항목 | PASS / FAIL / NOT RUN | 실제 근거 |
|---|---|---|
| CPU tests | PASS | 로컬 93 passed / 3 skipped, 서버 exit 0 |
| CUDA | PASS | `doctor` exit 0 |
| `structural_validation` | PASS | 500개 shape·dtype·범위·카메라 순서 |
| `source_and_split_validation` | PASS | 보호 29개와 겹침 0, 초기 배치 529개 모두 다름, 선언된 split과 새 겹침 없음 |
| `episode_content_integrity` | PASS | 학습 500개 모두 PASS. 만든 파일 자체를 다시 검사 |
| `sampled_action_replay` | PASS | 14/14 성공 (대체 4 + 기존 5 + 새 5), 첫 성공 step이 파일과 같음 |
| `policy_io_parity` (새 dataset 경로, 60k 정책) | PASS | 10개 궤적, 이력 70개·청크 70개, 차이 0.0 |
| 선택 정책 checkpoint ↔ export | PASS | tensor 160개 bit 일치 |
| 선택 정책 I/O parity | PASS | 이력 70개 상태·RGB·encoder feature 차이 0.0, 청크 70개 upstream 대 adapter 0.0 |
| qualification B 단일 사용 | PASS | ledger에 80k 정책으로 1회 기록 |
| 상태+정책 복원 audit | NOT RUN | 37/100이 gate 미만 |
| 소규모 분기 수집 | NOT RUN | 같은 이유 |
| 개입 행동 영상 | NOT RUN | 같은 이유 |
| seed split | PASS | 평가 seed와 시연 장면 겹침 없음 (09 실행 시 검사) |
| 데이터 완료 상태 | PASS | monitor 10 × 20, select 3 × 50, qualify 100 모두 완료. 제외한 episode 없음 |

## 관측 결과

### 1. 전수 검사

| 역할 (stage1d_001 기준) | 수 | PASS | FAIL |
|---|---:|---:|---:|
| 기존 학습 (D100) | 100 | 99 | 1 |
| 새로 채택했던 시연 | 400 | 397 | 3 |
| 보호 (held-out 20 + 잔여 9) | 29 | 29 | 0 |
| 미채택 성공분 | 16 | 15 | 1 |
| 합계 | 545 | 540 | 5 |

- FAIL 5개는 원본 31, 230, 382, 414, 536이다. 모두 변환에 실패한 원본(30, 229, 381, 413, 535)의 바로 다음 episode이고, 저장된 첫 상태가 그 실패한 episode의 것과 일치했다. reset 직후의 장면은 선언된 원본의 첫 상태와 일치했다.
- 다섯 개 모두 확정 신호 4종(success 없는 terminated, truncated 재시작, 첫 관측 불일치, 첫 상태 불일치)과 선별 신호가 함께 나왔다. QUARANTINE은 없었다.
- 원본 536은 예상이 아니라 검사 결과로 FAIL이다.
- 결함 궤적의 앞부분에 보호 장면이나 평가 장면이 섞여 든 경우는 없었다.
- 음성 대조: 길지만 연속인 학습 ID 435(319 step), 447(356 step)은 PASS로 남았다.
- 정상 궤적 1개(옛 학습 ID 339)는 마지막에 success가 켜졌다 꺼졌다 다시 켜진다. 결함이 아니라 정보로 기록했다 (AMENDMENT_004 3절).
- 선별 기준 `cube_path_not_source`(5 cm)에 대해 실제 값은 PASS 궤적 최대 3.4 cm, FAIL 궤적 최소 15.7 cm였다.

### 2. stage1d_001 오염 범위 (출처: `integrity_summary.json`)

| 항목 | 값 |
|---|---:|
| 실패 접두부 step | 722 / 54,521 (1.3%) |
| reset dummy 전이 | 4 |
| 접두부나 reset이 걸친 학습 window | 730 / 54,021 (1.4%) |
| target에 dummy 행동이 들어간 window | 64 |

개수일 뿐이며, 정책에 미친 영향의 크기는 측정하지 않았다.

### 3. D500-clean 구성

| 구성 | 수 |
|---|---:|
| 유지된 기존 시연 | 99 |
| 유지된 새 시연 | 397 |
| 대체 (원본 533, 534, 537, 538 → 학습 ID 30, 199, 350, 381) | 4 |
| 합계 | 500 (53,820 transition) |

- 그리퍼 명령은 500개 전체에서 ±1뿐이다 (dummy 행동 없음).
- 쓰지 않고 남은 대체 후보는 11개다.
- 첫 성공이 horizon 200을 넘는 시연 2개(435, 447)는 그대로 있다.

### 4. 학습과 후보 선택

- 100,000 update, 1시간 46분, 마지막 loss 0.00046.
- monitor 20개(관찰용): 10k부터 0, 2, 4, 5, 5, 6, 7, 6, 6, 5.
- 학습 중 upstream 평가 5회(무작위 seed, 관찰용): 최고 4/5.

| 후보 (EMA) | Stage 1B D100: success_once / at_end | D500-clean: success_once / at_end |
|---|---:|---:|
| 60k | 4/50 / 3/50 | 21/50 / 16/50 |
| 80k | 3/50 / 3/50 | **23/50** / 19/50 |
| 100k | 3/50 / 3/50 | 13/50 / 12/50 |

같은 select 50개 seed다. 선택에 쓴 개발 결과이며, Stage 1B는 동시 실행, 이번은 순차 실행이었다.

### 5. Qualification

| | 장면 | success_once | Wilson 95% | success_at_end |
|---|---|---:|---|---:|
| Stage 1B 60k (legacy, 결함 시연 1개 포함 D100) | `policy_qualify_A` 50200~50299 | 4/100 | 1.6%~9.8% | 3/100 |
| D500-clean 80k | `policy_qualify_B` 50300~50399 | **37/100** | 28.2%~46.8% | 31/100 |

- A와 B는 서로 다른 장면이다. 합치거나 paired 비교를 하지 않는다.
- 37/100은 운영 기준 50/100 미만이다. 구간은 기술용이며 모집단 성공률에 대한 보장이 아니다.
- 성공한 episode의 평균 완료 step은 112.8, 정책 호출당 0.166초.
- 둘 다 개발용 정책 확보 검사이며 논문 최종 Test가 아니다.

### 6. Stage 1C와 같은 표본의 비교 (장면 20개, 이력 100개 × noise 4개)

20개 장면 모두 내용 검사를 통과해 표본을 바꾸지 않았다.

| | Stage 1B 60k | D500-clean 80k |
|---|---:|---:|
| F-policy, 학습 장면 | 6/10 | 7/10 |
| F-policy, 학습에 쓰지 않은 장면 | 0/10 | 3/10 |
| R-policy, 학습 장면 | 5/10 | 7/10 |
| R-policy, 학습에 쓰지 않은 장면 | 0/10 | 3/10 |
| 큐브에 도달하지 못한 미학습 장면 (F) | 6 | 2 |
| F와 R의 성공 여부가 같은 장면 | 19/20 | 20/20 |
| 실행 8-step xyz MAE, 학습 / 미학습 | 0.0033 / 0.0105 | 0.0034 / 0.0088 |
| denoising loss (100 timestep 평균), 학습 / 미학습 | 0.00038 / 0.0174 | 0.00058 / 0.0075 |

- "학습 장면" 10개는 이번에도 학습 집합에 들어 있고, "미학습 장면" 10개는 이번에도 학습에서 제외되어 있다.
- 미학습 장면에서 실패한 7개의 가장 먼 단계: 큐브 미도달 2, 파지 후 들지 못함 2, 들었으나 정렬 못함 2, 정렬했으나 성공 판정 없음 1.
- 장면 20개는 개발 판단용이다. 일반화 성능의 추정치가 아니다.
- RGB 민감도(D4)는 반복하지 않았다.

### 7. 시간과 비용

- 가동 2시간 28분. 유휴 대기 없이 학습 → 선택 → qualification → 비교 → 백업 → 삭제를 이어서 실행했다.
- 크레딧 9.7887 → 8.8998, **이번 인스턴스 약 $0.89**. 표시 단가로 계산한 값(2.47 h × $0.2281 ≈ $0.56)보다 약 $0.33 많다. 업로드 2.4 GB와 다운로드 1.5 GB의 전송 요금으로 추정하지만 청구 내역으로 확인하지는 않았다.
- Stage 1D 누적: $0.24 + $0.89 = **약 $1.13** (상한 $2.00).
- 학습 중 VRAM·RAM 기록: `logs/s1dc_train_gpu.csv`.

## 변경과 실패

변경한 파일과 이유:
- `src/hci/integrity.py` (신규): 내용 무결성 판정, 내용 출처 대조, 학습 window 집계.
- `scripts/17_stage1d_clean.py` (신규): 전수 검사, 제외·대체, 구성, 재생 검사, 학습 시작 gate.
- `scripts/stage1d/run_clean.sh` (신규): 서버 gate, 학습(자기 process group에서만 중단), 선택, 비교, 백업.
- `scripts/02_prepare_demos.sh`, `scripts/stage1d/run.sh`: 내용 검사 없이 변환기를 쓰지 못하게 하는 guard.
- `tests/test_stage1d_clean.py` (신규, 15개): 명세의 회귀 항목 10개 포함.
- Codex 검토 6건 반영: 원본 경로 대조 신호 추가, 구성 단계에서 검사 재계산 후 byte 대조, gate가 재생·parity 보고서의 범위까지 확인, 중단 전 프로세스 확인, monitor 시간 제한, 0~9 구간에서는 비교 생략.

실패한 명령 및 원인: 없음.

수행하지 않은 일: 복원 audit, 분기 수집, 영상, D100-clean 대조 학습, 변환기 수정·재변환, 추가 학습, calibration, 최종 Test.

한계:
- 학습 seed 1개, qualification 1회다.
- 데이터 구성(결함 제외)과 데이터 수(100 → 500)가 동시에 바뀌었다. 각각의 효과는 분리되지 않는다.
- 내용 검사의 PASS는 선언한 신호에 대한 것이다. 5 cm·2 cm·0.3 rad는 이 데이터에서 잰 선별 기준이다.
- 100k 후보가 60k·80k보다 낮았다 (13/50). 원인은 조사하지 않았다.
- 변환기의 동작은 코드를 읽고 데이터에서 확인한 것이며, 수정 후 재실행으로 검증하지 않았다.

## 다음 단계

기본 정책 gate가 아직 열리지 않았으므로, 복원 audit와 분기 수집은 계속 보류다. 해당 항목의 통과·실패는 판정하지 않는다.

결정이 필요한 것: 37/100 정책으로 무엇을 할지. 이 단계의 규칙상 자동으로 이어지는 작업은 없다. 가능한 방향만 적는다.

| 방향 | 내용 | 예상 계산량 |
|---|---|---|
| 데이터 추가 | 원본 1000개 중 남은 약 450개를 변환해 시연을 늘림. 변환기 결함이 그대로이므로 내용 검사를 반드시 거쳐야 하고, 새 qualification 집합이 필요하다 (A·B 모두 사용됨) | 변환 약 10분 + 학습 약 1시간 50분 + 평가 약 30분, 약 $0.9~1.0 |
| 기준 재검토 | 50/100 기준 자체를 개발 단계 운영 기준으로 다시 정함. 정하면 그 사실을 변경 기록에 남겨야 한다 | 계산 없음 |
| 실패 위치 진단 | 37/100의 실패 63개가 어느 단계에서 멈추는지 기록 (이미 있는 정책, 추론만) | 약 15분, 약 $0.1~0.2 |

- 어느 방향이든 `policy_qualify_A`와 `policy_qualify_B`는 다시 쓸 수 없다.
- 동결/승인이 필요한 사항: 위 선택. 최종 Test·calibration과는 무관하다.

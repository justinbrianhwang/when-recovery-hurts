# Stage 보고서 — Stage 1D (D500 확장), 데이터 검사에서 중단

데이터 검사에서 중단하여 D500 본 학습 또는 qualification은 실행하지 않았다.

실행 ID: `stage1d_001` (정책 run `stack_1d_s1_d500`은 약 2천 update에서 중단, checkpoint 없음)
단계: Stage 1D — 시연 500개 확장 및 기본 정책 재확보
작업/정책 seed: StackCube-v1 / 학습 seed 1 (학습 미완료)
실제 실행 환경: Vast.ai 인스턴스 60595133 (RTX 3090 24 GB, i7-11700K, RAM 62 GB, $0.2833/h), 2026-11-14 04:41~05:14 UTC. 삭제 완료.
코드/upstream 버전: 저장소 `78c5da9` (계획 고정 시점 `464732b`, 이후 변경은 실행기의 따옴표 한 줄) / ManiSkill `a4a4f9272ad64b1564035874b605ceb687b63ed8`

## 요약

- 새 시연 400개를 변환해 D500 학습 파일을 만들었고, 사전에 정한 데이터 검사는 모두 통과했다.
- 본 학습을 시작한 직후, 학습 궤적 500개 중 **4개**가 "직전 원본 episode의 실패한 재생 + 장면 순간 이동 + 의미 없는 행동 1개 + 채택된 episode"로 이어 붙은 궤적임을 확인했다.
- 명세 4절("실제 obs/action 정렬 결함이 추가로 발견되면 학습을 중단하고 별도 보고")에 따라 약 2천 update에서 학습을 멈췄다.
- 후보 평가, `policy_qualify_B`, 새 정책 진단, 복원 audit, 분기 수집은 실행하지 않았다. `policy_qualify_B`는 미사용 그대로다.
- 이 결함은 Stage 1B의 D100에도 1개 있었다 (`traj_30`). Stage 1C에서 "첫 성공이 step 201인 느린 시연"으로 기록한 것의 실제 내용이다.

## 실제 수행한 명령

모든 명령은 `logs/stage1d_001.status`(사본 `results/stage1d/commands_stage1d_001.status`)에 인자 전체와 함께 있다. 시각은 UTC.

| 명령 | 종료 코드 | 로그 경로 | 시작/종료 시각 |
|---|---:|---|---|
| (로컬) `python scripts/16_stage1d_data.py plan …` | 0 | `runs/stage1d_001/plan/` | 대여 전 |
| `python -m hci doctor --out runs/stage1d_001/doctor` | 0 | `logs/s1d_doctor.log` | 04:46:03 / 04:46:06 |
| `python -m pytest -q` (서버) | 0 | `logs/s1d_pytest.log` | 04:46:06 / 04:46:11 |
| `16_stage1d_data.py subset` | 0 | `logs/s1d_subset.log` | 04:46:11 |
| `python -m mani_skill.trajectory.replay_trajectory --traj-path runs/stage1d_001/new_source/trajectory.h5 --use-first-env-state -c pd_ee_delta_pos -o rgb -b physx_cpu --num-envs 1 --count 420 --save-traj` | 0 | `logs/s1d_convert.log` | 04:46:11 / 04:54:19 |
| `16_stage1d_data.py build` | 0 | `logs/s1d_build.log` | 04:54:19 / 04:55:45 |
| `16_stage1d_data.py validate` | 0 | `logs/s1d_validate.log` | 04:55:45 / 04:55:56 |
| `08_policy_io_audit.py --policy …d100_i060000 --demo …/io_parity_sample/…` | 0 | `logs/s1d_io_dataset_path.log` | 04:55:56 / 04:57:31 |
| `03_train_dp.py --run stack_1d_probe_b256 --num-demos 500 --total-iters 1000 --batch-size 256 …` (프로브) | 0 | `logs/s1d_probe_b256.log` | 04:57:53 / 05:00:49 |
| `timeout 9000 python scripts/03_train_dp.py --run stack_1d_s1_d500 --num-demos 500 --total-iters 100000 …` | 운영자 중단 | `logs/train_stack_1d_s1_d500.log` | 05:01:30 / 약 05:04:40 |
| 불연속 검사 (`runs/stage1d_001/quirk2.py`) | 0 | `runs/stage1d_001/stop_check/discontinuity_check.txt` | 05:05 |

중단 명령 한 번은 기록에 남지 않았다. 처음 보낸 중단 명령이 자기 자신의 SSH 세션까지 종료시켜 사유 기록이 빠졌고, 30초 뒤 다시 기록했다 (status의 `STOPPED BY OPERATOR` 줄). 학습 프로세스는 첫 명령에서 종료되었다.

## 산출물

| 파일 | 용도 | SHA256 |
|---|---|---|
| `runs/stage1d_001/plan/stage1d_plan.json` | 대여 전 고정한 계획 (역할, 변환 창, 후보·qualification 규칙, 예산) | `f9ebe01179ec6ea3…` |
| `runs/stage1d_001/data/train_d500/trajectory.rgb.pd_ee_delta_pos.physx_cpu.h5` | 학습 전용 D500 파일 (1.83 GB, **결함 궤적 4개 포함**) | `3c138e4be42741bb…` |
| `runs/stage1d_001/new_source/trajectory.rgb.pd_ee_delta_pos.physx_cpu.h5` | 새 변환 416개 (1.53 GB) | `8b0ff52e37e3331f…` |
| `runs/stage1d_001/data/train_d500_manifest.json` 외 3개 | 원본 ID 매핑, 보호 집합, 중복 검사 | `results/stage1d/`에 사본 |
| `runs/stage1d_001/stop_check/discontinuity_check.txt` | 중단 근거 | `results/stage1d/`에 사본 |
| `stage1d_stop.tgz` (3.29 GB, 서버에서 생성) | 백업 묶음 | `66692866ebafb7d8…` (다운로드 후 일치) |

묶음 안 50개 파일의 목록은 `results/stage1d/backup_sha256_stage1d_001_stop.txt`이며 로컬에서 50/50 일치했다. 프로브 모델의 `.pt`는 버리는 모델이므로 백업하지 않았다.

## 검증

| 항목 | PASS / FAIL / NOT RUN | 실제 근거 |
|---|---|---|
| CPU tests | PASS | 로컬 78 passed / 3 skipped, 서버 exit 0 |
| CUDA | PASS | `doctor` exit 0 (이번 호스트는 컨테이너 수정 불필요) |
| 보존 자료 해시 | PASS | Stage 1B 193/193, Stage 1C 261/261 |
| D100 payload 동일 | PASS | 100개 모두 dataset별 hash 일치 (복사, 재렌더링 없음) |
| 보호 집합 비중복 | PASS | 학습 500개와 보호 29개의 seed 겹침 0, 초기 배치 529개 모두 다름 |
| 새 dataset 경로 I/O 정렬 | PASS | 기존 3 + 새 3 시연, 이력 42개·청크 42개, 상태·RGB·target 정렬 차이 0.0 |
| expert action replay | PASS | 기존 5 + 새 5, 10/10 성공, 첫 성공 step이 파일과 같음 |
| **학습 궤적의 내용 연속성** | **FAIL** | 500개 중 4개에 장면 불연속 (아래) |
| D500 학습 | NOT RUN (중단) | 약 2천 / 100,000 update |
| 후보 선택 / qualify_B | NOT RUN | ledger에 `policy_qualify_A.json`만 있음 |
| 새 정책 parity, Stage 1C 비교 진단 | NOT RUN | |
| 상태+정책 복원 audit, 소규모 분기 | NOT RUN | |
| seed split | PASS | 새 시연 seed 133~566, 선언된 split과 겹침 없음 |

"학습 궤적의 내용 연속성"은 사전에 정한 검사 항목이 아니었다. 학습 시작 후 다른 수치(변환 파일 첫 상태와 원본 상태의 차이가 1.75로 큼)를 확인하다가 발견했다.

## 관측 결과

### 데이터 구성

| 집합 | 수 |
|---|---:|
| D100 (기존 학습, 복사) | 100 |
| Dnew400 (새로 채택) | 400 |
| D500 | 500 (54,521 transition) |
| V20 (보호) | 20 |
| 보호 잔여 | 9 |
| 새 변환 창 | 420 → 성공 416, 실패 4 (원본 episode 229, 381, 413, 535) |
| 채택하지 않은 성공분 | 16 |

### 중단 사유: 이어 붙은 궤적 4개

| 학습 ID | 역할 | 원본 episode | 길이 | 앞부분 (직전 실패 episode) | 순간 이동 시점 | 그 시점의 행동 |
|---:|---|---:|---:|---|---:|---|
| 30 | D100 | 31 | 205 | 원본 30, 108 step, 성공 0 | 108 | `[-0.091, 0.521, 0.517, 0.504]` |
| 199 | 새 시연 | 230 | 212 | 원본 229, 108 step, 성공 0 | 108 | `[0.706, -0.909, -0.055, -0.716]` |
| 350 | 새 시연 | 382 | 194 | 원본 381, 95 step, 성공 0 | 95 | `[-0.390, 0.825, -0.169, -0.043]` |
| 381 | 새 시연 | 414 | 534 | 원본 413, 411 step, 성공 0 | 411 | `[0.890, 0.670, 0.692, -0.137]` |

- 네 궤적 모두 **변환에 실패한 원본 episode의 바로 다음 episode**다. 순간 이동 직후의 큐브 위치는 해당 원본 episode의 첫 상태와 정확히 같다.
- 해석: 고정 commit의 변환기는 실패한 재생을 저장하지 않지만 기록 버퍼도 비우지 않아, 다음 episode가 그 뒤에 이어 붙는다. 이 해석은 위 관측(실패 직후 episode에서만 발생, 앞부분 길이가 직전 원본 길이와 대응)에 근거한 것이며 upstream 코드를 수정해 확인하지는 않았다.
- 순간 이동 시점의 행동은 그리퍼 값이 0.504, -0.716, -0.043, -0.137이다. 정상 시연의 그리퍼 값은 ±1뿐이다. Stage 1B manifest의 `train_gripper_values`에 있던 0.5036이 이것이다.
- 영향 범위: 앞부분(실패한 시도) 722 step은 D500 transition의 1.3%. D100에서는 108 / 10,843 = 1.0%.
- 영향의 크기(정책 성능에 미치는 정도)는 측정하지 않았다.
- 학습에 쓰지 않은 성공분 16개 중 원본 536(실패한 535의 다음)도 같은 형태일 것으로 예상되지만 확인하지 않았다.

### 이전 보고서에 대한 영향

- Stage 1C의 "`traj_30`은 첫 성공이 step 201이라 horizon 안에 끝나지 않는 시연"은 사실 기술로는 맞지만 원인이 다르다. 느린 시연이 아니라 이어 붙은 궤적이다.
- 이번에 "첫 성공이 200을 넘는 시연"으로 기록된 5개 중 3개(학습 ID 30, 199, 381)가 이 경우다. 나머지 2개(435, 447)는 불연속이 없다. 350은 불연속이 있지만 첫 성공이 190이라 그 목록에는 없다.
- Stage 1B 정책(4/100)의 학습 데이터에도 같은 비율의 결함이 있었다. 그 결과 수치는 바뀌지 않는다.
- Stage 1C 정정: "F-policy와 R-policy 40쌍 중 39쌍"은 **20쌍 중 19쌍**이다 (AMENDMENT_003 6절).

### 시간/메모리/비용

- 변환 420개 8분 8초, D500 구성 1분 26초.
- 프로브: 19.2 update/s, VRAM 최대 11.7 GB, RAM 16.8 GB. 100k update 예상 약 90분.
- 가동 33분, 유휴 대기 없음. 크레딧 10.0298 → 9.7887, **실제 비용 약 $0.24** (상한 $2.00). 인스턴스 삭제 후 목록 0개 확인.
- 대여 단가는 검색 화면의 $0.207/h가 아니라 디스크 60 GB 포함 $0.2833/h였다.

## 변경과 실패

변경한 파일과 이유:
- `scripts/16_stage1d_data.py` (신규): 계획, 원본 부분집합, D500 구성, 검증.
- `scripts/stage1d/run.sh` (신규): 단계 실행기. parity를 qualification 앞에 두고 후보는 순차 평가.
- `scripts/14_scene_rollouts.py`, `scripts/15_offline_action_diag.py`: `--other-policy` 옵션 추가 (이번에는 실행되지 않음).
- `tests/test_stage1d_data.py` (신규, 7개), `amendments/AMENDMENT_003_stage1d_d500_data_expansion.md` (신규).
- Codex 검토(대여 전)에서 6건을 받아 반영했다: 학습 시간 상한, monitor가 멈추지 않는 경우, suffix 재시도 시 데이터 경로, 보존 파일 누락을 실패로 처리, 계획과 split 파일의 hash 대조, 백업 목록을 압축 해제본에서 계산.

실패한 명령 및 원인: 없음 (exit code 기준). 학습은 운영자 중단.

수행하지 않은 일: D500 학습 완료, 후보 선택, `policy_qualify_B`, 새 정책 I/O parity, Stage 1C 비교 진단, 복원 audit, 분기 수집. 결함 궤적을 제외하거나 고친 데이터 구성도 만들지 않았다 (선정 규칙 변경이므로 결정이 필요하다).

한계: 연속성 검사는 큐브 B 위치와 관절 각도의 step 간 변화량 기준이다(임계값 2 cm, 0.3 rad). 더 작은 이상은 잡지 못한다. 보호 집합 29개와 미채택 16개에는 이 검사를 돌리지 않았다.

## 다음 단계

결정이 필요한 것: 결함 궤적 4개의 처리.

| 선택지 | 내용 | D100 보존 | 비고 |
|---|---|---|---|
| A | 선언한 그대로 4개를 포함해 학습 | 유지 | D100과 결함 비율이 비슷해(1.0% 대 1.3%) 비교 조건은 같다. 알려진 결함을 학습에 넣는다. |
| B | 새 시연 3개만 제외하고, 미채택 성공분에서 원본 순서대로 불연속이 없는 3개로 대체. `traj_30`은 그대로 | 유지 | 선정 규칙에 "불연속 궤적 제외"가 추가된다. 정책 결과를 보기 전의 변경이다. |
| C | 4개 모두에서 실패한 앞부분을 잘라내거나 4개를 제외 | 깨짐 | 데이터 수와 데이터 품질이 동시에 바뀐다. |

- 어느 쪽이든 변환 결과는 보존되어 있어 다시 변환할 필요는 없다 (B·C는 구성과 검증만 다시).
- 재실행 예상: 설치 5분, 업로드(약 3.4 GB) 10~15분, 프로브 3분, 학습 약 95분, 선택·parity·qualification 약 30분, 진단 약 15분, (통과 시) audit·분기 약 15분, 백업 약 15분. 약 3.2시간 × $0.28 ≈ **$0.9**. 이번 사용분과 합쳐 약 $1.15로 상한 $2.00 안이다.
- 재실행 시에는 연속성 검사를 데이터 구성 단계의 정식 검사로 넣는다.
- 동결/승인이 필요한 사항: 위 선택, 그리고 선택에 따른 AMENDMENT_003 보완. 최종 Test·calibration과는 무관하다.

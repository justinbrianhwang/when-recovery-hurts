# Stage 보고서 — Stage 2B 대체 난수 Continue를 포함한 네 arm 대조

D500-clean 80k 정책을 그대로 두고 새 train 장면 24개에서 네 arm의 분기 1,344회를 끝까지 수집했다. 이 표본에서 Refresh의 평균 결과는 두 Continue 기준과 0.03 이내로 차이 났고, Retreat–Retry는 두 기준보다 0.16~0.25 낮았다. 개발용 기술통계이고, 개입의 효용이나 안전성을 입증하거나 반증하지 않는다.

실행 ID: `stage2b_001`
단계: Stage 2B — 고정 정책의 네 arm 대조 (복원 audit + 분기 수집)
작업/정책 seed: StackCube-v1 / 정책 `stack_1d_s1_d500_clean_v1_i080000` (학습 seed 1, 재학습 없음)
실제 실행 환경: Vast.ai 인스턴스 60690178 (RTX 3090 24 GB, driver 580.65.06, Ryzen 9 5950X, 단가 $0.2452/h), 2026-11-14 19:54~21:07 UTC. 삭제 완료.
코드/upstream 버전: 저장소 `d82168b` + 이 단계의 새 파일(실행 시점에는 미커밋, 파일별 SHA256을 plan에 고정) / ManiSkill `a4a4f9272ad64b1564035874b605ceb687b63ed8`
변경 기록: `amendments/AMENDMENT_006_stage2b_noise_control.md`

## 상태

```text
base_policy_qualification: BELOW_ORIGINAL_GATE (37/100 < 50/100)
stage2a: COMPLETED_EXPLORATORY_ONLY
stage2b: EXPLORATORY_FOUR_ARM_CONTROL (completed)
formal_study_ready: NOT_ESTABLISHED
calibration: NOT_RUN
final_test: NOT_RUN
```

qualification 판정은 그대로다. 이 실험은 Stage 2A 결과를 본 뒤에 정한 설계이고, 사전 등록된 확증 실험도 독립 검증도 아니다.

## 요약

네 arm은 Continue(C0), 난수만 바꾼 Continue(C1), Refresh(F), Retreat–Retry(R)이다.

- 복원 audit는 이 host에서 새로 통과했다. 18개 checkpoint에서 비교한 모든 항목의 차이가 0.0이다.
- 계획한 24개 장면(seed 10024~10047)을 모두 실행했다. snapshot 84개, block 336개, continuation 1,344회다. 상한(96 / 384 / 1,536)보다 적은 이유는 trunk가 먼저 성공해 checkpoint가 없어진 12곳뿐이다.
- C1은 의도대로 동작했다. 336개 block 모두에서 남은 queue를 다 쓸 때까지 C0와 명령·상태·관측이 같았고(차이 0.0), 첫 추론은 같은 step에서 다른 seed로 일어났다.
- 재현 검사는 전부 일치했다: trunk 대 C0 84회, C1 재실행 6회, C0 재실행 6회.
- 성공 block 수(336개 중): C0 93, C1 98, F 94, R 39.
- **Refresh**: C0 기준 0→1 21건, 1→0 20건. C1 기준 18건, 22건. 장면 평균 차이는 −0.012(C0 기준), −0.026(C1 기준)이다.
- **Retreat–Retry**: C0 기준 0→1 15건, 1→0 69건. C1 기준 14건, 73건. 장면 평균 차이는 −0.235, −0.249이다.
- **C1 대 C0**(난수만 다름): 0→1 18건, 1→0 13건. 장면 평균 차이 +0.014.
- Retreat만 성공한 block(C0 실패, F 실패, R 성공)은 12개, 5개 장면이다. 반대로 C0와 F는 성공하고 R만 실패한 block은 53개, 11개 장면이다.
- 비용은 credit 차이로 약 $0.30이다(Stage 2B 상한 $1.50).

## 실제 수행한 명령

로컬(대여 전):

| 명령 | 종료 코드 | 로그 경로 | 시작/종료 시각 |
|---|---:|---|---|
| `python -m pytest -q` | 0 (128 passed, 3 skipped) | 터미널 | 2026-11-14 |
| `python scripts/20_stage2b_pilot.py plan --out runs/stage2b_001` | 0 | `runs/stage2b_001/plan.json` | 2026-11-14 19:53:50 UTC |

서버(전체 기록: `results/stage2b/commands_stage2b_001.status`):

| 명령 | 종료 코드 | 로그 경로 | 시작/종료 시각 (UTC) |
|---|---:|---|---|
| `scripts/00_system_packages.sh`, `scripts/01_bootstrap_vast.sh` | 0, 0 | `logs/s2b_install_sys.log`, `logs/s2b_bootstrap.log` | 19:56:41 / 19:59:27 |
| `python -m hci doctor --out runs/stage2b_001/doctor` | 0 | `logs/s2b_doctor.log` | 19:59:27 / 19:59:31 |
| `python -m pytest -q` | 0 (131 passed) | `logs/s2b_pytest.log` | 19:59:31 / 19:59:48 |
| `20_stage2b_pilot.py scenes --plan … --keys … --out …/scenes.json` | 0 | `logs/s2b_scenes.log` | 19:59:48 / 20:00:32 |
| `20_stage2b_pilot.py audit --plan … --out …/audit` | 0 | `logs/s2b_audit.log` | 20:00:32 / 20:09:30 |
| `20_stage2b_pilot.py collect … --start 0 --stop 2 --label first2` | 0 | `logs/s2b_collect_first2.log` | 20:09:30 / 20:22:38 |
| `20_stage2b_pilot.py techcheck …` | 0 | `logs/s2b_techcheck.log` | 20:22:38 / 20:22:39 |
| `20_stage2b_pilot.py scale … --usd-per-hour 0.26 --spent-usd 0.121 --workers 6` | 0 | `logs/s2b_scale.log` | 20:22:56 / 20:22:57 |
| `timeout 7800 20_stage2b_pilot.py collect … --start {2..7} --stop 24 --step 6` (worker 6개) | 모두 0 | `logs/s2b_collect_w{0..5}.log` | 20:22:57 / 21:03:35 |
| `20_stage2b_pilot.py finalize … --results …/results` | 0 | `logs/s2b_finalize.log` | 21:03:35 / 21:03:37 |
| `scripts/stage2b/run.sh pack` | 0 | `logs/s2b_pack.out` | 21:03:37 / 21:03 (묶음 파일의 시각; 종료 시각은 따로 기록되지 않음) |

실패한 실험 명령은 없다. 첫 업로드에서 연결이 끊겨 파일이 일부만 올라갔고, 그 상태에서 시작 명령이 한 번 나갔지만 실행 스크립트가 서버에 없어 아무것도 실행되지 않았다. 다시 올려 SHA256을 대조한 뒤 시작했다.

## 산출물

원본은 `runs/stage2b_001/`(git 제외)와 `data_backup/stage2b_001/stage2b_out.tgz`(git 제외)에 있고, 표·기록·영상은 `results/stage2b/`에 복사했다.

| 파일 | 용도 | SHA256 (앞 16자) |
|---|---|---|
| `results/stage2b/plan.json` | 실행 전 고정: 정책·설정·코드 hash, 장면 목록, arm과 난수 키, 상한 | `9bbe7aa53df47281` |
| `results/stage2b/reference_scene_keys.json` | 이미 기록된 초기 배치(시연 1000 + Stage 2A 24) | `5aa107fb1e4ab019` |
| `results/stage2b/scenes.json` | 새 장면의 초기 배치 대조 결과 | `231fda6f6b5c47ab` |
| `results/stage2b/restore_audit.json` | 복원 audit | `006d4b234883ee12` |
| `results/stage2b/techcheck.json` | 처음 2개 장면의 기술 확인(20개 항목) | `c06147762921a24d` |
| `results/stage2b/scale_decision.json` | 규모 결정 | `f375592a36715b55` |
| `results/stage2b/dataset.json` | 완료 상태와 문제 목록 | `9da08c8b45df162f` |
| `results/stage2b/scenes/*.json` (24개) | 장면별 원본 기록(네 arm, 추론 호출 기록 포함) | 백업 목록 참조 |
| `results/stage2b/pair_outcomes_four_arm.csv` | block당 한 행: 네 결과와 여섯 쌍의 분류 | `5def797cc9d7c4b3` |
| `results/stage2b/pair_summary.csv` | 쌍별·시점별 건수와 평균 차이 | `96cc96304f2b6ce1` |
| `results/stage2b/scene_equal_weight_summary.csv` | 장면별 평균과 차이 | `3d3a6e99a0f07a6e` |
| `results/stage2b/refresh_retreat_complementarity.csv` | F/R 보완성 표 | `d97c8041b0a153f9` |
| `results/stage2b/inference_step_seed_trace.csv` | 추론 호출별 step·난수 키·seed·chunk hash | `7171b866883d844c` |
| `results/stage2b/c0_c1_queue_prefix_parity.csv` | block별 C0/C1 queue 구간 일치 결과 | `a0c4307a3aaf22d9` |
| `results/stage2b/observation_fields.json` | 온라인 입력과 진단 전용 항목의 구분 | `67cd36ad488c065f` |
| `results/stage2b/skip_reasons.csv` | 실행하지 않은 checkpoint와 사유 | `5fb6dbf4f66c6f9b` |
| `results/stage2b/summary.json` | 집계 전체 | `8f6b3b6ff854fa2c` |
| `results/stage2b/videos/*.mp4` (8개) | 처음 2개 장면, 첫 checkpoint, r=0의 네 arm | 백업 목록 참조 |
| `results/stage2b/backup_sha256_stage2b_001.txt` | 서버 묶음의 파일별 SHA256(129개) | — |
| `runs/stage2b_001/branches/inputs/*.npz` (24개, git 제외) | snapshot별 온라인 입력 | 백업 목록 참조 |
| `runs/stage2b_001/branches/traces/*.npz` (24개, git 제외) | step별 진단 기록 | 백업 목록 참조 |

서버 묶음 `stage2b_out.tgz`의 SHA256은 `7204cbebad51f5f38c59bfa1f4dd39a38563a76501e4b918582723b0f57511c1`이다. 서버와 로컬에서 같았고, 풀어서 129개 파일을 모두 대조했다(129/129 일치).

## 검증

| 항목 | PASS / FAIL / NOT RUN | 실제 근거 |
|---|---|---|
| CPU tests | PASS | 로컬 128 passed, 3 skipped(imageio·mani_skill·gymnasium 없음). 서버 131 passed. Stage 2B 테스트는 20개(mock 물리) |
| CUDA | PASS | `hci doctor` exit 0, RTX 3090, driver 580.65.06 |
| Vulkan/RGB | PASS | `hci doctor` exit 0, 영상 8개 기록 |
| 정책 rollout | PASS (실행됨) | trunk 24회, 분기 1,344회. 성능 주장은 아님 |
| 상태+정책 복원 | PASS | audit 18개 checkpoint 전 항목 차이 0.0. 수집 중 모든 분기의 시작 상태 대조 통과 |
| C1이 난수 키만 바꾸는지 | PASS | 336/336 block에서 queue 구간 일치(차이 0.0), 추론 seed 전부 선언한 키와 일치 |
| 개입 행동 영상 | PASS | 8개(장면 10024, 10025 × 네 arm) |
| seed split | PASS | train에서만 선택, 다른 용도의 seed·Stage 2A 장면과 겹침 없음, 가장 가까운 기존 배치와 1.7 cm 차이 |
| 데이터 완료 상태 | PASS | `dataset.json` completed=true, problems=[] |
| 학습 | NOT RUN | 이 단계에 학습 없음 |
| 과학적 가설 검증 | NOT RUN | 기술통계만. 구간·p-value 없음 |

### 복원 audit

audit 장면 5개(seed 100~104, 모두 시연 장면과 겹침, 성능 분모로 쓰지 않음)에서 18개 checkpoint에 도달했다. 각 checkpoint에서 Continue 20 step을 기준으로 복원 후 반복, Retreat 분기 뒤 복원, Refresh 분기 뒤 복원, **C1 분기 뒤 복원**을 비교했다(각 18회).

- simulator 상태, 정책 history(관절·TCP·RGB), 남은 queue, 행동 기록, 환경이 돌려준 관측의 최대 차이: 모두 0.0.
- 카운터, flag, 개입·Retreat 기록, 난수 상태: 모두 일치.
- 정책 입력을 저장하는 함수가 실행 상태를 바꾸지 않는지: 18/18 완전 일치.

### C1이 정말 난수만 바꿨는가

| 검사 | 결과 |
|---|---|
| 남은 queue(모든 snapshot에서 4개)를 실행하는 동안 C0와 C1의 명령·simulator 상태·환경 관측 | 336/336 일치, 최대 차이 0.0 |
| 그 구간에서 추론 호출 | 0회 |
| 첫 추론 step이 C0와 C1에서 같고 `t + 4`인지 | 336/336 |
| 첫 추론의 seed가 C0와 C1에서 다른지 | 336/336 |
| queue를 실행하는 중에 끝난 block | 0개 |
| 모든 추론 호출의 seed가 `seed_for(scene, step, 난수 키, 'dp')`와 같은지 | 전부 일치(finalize에서 대조) |
| C1의 개입 기록 | 없음(336/336) |
| 대조 arm이 block 수를 바꿨는지 | 아니오: block 336 = snapshot 84 × repeat 4 |

### 재현 검사

- repeat 0의 C0가 복원하지 않은 trunk와 같은지: 84/84 일치.
- 처음 2개 장면에서 다른 분기를 모두 실행한 뒤 다시 실행: r=0의 C1 6/6, r=1의 C0 6/6 일치(행동, 결과, 최종 상태, 추론 step·seed·chunk).
  계획 상한은 16회였고, 장면 10024의 snapshot이 2개여서 12회다.

### 규모 결정

처음 2개 장면에서 continuation당 7.1초가 측정됐다(재실행과 trunk 포함 110회, 780초). 24개 장면의 예상 총액이 $0.59로 상한 안이어서 24를 골랐다. 입력은 시간과 비용뿐이다.

## 관측 결과

원본 episode 수: 장면 24개(trunk 24회). trunk 성공 8, 실패 16.
분기 실행 수: snapshot 84, block 336, continuation 1,344(+ 재현 검사 12회).
시간/저장량: 수집 약 54분(worker 6개), 장면 실행 시간 합 3.9시간. 서버 묶음 20.6 MB.

### 실행량과 빠진 칸

| checkpoint | 장면 | block | 빠진 장면 |
|---:|---:|---:|---|
| 12 | 24 | 96 | 0 |
| 52 | 24 | 96 | 0 |
| 92 | 20 | 80 | 4 (trunk가 먼저 성공) |
| 132 | 16 | 64 | 8 (trunk가 먼저 성공) |

빠진 12개 checkpoint는 모두 trunk가 그 전에 성공한 경우다. 새 장면으로 채우지 않았다.

### arm별 성공 block

| checkpoint | block | C0 | C1 | F | R |
|---:|---:|---:|---:|---:|---:|
| 12 | 96 | 37 | 40 | 41 | 15 |
| 52 | 96 | 36 | 40 | 34 | 9 |
| 92 | 80 | 18 | 17 | 19 | 11 |
| 132 | 64 | 2 | 1 | 0 | 4 |
| 전체 | 336 | 93 | 98 | 94 | 39 |

### 여섯 쌍 (block 336개)

"0→1"은 기준 arm이 실패한 block에서 비교 arm이 성공한 것, "1→0"은 그 반대다. 비율의 분모는 기준 arm의 실패 block 수와 성공 block 수다.

| 비교 − 기준 | 0→1 | 1→0 | 1→1 | 0→0 | 0→1 비율 | 1→0 비율 | 평균 차이 | 0→1 장면 | 1→0 장면 |
|---|---:|---:|---:|---:|---|---|---:|---:|---:|
| F − C0 | 21 | 20 | 73 | 222 | 21/243 = 8.6% | 20/93 = 21.5% | +0.003 | 10 | 9 |
| R − C0 | 15 | 69 | 24 | 228 | 15/243 = 6.2% | 69/93 = 74.2% | −0.161 | 5 | 13 |
| C1 − C0 | 18 | 13 | 80 | 225 | 18/243 = 7.4% | 13/93 = 14.0% | +0.015 | 7 | 7 |
| F − C1 | 18 | 22 | 76 | 220 | 18/238 = 7.6% | 22/98 = 22.4% | −0.012 | 10 | 9 |
| R − C1 | 14 | 73 | 25 | 224 | 14/238 = 5.9% | 73/98 = 74.5% | −0.176 | 4 | 13 |
| R − F | 16 | 71 | 23 | 226 | 16/242 = 6.6% | 71/94 = 75.5% | −0.164 | 6 | 16 |

- C1 − C0의 0→1과 1→0은 **noise outcome switch**다. 같은 Continue 규칙이 다른 난수 실현에서 낸 결과 차이이고, Rescue나 Harm이 아니다.
- `(F − C0) − (C1 − C0)`는 대수적으로 `F − C1`과 같다(+0.003 − 0.015 = −0.012). 난수를 보정한 순수 효과가 아니다.
- 네 arm의 결과가 모두 같은 block은 223개다(336개 중).

시점별 평균 차이(block 기준):

| checkpoint | F − C0 | F − C1 | R − C0 | R − C1 | C1 − C0 | R − F |
|---:|---:|---:|---:|---:|---:|---:|
| 12 | +0.042 | +0.010 | −0.229 | −0.260 | +0.031 | −0.271 |
| 52 | −0.021 | −0.062 | −0.281 | −0.323 | +0.042 | −0.260 |
| 92 | +0.013 | +0.025 | −0.087 | −0.075 | −0.013 | −0.100 |
| 132 | −0.031 | −0.016 | +0.031 | +0.047 | −0.016 | +0.062 |

step 132에서는 C0가 성공한 block이 2개, C1이 1개뿐이다. 그 시점의 1→0 비율은 분모가 1~2라서 읽을 수 없고, R − F의 1→0 비율은 분모가 0이어서 NA다.

### 장면 동일 가중 (주된 요약)

장면 안에서 도달한 checkpoint와 repeat의 평균을 낸 뒤 24개 장면의 평균을 냈다. 유효 checkpoint가 없는 장면은 없었다.

| 차이 | 장면 평균 | 중앙값 | 최소 | 최대 | 양(+) 장면 | 음(−) 장면 | 0인 장면 |
|---|---:|---:|---:|---:|---:|---:|---:|
| F − C0 | −0.012 | 0.000 | −0.500 | +0.188 | 6 | 6 | 12 |
| F − C1 | −0.026 | 0.000 | −0.500 | +0.188 | 5 | 7 | 12 |
| R − C0 | −0.235 | 0.000 | −1.000 | +0.375 | 4 | 11 | 9 |
| R − C1 | −0.249 | 0.000 | −1.000 | +0.188 | 3 | 11 | 10 |
| C1 − C0 | +0.014 | 0.000 | −0.062 | +0.188 | 4 | 1 | 19 |
| R − F | −0.223 | −0.062 | −1.000 | +0.188 | 2 | 15 | 7 |

arm별 장면 평균 성공 비율: C0 0.362, C1 0.376, F 0.350, R 0.127.

C0의 결과가 repeat에 따라 갈린 snapshot은 84개 중 13개다(Stage 2A는 91개 중 14개).

### Retreat의 보완 가치

C0 기준(괄호는 같은 snapshot에서 그 패턴이 나타난 repeat 수별 snapshot 수: 1개/2개/3개/4개):

| 패턴 | block | 장면 | snapshot | repeat 수별 |
|---|---:|---:|---:|---|
| C0 실패, F 실패, **R 성공** | 12 | 5 | 8 | 6 / 1 / 0 / 1 |
| C0 실패, **F 성공**, R 실패 | 18 | 10 | 14 | 10 / 4 / 0 / 0 |
| C0 실패, F 성공, R 성공 | 3 | 1 | 3 | 3 / 0 / 0 / 0 |
| C0 실패, 셋 다 실패 | 210 | 17 | 60 | 5 / 5 / 5 / 45 |
| C0 성공, F 성공, **R 실패** | 53 | 11 | 21 | 5 / 7 / 2 / 7 |
| C0 성공, **F 실패**, R 성공 | 4 | 2 | 3 | 2 / 1 / 0 / 0 |
| C0 성공, F 실패, R 실패 | 16 | 7 | 11 | 6 / 5 / 0 / 0 |
| C0 성공, 셋 다 성공 | 20 | 9 | 10 | 4 / 3 / 2 / 1 |

F와 R만의 2×2: 둘 다 실패 226, R만 성공 16(6개 장면), F만 성공 71(16개 장면), 둘 다 성공 23.

기준을 C1로 바꾸면: C1 실패·F 실패·R 성공 11 block(4개 장면), C1 성공·F 성공·R 실패 56 block(12개 장면). 방향은 같다.

- R만 성공한 12개 block 중 6개는 두 snapshot에 모여 있다: 장면 10035의 step 92(repeat 4개 모두)와 장면 10043의 step 132(2개). 나머지 6개는 snapshot당 repeat 하나씩이다.
- 비교 기준으로, 난수만 바꾼 C1이 C0의 실패를 성공으로 바꾼 block이 18개(7개 장면)다. R만 성공한 12개(5개 장면)는 이보다 많지 않다.
- 네 arm 중 R만 성공한 block은 8개(4개 장면)다.

### 시간과 추론 횟수

| arm | checkpoint 이후 step 평균 | 성공한 block의 step 평균 | 추론 호출 평균 |
|---|---:|---:|---:|
| C0 | 107.8 | 57.5 (93개) | 13.1 |
| C1 | 107.3 | 62.9 (98개) | 13.0 |
| F | 108.2 | 62.9 (94개) | 14.0 |
| R | 129.3 | 97.2 (39개) | 13.6 |

Retreat는 336회 모두 목표에 도달했다. 23~30 step(평균 25.4)을 썼고 실제 변위는 4.5~4.7 cm, 목표 오차는 3.1~5.0 mm다. 이 step은 같은 200-step 예산에서 빠진다.

### checkpoint 시점의 상태별 (관찰 라벨, 원인 아님)

단계 라벨은 simulator의 정답 상태로 만든 진단용이다. 정책 입력이 아니고, 그대로 배포할 수 있는 판단 기준도 아니다.

| checkpoint까지 도달한 단계 | block | 장면 | C0 | C1 | F | R |
|---|---:|---:|---:|---:|---:|---:|
| 큐브에 접근 전 | 188 | 24 | 37 | 40 | 41 | 15 |
| 접근했으나 파지 전 | 16 | 2 | 1 | 1 | 0 | 1 |
| 파지, 들어올리기 전 | 48 | 12 | 27 | 28 | 24 | 2 |
| 들어올림, 정렬 전 | 60 | 13 | 20 | 23 | 23 | 13 |
| 정렬, 성공 전 | 24 | 5 | 8 | 6 | 6 | 8 |

### trunk 결과별 (결과를 본 뒤 나눈 표)

장면을 trunk의 성공 여부로 나눈 것은 사전에 정한 분석이 아니다. 참고로만 적는다.

| | 장면 | block | C0 | C1 | F | R | R − C0의 0→1 / 1→0 | F − C0의 0→1 / 1→0 | C1 − C0의 0→1 / 1→0 |
|---|---:|---:|---:|---:|---:|---:|---|---|---|
| trunk 성공 | 8 | 80 | 76 | 77 | 69 | 19 | 0 / 57 | 3 / 10 | 3 / 2 |
| trunk 실패 | 16 | 256 | 17 | 21 | 25 | 20 | 15 / 12 | 18 / 10 | 15 / 11 |

- R의 1→0 69건 중 57건이 trunk가 성공한 8개 장면에서 나왔다. 이 8개 장면 모두에서 R − C0가 음수다(−0.33 ~ −1.00).
- trunk가 실패한 16개 장면에서는 세 비교(F, R, C1)의 0→1과 1→0 건수가 서로 비슷한 크기다.

### 영상

`results/stage2b/videos/`: 장면 10024와 10025의 step 12, r=0에서 네 arm. 결과를 보고 고른 예시는 없다.

## 다섯 질문에 대한 답

**1. C1이 정말 queue와 추론 시점은 유지하고 미래 정책 난수만 바꿨는가?**
그렇다. 336개 block 전부에서 남은 queue를 실행하는 동안 C0와 차이가 0.0이었고, 첫 추론은 같은 step에서 다른 seed로 일어났다. 재실행 12회와 audit의 C1 분기 18회도 일치했다.

**2. C0와 C1의 결과 변동과 평균 차이는 장면별로 어느 정도인가?**
336개 block 중 31개(7개 장면)에서 결과가 바뀌었다(0→1 18, 1→0 13). 평균 차이는 block 기준 +0.015, 장면 기준 +0.014이고, 19개 장면에서는 차이가 0이다. 장면별 범위는 −0.062 ~ +0.188이다.

**3. F와 R의 평균 이득 방향이 두 기준에서 유지되는가?**
- Refresh는 이득이 관측되지 않았다. C0 기준 +0.003, C1 기준 −0.012(block), 장면 기준 −0.012와 −0.026이다. 양인 장면과 음인 장면 수가 비슷하다(6/6, 5/7). 참고로 C1 − C0의 차이는 +0.015다.
- Retreat–Retry는 두 기준 모두에서 음수다(block −0.161, −0.176, 장면 −0.235, −0.249). 음인 장면이 11개, 양인 장면이 3~4개다.
- Stage 2A에서는 같은 정책으로 Refresh +20건, Retreat +7건이었다. 다른 장면이고 trunk 성공 수도 다르다(2A 4/24, 2B 8/24). 두 결과를 합치지 않으며, 2A의 방향이 새 장면에서 다시 나오지 않았다고만 적는다.

**4. R이 F로 못 푸는 상태를 여러 장면에서 추가로 푸는가? 추가 악화와 비용은 무엇인가?**
- R만 성공한 block은 12개(5개 장면)이고, repeat 여러 개에서 반복된 snapshot은 2개다. 난수만 바꿔도 생기는 0→1(18개, 7개 장면)보다 많지 않다.
- 반대쪽은 크다. C0와 F가 성공한 block 73개 중 53개에서 R이 실패했다(11개 장면, 그중 7개 snapshot은 repeat 4개 모두).
- 비용: Retreat가 23~30 step을 쓰고, 성공하더라도 checkpoint 이후 평균 97 step이 걸렸다(C0 57 step).

**5. 다음은 Refresh-only 선택기, 다중 개입 선택기, 복구 행동 수정 중 무엇인가?**
명세 8절의 3번, **"복구 행동 개선 또는 현재 주장 보류"**에 해당한다고 판단한다.
- 1번(Refresh 중심 선택기)의 전제인 "두 기준 모두에서 F의 평균 이득"이 이 표본에서 관측되지 않았다.
- 2번(다중 개입 선택기)의 전제인 "R만 성공시키는 상태가 여러 장면에"는 5개 장면에서 나왔지만 대부분 repeat 하나짜리이고, 난수만 바꾼 Continue의 0→1 건수(18개, 7개 장면)보다 많지 않다.
- 4번(도구 수정)은 해당하지 않는다. 재현·정합성 검사는 모두 통과했다.

이것은 연구 진행 판단이고 통계적 유의성 판정이 아니다. 판단에 쓴 수치는 위 표에 그대로 있다.

## 해석의 범위

- 독립 표본은 장면 24개다. 한 장면의 block은 과거를 공유한다. 구간과 p-value를 계산하지 않았다.
- checkpoint에서 선택을 **강제**한 결과다. 필요할 때만 개입하는 온라인 정책의 성능이나 악화율 `H(g)`가 아니다. Retreat의 1→0이 많다는 것은 "성공하고 있던 실행에 고정 시점으로 Retreat를 넣으면 대부분 실패했다"는 뜻이지, Retreat를 쓰는 선택기가 해롭다는 뜻이 아니다.
- 네 번째 arm은 원래 재계획 시점의 재표본 변동에 대한 대조다. Refresh의 구성 요소(새 관측, chunk 폐기, 이른 추론, 다른 난수)를 분리하지 못한다.
- 서로 다른 seed는 다른 의사난수 실현일 뿐, 통계적 독립의 증명이 아니다.
- 단계 라벨과 trunk 결과별 표는 관찰이고 원인이 아니다.
- 이 simulator, 이 정책(qualification 37/100, 기준 미달), 이 Retreat 설정(위로 5 cm)에서의 결과다.
- Stage 2A와 2B는 따로 읽는다. 합쳐서 확증 표본으로 쓰지 않는다.

## 변경과 실패

변경한 파일과 이유:

- `src/hci/pilot2b.py` (신규): 네 arm 수집, C0/C1 queue 구간 일치 검사, audit의 C1 분기와 입력 저장 검사, 표. `pilot.py`의 함수를 재사용한다.
- `scripts/20_stage2b_pilot.py`, `scripts/stage2b/run.sh` (신규): plan에 묶인 실행기.
- `src/hci/policies.py`: `DiffusionPolicy.generate`가 방금 쓴 seed를 `last_seed`에 남긴다(기록용 한 줄). seed 값과 추론 경로는 그대로다.
- `tests/test_stage2b_pilot.py` (신규, 20개): 네 arm의 같은 출발점, C1의 queue·history 무변경, 두 인덱스의 분리, 잘못 구현한 C1 네 종류의 검출, 재실행 일치, 실행 순서 무관, 분기 간 누출 검출, queue 실행 중 종료 허용, block 키로 조인, C1을 개입·block으로 세지 않음, 일부 실패 시 미완료, 관측 복원 오류 검출, 완료 판정을 기록에서 다시 계산 등.
- `amendments/AMENDMENT_006_stage2b_noise_control.md`, `README.md`, `.gitattributes`.
- `pilot.py`, `simulation.py`, 기존 collect/audit, Stage 2A 실행기는 수정하지 않았다. `policies.py`가 바뀌어 Stage 2A plan의 코드 hash와는 더 이상 일치하지 않는다(Stage 2A 재실행에는 커밋 `d82168b`가 필요하다).

Codex 검토에서 3건을 받아 실행 전에 모두 반영했다: 규모 결정 파일이 없으면 완료로 표시하지 않음, 재현 검사의 개수·대상·일치를 기록에서 다시 확인(저장된 flag를 믿지 않음), finalize에서 trace 파일과 영상의 존재 확인.

실패한 명령 및 원인: 실험 명령 실패 없음. 로컬 전체 테스트를 Codex 실행과 동시에 돌린 한 번은 103 passed, 6 skipped로 나왔다(모듈 3개가 통째로 skip). 단독으로 다시 실행하면 128 passed, 3 skipped였고 서버에서는 131개 모두 통과했다. skip의 원인은 확인하지 못했다.

수행하지 않은 일: 기본 정책 재학습, 체크포인트 재선택, qualification 재실행, Retreat 변경, 추론 설정 변경, Q/Rescue/Harm 예측기 학습, 온라인 선택기 평가, LTT/calibration, 최종 Test, 5,760회 수집. `RUN_FROZEN_TEST` 토큰은 만들지 않았다. 저장한 온라인 입력으로 어떤 모델도 학습하지 않았다.

## 비용

| 항목 | 값 |
|---|---|
| 대여 전 credit | $8.5101 |
| 삭제 후 credit | $8.2070 |
| Stage 2B 사용 | **약 $0.30** (상한 $1.50) |
| 표시 단가 × 가동 시간 | $0.2452/h × 1시간 13분 ≈ $0.30 |
| 업로드 / 다운로드 | 23 MB / 20.6 MB |

- Stage 2A($0.38), Stage 1D(누적 약 $1.14)와 따로 적는다.
- 유휴 대기는 없었다. 수집 종료 → 묶음 → 다운로드 → 129/129 hash 대조 → 삭제를 이어서 처리했다. 삭제 후 `No instances found`를 확인했다.

## 다음 단계

필요 작업: 한 가지로 좁혀 제안한다. **GPU 없이, 이미 저장된 기록으로 Retreat가 성공을 깨뜨린 53개 block을 살펴본다.** trunk가 성공한 8개 장면에서 Retreat 뒤에 무슨 일이 생기는지(큐브를 놓치는지, 다시 접근하지 못하는지, 시간이 모자란지)를 step별 trace와 영상으로 확인하는 일이다. 그 결과로 Retreat 정의를 고칠지, 현재 주장을 보류할지를 정할 수 있다. 지금 자료로는 선택기를 학습할 근거가 약하다: Refresh의 평균 차이는 두 기준 모두에서 0.03 이내였고, Retreat만 성공한 block(12개)은 난수만 바꾼 Continue가 실패를 성공으로 바꾼 block(18개)보다 많지 않았다. 두 건수의 비교는 참고이고, 난수 효과를 뺀 값이 아니다.

예상 계산량의 산출 근거: 저장된 trace 24개와 장면 기록만 읽는다. GPU 0시간, $0. 영상이 더 필요하면 그때 장면 수 개의 재실행 비용을 따로 보고한다.

동결/승인이 필요한 사항:

- 위 분석을 할지, 아니면 다른 방향(Retreat 재설계, 기본 정책 보강, 주장 보류)으로 갈지는 사용자 결정이다.
- Retreat 설정을 바꾸면 새 amendment가 필요하고, 지금까지의 Stage 2A·2B 결과는 이전 설정의 개발 자료로 남는다.
- Stage 2A·2B의 48개 장면은 결과를 본 개발 표본이다. train에는 72개가 남아 있다.
- 추가 실행은 자동으로 시작하지 않는다.

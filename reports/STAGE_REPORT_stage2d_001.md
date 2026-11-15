# Stage 보고서 — Stage 2D 반환 시점을 맞춘 0 / 2 / 5 cm 대조

D500-clean 80k 정책을 그대로 두고 새 train 장면 12개에서 네 arm의 분기 704회를 끝까지 수집했다. 이 표본에서 세 대조 arm(5 cm 이동, 2 cm 이동 후 유지, 제자리 유지)의 성공 수는 176개 block 중 32 / 26 / 29로 서로 가까웠고, 차이의 방향은 checkpoint와 장면에 따라 반대로 갈렸다. Stage 2B에서 보인 "5 cm Retreat가 Continue보다 낮다"는 방향은 이 표본 전체에서는 다시 나오지 않았다. 개발용 기술통계이고, 어떤 arm의 효용이나 원인을 입증하지 않는다.

실행 ID: `stage2d_001`
단계: Stage 2D — 고정 정책의 네 arm 대조 (복원 audit + 분기 수집)
작업/정책 seed: StackCube-v1 / 정책 `stack_1d_s1_d500_clean_v1_i080000` (학습 seed 1, 재학습 없음)
실제 실행 환경: Vast.ai 인스턴스 60710880 (RTX 3090 24 GB, driver 595.99.02, Core i9-11900K, 단가 $0.2207/h), 2026-11-14 23:56 ~ 2026-11-15 00:52 UTC. 삭제 완료.
코드/upstream 버전: 저장소 `7bb6b4d` + 이 단계의 새 파일(실행 시점에는 미커밋, 파일별 SHA256을 plan에 고정) / ManiSkill `a4a4f9272ad64b1564035874b605ceb687b63ed8`
변경 기록: `amendments/AMENDMENT_008_stage2d_duration_matched_controls.md`

## 상태

```text
base_policy_qualification: BELOW_ORIGINAL_GATE (37/100 < 50/100)
stage2d: EXPLORATORY_DURATION_MATCHED_CONTROL (completed)
observation_based_selector: NOT_TRAINED
formal_study_ready: NOT_ESTABLISHED
calibration: NOT_RUN
final_test: NOT_RUN
```

qualification 판정은 그대로다. 이 실험은 Stage 2B의 결과와 Stage 2C의 사후 분석·보충 사례까지 본 뒤에 정한 설계이고, 확증 실험도 독립 검증도 아니다.

## 요약

네 arm은 Continue(C0), 기존 5 cm Retreat–Retry(R5), 2 cm 이동 후 유지(R2M), 제자리 유지(H0M)이다. R2M과 H0M은 R5가 그 snapshot에서 쓴 step 수(T_ref)만큼 제어한 뒤 정책에 제어를 돌려준다.

- 복원 audit는 이 host에서 새로 통과했다. 18개 checkpoint에서 새 arm을 포함해 비교한 모든 항목의 차이가 0.0이다.
- 계획한 12개 장면(seed 10048~10059)을 모두 실행했다. snapshot 44개, block 176개, continuation 704회다. 상한(48 / 192 / 768)보다 적은 이유는 trunk가 먼저 성공해 checkpoint가 없어진 4곳뿐이다.
- 반환 시점은 의도대로 맞았다. 176개 block 모두에서 세 대조 arm이 checkpoint + T_ref에서 같은 seed로 첫 추론을 했고, 제어 구간의 추론은 0회였다. T_ref는 23~27 step(평균 25.0)이다.
- R5의 제어 구간은 176회 모두 참조 실행과 같았다. 재현 검사도 전부 일치했다(trunk 대 C0 44회, R2M 재실행 8회, H0M 재실행 8회).
- 성공 block 수(176개 중): C0 25, R5 32, R2M 26, H0M 29.
- **R2M 대 R5**: 0→1 18건, 1→0 24건. 장면 평균 차이 −0.031.
- **H0M 대 R5**: 0→1 23건, 1→0 26건. 장면 평균 차이 −0.030.
- **R2M 대 H0M**: 0→1 9건, 1→0 12건. 장면 평균 차이 −0.002.
- checkpoint별로 방향이 갈렸다. step 12에서는 R2M·H0M이 R5보다 많이 성공했고(17·18 대 9), step 52에서는 R5가 더 많이 성공했다(18 대 5·7).
- C0가 성공한 25개 block 가운데 R5는 18개, R2M은 14개, H0M은 13개에서 실패했다. 제자리 유지를 명령한 H0M에서도 절반 이상이 실패로 바뀌었다.
- 비용은 credit 차이로 약 $0.21이다(Stage 2D 상한 $1.00).

## 실제 수행한 명령

로컬(대여 전):

| 명령 | 종료 코드 | 로그 경로 | 시작/종료 시각 |
|---|---:|---|---|
| `python -m pytest -q` | 0 (178 passed, 3 skipped) | 터미널 | 2026-11-14 |
| `codex exec … (실행 전 코드 검토)` | 0 (결함 보고 없음) | 세션 기록 | 2026-11-14 |
| `python scripts/23_stage2d_pilot.py plan --out runs/stage2d_001` | 0 | `runs/stage2d_001/plan.json` | 2026-11-14 23:55:18 UTC |

서버(전체 기록: `results/stage2d/commands_stage2d_001.status`):

| 명령 | 종료 코드 | 로그 경로 | 시작/종료 시각 (UTC) |
|---|---:|---|---|
| `scripts/00_system_packages.sh`, `scripts/01_bootstrap_vast.sh` | 0, 0 | `logs/s2d_install_sys.log`, `logs/s2d_bootstrap.log` | 23:58:13 / 00:01:56 |
| `python -m hci doctor --out runs/stage2d_001/doctor` | 0 | `logs/s2d_doctor.log` | 00:01:56 / 00:02:00 |
| `python -m pytest -q` | 0 (181 passed) | `logs/s2d_pytest.log` | 00:02:00 / 00:02:24 |
| `23_stage2d_pilot.py scenes --plan … --keys … --out …/scenes.json` | 0 | `logs/s2d_scenes.log` | 00:02:24 / 00:03:00 |
| `23_stage2d_pilot.py audit --plan … --out …/audit` | 0 | `logs/s2d_audit.log` | 00:03:00 / 00:13:57 |
| `23_stage2d_pilot.py collect … --start 0 --stop 2 --label first2` | 0 | `logs/s2d_collect_first2.log` | 00:13:57 / 00:29:48 |
| `23_stage2d_pilot.py techcheck …` | 0 | `logs/s2d_techcheck.log` | 00:29:48 / 00:29:49 |
| `23_stage2d_pilot.py scale … --usd-per-hour 0.23 --spent-usd 0.13 --workers 6` | 0 | `logs/s2d_scale.log` | 00:30:36 / 00:30:37 |
| `23_stage2d_pilot.py collect … --start 2+w --stop 12 --step 6` (worker 6개) | 모두 0 | `logs/s2d_collect_w0..5.log` | 00:30:37 / 00:51:01 |
| `23_stage2d_pilot.py finalize … --results …/results` | 0 | `logs/s2d_finalize.log` | 00:51:01 / 00:51:03 |
| `scripts/stage2d/run.sh pack` | 0 | `logs/s2d_pack.out` | 00:51 |

로컬(회수 후):

| 명령 | 종료 코드 | 비고 |
|---|---:|---|
| 백업 archive SHA256 대조, 풀어서 파일별 대조 | 0 | archive `98922522cd27099f…`, 95개 파일 모두 일치 |
| `vastai destroy instance 60710880` → `show instances` | 0 | `No instances found` (00:52 UTC) |
| `21_stage2c_offline_review.py verify --label input_hash_check_after_stage2d` | 0 | Stage 2B 입력 123개 파일 변화 없음 |

실패한 명령은 없다. 서버 pytest가 로컬보다 3개 많이 통과한 것은 로컬에서 건너뛰는 테스트가 서버에서는 실행되기 때문이다.

## 산출물

원본은 `runs/stage2d_001/`(Git 제외)와 `data_backup/stage2d_001/stage2d_out.tgz`에 있고, 아래는 저장소에 넣은 사본이다(SHA256 앞 16자리).

| 파일 | 용도 | SHA256 |
|---|---|---|
| `results/stage2d/plan.json` | 실행 전 고정한 정책·설정·코드 hash, arm 정의, T_ref 규칙, 장면, 상한 | `8989c723a5dc62c5` |
| `results/stage2d/scenes.json` | 새 장면의 초기 배치 대조(교체 0건) | `dab9e1066d12185c` |
| `results/stage2d/restore_audit.json` | 복원 audit 전체 기록 | `3820a4f46a775c3e` |
| `results/stage2d/techcheck.json` | 처음 2개 장면의 기술 검사 25항목 | `a950d0dd15eaeb2f` |
| `results/stage2d/scale_decision.json` | 시간·비용만으로 정한 규모(12개) | `63bd01468ff30b4c` |
| `results/stage2d/dataset.json` | 수집 완료 상태와 문제 목록(없음) | `26905530b22e8d2d` |
| `results/stage2d/summary.json` | 집계 전체 | `85babf64fa295224` |
| `results/stage2d/pair_outcomes.csv` | block별 네 결과, 여섯 쌍의 분류, 제어 구간 기록 | `745d7846bef6327d` |
| `results/stage2d/pair_summary.csv` | 쌍별·checkpoint별 네 범주 수 | `51509c7f0be92122` |
| `results/stage2d/scene_summary.csv` | 장면별 평균과 쌍별 건수 | `984b3ceb7e3a0e1c` |
| `results/stage2d/control_duration_reference.csv` | snapshot별 T_ref와 참조 실행 기록 | `d751a47c8e2636dd` |
| `results/stage2d/handoff_parity.csv` | block별 반환 step·seed·queue·history | `0d26db3a9b03330e` |
| `results/stage2d/event_rows.csv`, `event_summary.csv` | 분기별 사건 읽기와 요약 | `7a232db5cb804929`, `c233f32a341bd089` |
| `results/stage2d/inference_step_seed_trace.csv` | 추론 호출별 step·seed | `21e2e30b8e1ba2b9` |
| `results/stage2d/skip_reasons.csv` | 빠진 checkpoint 4곳과 사유 | `b772c0cad825b046` |
| `results/stage2d/observation_fields.json` | 온라인 입력과 진단 전용 항목의 구분 | `3c0861fd2693b039` |
| `results/stage2d/scenes/*.json` (12), `videos/*.mp4` (8) | 장면별 원본 기록, 사전 규칙으로 정한 영상 | `backup_sha256_stage2d_001.txt` 참조 |
| `src/hci/pilot2d.py` | R2M·H0M 제어, 수집기, 표 | `efefdc292630bb56` |
| `scripts/23_stage2d_pilot.py`, `scripts/stage2d/run.sh` | 실행기 | `34c47e42835f2819`, `70e8c4a688ddf68b` |
| `tests/test_stage2d_pilot.py` | 로컬 검사 21개 | `793a2abe090014b5` |

## 검증

| 항목 | PASS / FAIL / NOT RUN | 실제 근거 |
|---|---|---|
| CPU tests | PASS | 로컬 178 passed / 3 skipped, 서버 181 passed. mock 물리이며 조작 성능과 무관하다 |
| CUDA | PASS | `hci doctor` exit 0, RTX 3090에서 정책 추론 실행 |
| Vulkan/RGB | PASS | 수집 전체가 RGB 관측으로 실행됨. 영상 8개 기록 |
| 정책 rollout | PASS | trunk 12회, continuation 704회, 참조 44회, 재실행 16회가 오류 없이 끝남 |
| 상태+정책 복원 | PASS | audit 18개 checkpoint: simulator 상태·history·queue·행동 기록·RGB·난수 상태의 차이가 모두 0.0. R5 제어 전용 참조, R5, R2M, H0M 각각을 실행한 뒤 복원한 trace도 일치 |
| R5 = 참조 | PASS | 176회 모두 명령·상태·관측·step 수·기록이 참조와 같음(차이 0.0) |
| 반환 시점 일치 | PASS | 176개 block: 제어 구간 추론 0회, 반환 시 queue 0, 첫 추론이 checkpoint + T_ref에서 세 arm 같은 seed. 선언한 seed 규칙과도 일치 |
| 재현 검사 | PASS | repeat 0의 C0 대 trunk 44/44, R2M 재실행 8/8, H0M 재실행 8/8 |
| 개입 행동 영상 | PASS | 처음 2개 장면의 첫 checkpoint(step 12), r=0, 네 arm. 결과를 보고 고른 영상은 없다 |
| seed split | PASS | train에서 Stage 2A·2B와 다른 용도의 seed를 뺀 앞 12개. 저장된 좌표 1,048개와 audit·dev·policy 평가 집합에 같은 배치 없음(가장 가까운 것도 8.6 mm 차이). calibration·Test는 reset하지 않음 |
| 데이터 완료 상태 | PASS | `dataset.json`: completed, 문제 0건, 오류 파일 0개 |
| 이전 단계 입력 | PASS | Stage 2B 입력 123개 파일 hash 변화 없음. Stage 2B·2C 결과 파일에 쓴 것 없음 |
| 과학적 가설 | NOT RUN | 이 단계는 가설 검증이 아니다 |

## 관측 결과

원본 episode 수: 12 (trunk 성공 3, 실패 9)
분기 실행 수: 704 (snapshot 44 × repeat 4 × arm 4). 별도로 T_ref 참조 44회, 재현 검사 16회
성공/실패 수(176개 block): C0 25 / 151, R5 32 / 144, R2M 26 / 150, H0M 29 / 147
Rescue/Harm 분자와 분모: 아래 표의 0→1, 1→0. 기준 arm의 결과가 분모다
시간/메모리/저장량: 서버 56분, 장면당 5.5~12.5분(worker 6개 병렬), 백업 archive 12 MB

### 제어 구간이 실제로 한 일

| | R5 | R2M | H0M |
|---|---:|---:|---:|
| 제어 step 수 | 23~27 (평균 25.0) | R5와 같음 | R5와 같음 |
| 목표 도달 | 176/176 | 176/176 | 해당 없음(시작 위치가 목표) |
| 실제 TCP 이동량 | 4.5~4.7 cm | 1.5~1.7 cm | 0~0.5 cm |
| 목표 도달 step | 23~27 | 6~12 | — |
| 유지 step 수 | 0 | 15~17 | 23~27 |
| 유지 중 목표에서 가장 멀어진 거리 | — | 0.8~1.4 mm | 0~9.5 mm (평균 3.0 mm) |
| 제어 중 episode 종료 | 0 | 0 | 0 |

- 도달 오차 5 mm를 그대로 썼기 때문에 "2 cm" arm의 실제 이동은 약 1.6 cm, "5 cm" arm은 약 4.6 cm다.
- H0M은 위치를 완전히 유지하지 못했다. 36개 block에서 유지 중 5 mm 넘게 벗어났다(예: 장면 10050의 step 92, 큐브를 쥔 상태에서 9.4 mm). 그대로 결과에 넣었다.
- H0M의 한 snapshot(10050, step 132)에서는 유지 중 큐브 B가 2 mm 넘게 움직였다(4개 block). 접촉 센서가 없어서 이유는 알 수 없다.

### 주 비교 (같은 반환 시점, 같은 첫 추론 seed)

| 비교 | 0→1 (장면 수) | 1→0 (장면 수) | 1→1 | 0→0 | block 평균 차이 | 장면 평균 차이 | 장면 +/−/0 |
|---|---:|---:|---:|---:|---:|---:|---|
| R2M 대 R5 | 18 (7) | 24 (8) | 8 | 126 | −0.034 | −0.031 | 3 / 7 / 2 |
| H0M 대 R5 | 23 (7) | 26 (8) | 6 | 121 | −0.017 | −0.030 | 5 / 6 / 1 |
| R2M 대 H0M | 9 (5) | 12 (6) | 17 | 138 | −0.017 | −0.002 | 5 / 4 / 3 |

보조 비교(C0 기준. 중단, queue 폐기, 호출 시점, 예산, 난수의 차이가 함께 들어 있다):

| 비교 | 0→1 (장면 수) | 1→0 (장면 수) | 1→1 | 0→0 | 장면 평균 차이 | 장면 +/−/0 |
|---|---:|---:|---:|---:|---:|---|
| R5 대 C0 | 25 (7) | 18 (4) | 7 | 126 | +0.030 | 7 / 2 / 3 |
| R2M 대 C0 | 15 (8) | 14 (4) | 11 | 136 | −0.002 | 7 / 2 / 3 |
| H0M 대 C0 | 17 (7) | 13 (4) | 12 | 134 | 0.000 | 4 / 3 / 5 |

장면별 평균 성공 비율은 C0 0.177, R5 0.207, R2M 0.175, H0M 0.177이다. 독립 표본은 장면 12개이고, 구간이나 p-value는 계산하지 않았다.

### checkpoint별

| checkpoint | block 수 (장면) | 큐브를 쥔 block | C0 | R5 | R2M | H0M |
|---|---:|---:|---:|---:|---:|---:|
| step 12 | 48 (12) | 0 | 8 | 9 | 17 | 18 |
| step 52 | 48 (12) | 28 | 9 | 18 | 5 | 7 |
| step 92 | 44 (11) | 24 | 8 | 5 | 4 | 4 |
| step 132 | 36 (9) | 4 | 0 | 0 | 0 | 0 |

- step 12(모두 접근 전)에서는 R2M과 H0M이 R5와 C0의 약 두 배였다. step 52에서는 반대로 R5가 가장 많았다. step 132에서는 어느 arm도 성공하지 못했다.
- checkpoint가 늦을수록 도달한 장면이 줄어든다(12 → 12 → 11 → 9). checkpoint 사이의 비교는 같은 표본의 비교가 아니다.
- 장면 안에서도 방향이 엇갈린다. 예: 10051(C0 10/12)에서는 R5 0/12, R2M 4/12, H0M 6/12였고, 10059의 step 52에서는 R5만 4/4, 나머지는 0/4였다.
- 176개 block 중 110개는 네 arm 모두 실패했고 2개는 모두 성공했다. 세 대조 arm의 결과가 서로 같은 block은 120개다.

### C0의 결과로 나눈 건수

| | R5 | R2M | H0M |
|---|---:|---:|---:|
| C0 성공 25개 block 중 실패 | 18 | 14 | 13 |
| C0 실패 151개 block 중 성공 | 25 | 15 | 17 |

C0가 성공한 block은 4개 장면에서만 나왔다. Stage 2B에서는 C0 성공 93개 중 Retreat가 69개에서 실패했고 C0 실패 243개 중 15개에서 성공했다. 두 단계는 장면이 달라 합치지 않는다.

### 사건 읽기 (step별 기록, 원인 라벨 없음)

분모는 checkpoint에서 큐브를 쥐고 있던 모든 block이다: 56개 block, 7개 장면(step 52에 28개, step 92에 24개, step 132에 4개).

| 56개 block 중 | C0 | R5 | R2M | H0M |
|---|---:|---:|---:|---:|
| 제어 구간 내내 파지 flag 유지 | 해당 없음 | 56 | 56 | 56 |
| 제어 구간에 큐브 B가 2 mm 넘게 이동 | 해당 없음 | 0 | 0 | 0 |
| 반환 뒤(C0는 checkpoint 뒤) 열기 명령 있음 / 없음 | 56 / 0 | 55 / 1 | 56 / 0 | 56 / 0 |
| 첫 파지 flag 해제 있음 / 없음 | 56 / 0 | 55 / 1 | 56 / 0 | 56 / 0 |
| 첫 해제 때 큐브 A–B 수평거리 ≤ 2 cm | 7 | 12 | 9 | 3 |
| 첫 해제 때 on-cube flag 켜짐 | 32 | 21 | 10 | 21 |
| 첫 해제 때 수평거리의 중앙값 | 3.1 cm | 4.3 cm | 4.0 cm | 4.1 cm |
| 최종 성공 | 17 | 21 | 9 | 11 |

- 세 대조 arm 모두 제어 구간에서는 파지 flag가 꺼지지 않았다. Stage 2C에서 R에 대해 본 것과 같고, 이번에는 R2M과 H0M도 그랬다.
- 첫 열기 명령과 첫 파지 flag 해제는 찾은 모든 block에서 같은 step이었다.
- "수평거리 2 cm"는 Stage 2C에서 쓴 진단 기준이고 환경의 on-cube 판정(약 3.3 cm)보다 좁다. 그래서 2 cm 밖에서 놓고도 성공한 block이 있다.
- "첫 해제"는 마지막 해제가 아닐 수 있다. 놓았다가 다시 쥐는 경우를 따로 세지 않았다.
- 쥐고 있지 않던 120개 block(12개 장면)의 성공은 C0 8, R5 11, R2M 17, H0M 18이다.

대표 사례를 따로 고르거나 영상으로 확인하지는 않았다. 영상 8개는 사전 규칙대로 처음 2개 장면의 step 12 분기다.

## 해석 범위

말할 수 있는 것:

- 이 12개 장면에서 반환 시점과 첫 추론 seed를 맞추면, 5 cm 이동 arm의 성공 수는 2 cm 이동 arm이나 제자리 유지 arm보다 낮지 않았다(32 대 26, 29).
- 세 대조 arm의 차이는 한 방향이 아니었다. step 12와 step 52에서 차이의 방향이 반대였다.
- 제자리 유지를 명령한 H0M(실제 이동 0.5 cm 이하)에서도 C0가 성공한 block의 절반 이상(13/25)이 실패로 바뀌었다. 이 표본에서는 "5 cm나 2 cm의 이동"만으로 C0 대비 결과 변화를 설명할 수 없다.

말할 수 없는 것:

- 시간과 위치의 기여를 분해했다. R5·R2M·H0M은 최종 위치뿐 아니라 속도, 접촉, 관측 이력, 경로가 모두 다르다.
- H0M − C0가 시간 경과만의 효과다. H0M에는 위치 유지 제어, queue 폐기, 관측 갱신, 호출 시점 변경, 예산 소모가 들어 있고, 위치도 완전히 유지되지 않았다.
- Stage 2B의 Retreat 결과가 틀렸다, 또는 5 cm Retreat가 유익하다. 두 단계는 장면이 다르고 이번 표본은 C0 성공이 4개 장면에 몰려 있다.
- R2M이나 H0M이 쓸 만한 복구 동작이다. 둘 다 R5의 종료 시점을 미리 알아야 하는 진단용 변형이고, "2 cm 이동 후 즉시 반환"은 실행하지 않았다.
- 어떤 block에서 왜 실패했는지. 원인 라벨을 붙이지 않았다.
- 37/100 정책이 qualification을 통과했다, 선택기 학습이 가능하다 또는 불가능하다.

## 변경과 실패

변경한 파일과 이유:

- 신규: `src/hci/pilot2d.py`, `scripts/23_stage2d_pilot.py`, `scripts/stage2d/run.sh`, `tests/test_stage2d_pilot.py`, `amendments/AMENDMENT_008_stage2d_duration_matched_controls.md`, `results/stage2d/`, 이 보고서.
- 수정: `README.md`(Stage 2D 항목), `.gitattributes`(`results/stage2d/**`).
- `simulation.py`, `policies.py`, `pilot.py`, `pilot2b.py`와 기존 실행기는 수정하지 않았다. plan에 Stage 2B 실행 때와 파일 hash가 같다고 기록했다.
- Codex의 실행 전 검토에서 결함은 나오지 않았다. 내가 따로 본 한 곳은 실행 전에 고쳤다: R5 기록을 참조와 비교할 때 길이 값은 기존 audit 허용 오차(1e-6)로, 횟수와 라벨은 완전 일치로 본다. 실제 차이는 전부 0.0이었다.

실패한 명령 및 원인: 없음. 실행 전 로컬에서 테스트 2개가 처음에 실패했는데 둘 다 테스트 쪽 가정이 틀린 것이어서(snapshot이 없는 mock 장면, 값을 덮어쓰는 dict) 테스트를 고쳤다.

수행하지 않은 일: 재학습, 거리·시간 탐색, Refresh와 대체 난수 Continue의 재실행, 예측기·선택기 학습, qualification, calibration, 최종 Test, 12개를 넘는 수집, 대표 사례 선정과 영상 판독, 결과로 나눈 추가 사건 표.

비용: credit 8.1987 → 7.9875, 약 $0.21(상한 $1.00). 누적: Stage 1D 약 $1.14, 2A $0.38, 2B $0.30, 2C $0, 2D $0.21.

## 다음 단계

실행 전에 적어 둔 네 갈래 가운데 네 번째에 해당한다고 판단한다: 조건 사이의 차이가 소수 장면과 checkpoint에 달려 있고 방향이 엇갈린다.

권고: **현재 Retreat 계열(R5, R2M, H0M)의 효용 주장을 보류한다.** 후속 후보로 동결할 arm은 이 결과로 고를 수 없다.

- R2M이 R5와 H0M 모두보다 여러 장면에서 낫다는 조건은 충족되지 않았다(R5 대비 장면 3개에서 높고 7개에서 낮음).
- "H0M과 R2M이 비슷하고 둘 다 R5보다 낫다"도 충족되지 않았다. 둘은 서로 비슷했지만(장면 평균 차이 −0.002) R5보다 높지 않았다.
- "H0M도 R5처럼 나쁘다"는 전제가 성립하지 않았다. 이 표본에서는 R5가 C0보다 낮지 않았다.

필요 작업: 사용자의 결정. 자동으로 장면을 늘리거나 arm을 추가하지 않았다.
예상 계산량의 산출 근거: 이번 실측은 worker 6개에서 장면당 약 2분(continuation 1회 약 6.1초, 단일 실행 기준)이었다. 같은 설계로 장면을 12개 더 하면 약 50분, $0.2 안팎이다. 이것은 참고용 환산이고 제안이 아니다.
동결/승인이 필요한 사항: 없음. 정책, 설정, 후보 목록은 그대로다.

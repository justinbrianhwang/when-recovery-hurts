# Stage 보고서 — Stage 2A 탐색적 복구 파일럿

D500-clean 80k 정책을 그대로 두고 train 장면 24개에서 분기 1,092회를 끝까지 수집했다. 두 복구 행동 모두 Rescue와 Harm이 여러 장면에서 관측됐다. 이 결과는 개발용 기술통계이고, 개입의 효용이나 안전성을 입증하지 않는다.

실행 ID: `stage2a_001`
단계: Stage 2A — 고정 정책의 탐색적 복구 파일럿 (복원 audit + 분기 수집)
작업/정책 seed: StackCube-v1 / 정책 `stack_1d_s1_d500_clean_v1_i080000` (학습 seed 1, 재학습 없음)
실제 실행 환경: Vast.ai 인스턴스 60672162 (RTX 3090 24 GB, driver 595.91.07, 32 vCPU, 단가 $0.2483/h), 2026-11-14 16:56~18:27 UTC. 삭제 완료.
코드/upstream 버전: 저장소 `a5c0b1c` + 이 단계의 새 파일(실행 시점에는 미커밋, 파일별 SHA256을 plan에 고정) / ManiSkill `a4a4f9272ad64b1564035874b605ceb687b63ed8`
변경 기록: `amendments/AMENDMENT_005_stage2a_exploratory_scope.md`

## 상태

```text
base_policy_qualification: BELOW_ORIGINAL_GATE (37/100 < 50/100)
exploratory_recovery_pilot: AUTHORIZED_BY_STAGE2A_SCOPE
formal_study_ready: NOT_ESTABLISHED
calibration: NOT_RUN
final_test: NOT_RUN
```

qualification 판정은 그대로다. 이 파일럿은 qualification 결과를 본 뒤에 정한 후속 탐색이고, 사전 등록된 확증 실험이 아니다.

## 요약

- 복원 audit는 이 정책·설정·host에서 통과했다. 18개 checkpoint에서 비교한 모든 항목의 차이가 0.0이다.
- 계획한 24개 장면을 모두 실행했다. snapshot 91개, block 364개, continuation 1,092회다. 계획 상한(96 / 384 / 1,152)보다 적은 이유는 trunk가 먼저 성공해 checkpoint가 없어진 5곳뿐이다.
- 재현 검사는 전부 일치했다. repeat 0의 Continue가 복원하지 않은 trunk와 같은지 91회, 같은 Continue를 다시 실행한 8회 모두 차이 0.0이다.
- trunk 성공은 24개 장면 중 4개다.
- Refresh: Rescue 30, Harm 10, 둘 다 성공 57, 둘 다 실패 267 (block 364개).
- Retreat–Retry: Rescue 33, Harm 26, 둘 다 성공 41, 둘 다 실패 264 (block 364개).
- Rescue가 한 번이라도 나온 장면은 13개, Harm이 나온 장면은 10개, 둘 다 나온 장면은 6개다.
- Continue 자체가 미래 noise에 따라 바뀐다. snapshot 91개 중 14개에서 repeat에 따라 Continue의 성공 여부가 갈렸다. Rescue와 Harm 수치는 이 변동과 함께 읽어야 한다(아래 "noise 참고치").
- 비용은 credit 차이로 약 $0.38이다(Stage 2A 상한 $1.50).

## 실제 수행한 명령

인자 전체는 `logs/stage2a_001.status`(사본 `results/stage2a/commands_stage2a_001.status`)에 있다. 시각은 UTC.

| 명령 | 종료 코드 | 로그 경로 | 시작/종료 시각 |
|---|---:|---|---|
| (로컬) `python -m pytest -q` | 0 (108 passed, 3 skipped) | — | 대여 전 |
| (로컬) `18_stage2a_pilot.py plan --out runs/stage2a_001 …` | 0 | `runs/stage2a_001/plan.json` | 대여 전 |
| `00_system_packages.sh`, `01_bootstrap_vast.sh` | 0, 0 | `logs/s2a_install_sys.log`, `logs/s2a_bootstrap.log` | 16:59:07 / 17:03:24 |
| `python -m hci doctor` / `python -m pytest -q` | 0 / 0 (111 passed) | `logs/s2a_doctor.log`, `logs/s2a_pytest.log` | 17:03:24 / 17:03:47 |
| `18_stage2a_pilot.py scenes --plan … --demo-keys … --out runs/stage2a_001/scenes.json` | 0 | `logs/s2a_scenes.log` | 17:03:47 / 17:05:04 |
| `18_stage2a_pilot.py audit --plan … --out runs/stage2a_001/audit` | 0 | `logs/s2a_audit.log` | 17:05:04 / 17:15:06 |
| `18_stage2a_pilot.py collect --start 0 --stop 2 --label first2` | 0 | `logs/s2a_collect_first2.log` | 17:15:06 / 17:37:56 |
| `18_stage2a_pilot.py techcheck` | 0 | `logs/s2a_techcheck.log` | 17:37:56 / 17:37:58 |
| `18_stage2a_pilot.py scale --usd-per-hour 0.27 --spent-usd 0.18 --reserve-usd 0.25 --workers 6 --efficiency 0.7` | 0 | `logs/s2a_scale.log` | 17:38:15 / 17:38:17 |
| `timeout 5400 18_stage2a_pilot.py collect --start {2..7} --stop 24 --step 6 --label w{0..5}` (6개 동시) | 모두 0 | `logs/s2a_collect_w0.log` ~ `w5.log` | 17:38:17 / 18:25:40 |
| `18_stage2a_pilot.py finalize --results runs/stage2a_001/results` | 0 | `logs/s2a_finalize.log` | 18:25:40 / 18:25:42 |
| `run.sh pack` | 0 | `logs/s2a_pack.out` | 18:25 |
| (로컬) `19_stage2a_noise_reference.py --scenes … --out results/stage2a/posthoc_continue_noise_reference.json` | 0 | — | 삭제 후 |

실패한 실험 명령은 없다. 인스턴스 생성 명령은 `"success": false`를 돌려줬지만 계약은 정지 상태로 만들어져 있었고, `start instance`로 시작했다.

## 산출물

| 파일 | 용도 | SHA256 |
|---|---|---|
| `runs/stage2a_001/plan.json` | 대여 전 고정한 계획 (정책·설정·코드 hash, 장면 목록, 상한) | `d040d8a8646ea118…` |
| `runs/stage2a_001/scenes.json` | 초기 큐브 배치 대조 결과와 확정 목록 | `ce721a3268f8984d…` |
| `runs/stage2a_001/audit/audit.json` | 이 정책의 복원 audit | `c04ba0295c4b003f…` |
| `runs/stage2a_001/techcheck.json` | 처음 2개 장면의 기술 확인 | `2544593d7c8b4d45…` |
| `runs/stage2a_001/scale_decision.json` | 규모 결정 (시간과 비용만 입력) | `f7c65796213db2b9…` |
| `runs/stage2a_001/branches/dataset.json` | 수집 완료 상태 | `88c475f67644bff6…` |
| `runs/stage2a_001/branches/scenes/*.json` (24개) | 장면별 block 기록 | 백업 목록 참조 |
| `runs/stage2a_001/branches/traces/*.npz` (24개) | step별 TCP·큐브·파지·행동 기록 (git 제외) | 백업 목록 참조 |
| `results/stage2a/pair_outcomes.csv` | block × 행동 728행 | `9cbf2ee204def536…` |
| `results/stage2a/scene_summary.csv` | 장면별 요약 24행 | `605468fc678b3acc…` |
| `results/stage2a/skip_reasons.csv` | 실행하지 않은 checkpoint 5행 | `88aca709008d5866…` |
| `results/stage2a/summary.json` | 집계 표 | `2a529bebf8378b96…` |
| `results/stage2a/videos/*.mp4` (6개) | 장면 10000, 10001의 step 12, repeat 0, 세 선택 | 백업 목록 참조 |
| `stage2a_out.tgz` (6.4 MB) | 백업 묶음 | `354096a2e8b39546…` (다운로드 후 일치) |

묶음 안 98개 파일 목록은 `results/stage2a/backup_sha256_stage2a_001.txt`이고 로컬에서 98/98 일치했다. 서버가 만든 `summary.json`과 `pair_outcomes.csv`는 로컬에서 장면 기록으로 다시 계산해 같은 값임을 확인했다.

정책 파일은 대여 전에 보존 목록(`results/stage1d_clean/backup_sha256_stage1d_clean_002.txt`)의 전체 SHA256과 대조했다: `weights.pt` `faacd2d86464c725a4fb95563a5ed5808019a8b420795b9a2a931822ed89db0e`, `policy_meta.json` `9976053598632fd58c48b89c9dffabf8f97b3d18ccf5062127c9c0be3de8a468`.

## 검증

| 항목 | PASS / FAIL / NOT RUN | 실제 근거 |
|---|---|---|
| CPU tests | PASS | 로컬 108 passed 3 skipped, 서버 111 passed. mock 물리이고 조작 성공과는 무관 |
| CUDA | PASS | `hci doctor`: RTX 3090, driver 595.91.07 |
| Vulkan/RGB | PASS | `vulkaninfo`가 NVIDIA 장치를 보고. 복원 직후 환경이 돌려준 RGB와 저장된 RGB의 차이 0.0 |
| 정책 rollout | PASS (실행됨) | trunk 24회. 성공 4, 실패 20. 성능 기준 통과를 뜻하지 않음 |
| 상태+정책 복원 | PASS | 아래 "복원 audit" |
| 개입 행동 영상 | PASS | 6개 저장 |
| seed split | PASS | 장면 24개는 train 앞부분. 다른 선언 집합과 seed 겹침 0, 초기 배치가 같은 장면 0 |
| 데이터 완료 상태 | PASS | 24/24 장면 기록, 오류 파일 0, `completed: true`, 재현 검사 불일치 0 |
| 학습 완료 | NOT RUN | 이번 단계에 학습 없음 |
| 과학적 가설 검증 | NOT RUN | 탐색적 기술통계만. 가설 검증이 아님 |

### 복원 audit

- 대상: audit seed 100~104. 다섯 개 모두 시연 장면과 겹친다. 복원 검사에만 썼고 성능 표본이나 분모가 아니다.
- 도달한 checkpoint는 18개다(seed별 3, 3, 4, 4, 4). 도달하지 못한 2개는 trunk가 step 101, 120에 먼저 성공한 step 132다.
- 각 checkpoint에서 Continue 20 step → 복원 후 반복 → Retreat 분기 종료 후 복원·반복 → Refresh 분기 종료 후 복원·반복. Retreat 분기 18회, Refresh 분기 18회를 실제로 실행했다.
- 비교 항목: simulator 상태, step 카운터, 정책 history(관절·TCP, RGB), 남은 action queue, 마지막 행동, 실행 행동 기록, TCP 기록, 종료·성공 flag, 개입·Retreat 기록, queue·history 길이, 추론 횟수, 난수 상태(Python, NumPy, Torch, CUDA, 환경), 복원 직후 환경이 돌려준 관측.
- 결과: 수치 항목의 최대 차이는 모두 0.0, 정수·flag·난수 항목은 모두 일치.
- 범위: 같은 장면에서 검사한 상태들에 대한 결과다. simulator 전체의 결정성 증명이 아니다.

### 처음 2개 장면의 기술 확인

결과의 유불리는 읽지 않는 검사다. 12개 항목 모두 통과했다: audit 통과, 기록 정합성, 오류 파일 없음, continuation 96회 이내, trunk 재현, Continue 재실행 일치, 재현 검사 누락 없음, 영상 6개, Retreat가 실제 제어 step으로 실행됨, Retreat 중 그리퍼 명령 유지, Refresh가 checkpoint에서 추론함, horizon 준수.

### 규모 결정

처음 2개 장면에서 continuation당 12.8초가 나왔다. 24개 전체의 예상 누적 비용은 $0.75로 상한 $1.50 안이어서 24개로 정했다. 입력은 시간과 금액뿐이다.

## 관측 결과

원본 episode 수: 24 (train seed 10000~10023, 역할 `stage2a_exploratory_dev`)
분기 실행 수: 1,092 (block 364 × 세 선택). 별도로 trunk 24회, 재현 검사용 Continue 8회, audit.
성공/실패 수: trunk 성공 4 / 실패 20. Continue가 성공한 block 67 / 실패한 block 297.
시간/메모리/저장량: 인스턴스 1시간 31분. 장면당 225~784초(중앙값 702초). GPU 메모리는 프로세스당 약 0.6 GB. 백업 6.4 MB.

### 실행량과 빠진 칸

| | 계획 상한 | 실제 |
|---|---:|---:|
| 장면 | 24 | 24 |
| snapshot | 96 | 91 |
| block (scene, checkpoint, repeat) | 384 | 364 |
| continuation | 1,152 | 1,092 |

빠진 checkpoint 5개는 모두 "trunk가 그 전에 성공해 종료"다: 장면 10006·10011·10016의 step 132, 장면 10008의 step 92와 132. 그 밖의 부적격이나 오류는 없다. 실행하지 않은 칸은 60개(5 × 4 repeat × 3 선택)이고 다른 상태로 채우지 않았다.

### 네 분류

기준은 **같은 block의 Continue**다. block 하나에 Continue는 하나이고, 행동마다 한 번씩만 센다.

| 행동 | checkpoint | block | Rescue | Harm | 둘 다 성공 | 둘 다 실패 | Rescue가 나온 장면 | Harm이 나온 장면 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Refresh | 전체 | 364 | 30 | 10 | 57 | 267 | 10 | 8 |
| Refresh | 12 | 96 | 10 | 8 | 21 | 57 | 6 | 6 |
| Refresh | 52 | 96 | 12 | 2 | 21 | 61 | 7 | 2 |
| Refresh | 92 | 92 | 5 | 0 | 15 | 72 | 2 | 0 |
| Refresh | 132 | 80 | 3 | 0 | 0 | 77 | 1 | 0 |
| Retreat–Retry | 전체 | 364 | 33 | 26 | 41 | 264 | 9 | 8 |
| Retreat–Retry | 12 | 96 | 4 | 8 | 21 | 63 | 3 | 5 |
| Retreat–Retry | 52 | 96 | 14 | 8 | 15 | 59 | 7 | 5 |
| Retreat–Retry | 92 | 92 | 12 | 10 | 5 | 65 | 4 | 3 |
| Retreat–Retry | 132 | 80 | 3 | 0 | 0 | 77 | 1 | 0 |

조건부 비율(고정된 checkpoint 표본의 기술통계):

| 행동 | checkpoint | `H_branch` = Harm / Continue 성공 block | `R_branch` = Rescue / Continue 실패 block |
|---|---|---|---|
| Refresh | 전체 | 10 / 67 = 14.9% | 30 / 297 = 10.1% |
| Refresh | 12 | 8 / 29 = 27.6% | 10 / 67 = 14.9% |
| Refresh | 52 | 2 / 23 = 8.7% | 12 / 73 = 16.4% |
| Refresh | 92 | 0 / 15 = 0% | 5 / 77 = 6.5% |
| Refresh | 132 | 0 / 0 = NA | 3 / 80 = 3.8% |
| Retreat–Retry | 전체 | 26 / 67 = 38.8% | 33 / 297 = 11.1% |
| Retreat–Retry | 12 | 8 / 29 = 27.6% | 4 / 67 = 6.0% |
| Retreat–Retry | 52 | 8 / 23 = 34.8% | 14 / 73 = 19.2% |
| Retreat–Retry | 92 | 10 / 15 = 66.7% | 12 / 77 = 15.6% |
| Retreat–Retry | 132 | 0 / 0 = NA | 3 / 80 = 3.8% |

- step 132의 `H_branch`는 NA다. Continue가 성공한 block이 없어서 Harm을 관측할 수 없었다. 0%가 아니다.
- step 92에서 Refresh의 Harm 0건은 "Continue가 성공한 block 15개(장면 4개)에서 0건"이다. 안전하다는 뜻이 아니다.
- 구간이나 p-value는 계산하지 않았다. 한 장면의 block들은 같은 과거를 공유하므로 독립 표본이 아니다.

### noise 참고치 (결과를 본 뒤 추가한 계산)

Refresh와 Retreat–Retry는 같은 block의 Continue와 다른 시점에 추론하므로 다른 noise를 쓴다. 그래서 Rescue와 Harm에는 "행동의 효과"와 "noise가 달라진 효과"가 함께 들어 있다. 얼마나 큰지 가늠하려고, 같은 snapshot에서 **서로 다른 repeat의 Continue끼리** 결과가 뒤바뀌는 빈도를 저장된 기록에서 셌다. 재실행은 없다.

| checkpoint | Continue 실패 → 다른 repeat의 Continue 성공 | Continue 성공 → 다른 repeat의 Continue 실패 |
|---|---|---|
| 전체 | 47 / 891 = 5.3% | 47 / 201 = 23.4% |
| 12 | 31 / 201 = 15.4% | 31 / 87 = 35.6% |
| 52 | 13 / 219 = 5.9% | 13 / 69 = 18.8% |
| 92 | 3 / 231 = 1.3% | 3 / 45 = 6.7% |
| 132 | 0 / 240 = 0% | 0 / 0 = NA |

단위는 한 snapshot 안의 repeat 순서쌍(12쌍)이다. 뒤바뀜이 나온 장면은 각각 9개다.

읽는 법:

- step 12에서는 두 행동의 Rescue·Harm 비율이 이 참고치와 비슷하거나 낮다. 이 시점의 차이는 noise만으로도 나올 수 있는 크기다.
- step 52와 92에서는 두 행동 모두 Rescue 비율이 참고치보다 높다(Refresh 16.4%·6.5%, Retreat 19.2%·15.6% 대 참고치 5.9%·1.3%).
- 같은 시점에서 Retreat–Retry의 Harm 비율은 참고치보다 높고(34.8%·66.7% 대 18.8%·6.7%), Refresh의 Harm 비율은 참고치보다 낮다(8.7%·0% 대 18.8%·6.7%).
- 이 비교는 사후에 추가한 것이고 계획된 대조가 아니다. 참고치의 쌍은 서로 겹치고 장면 수가 적어 차이의 크기를 추정하는 데 쓸 수 없다. 같은 block 안에 "noise만 바꾼 Continue"를 대조 선택으로 넣어야 분리된다.

### 장면 단위

| 항목 | 장면 수 (24개 중) |
|---|---:|
| Rescue가 한 번이라도 나온 장면 | 13 |
| Harm이 한 번이라도 나온 장면 | 10 |
| Rescue와 Harm이 모두 나온 장면 | 6 |
| Refresh와 Retreat–Retry의 결과가 달라진 block이 있는 장면 | 17 |
| 어떤 선택·repeat에서도 성공이 한 번도 없는 장면 | 7 |

- Rescue는 모두 trunk가 실패한 장면에서 나왔다(Refresh 10개 장면, Retreat–Retry 9개 장면).
- Harm은 trunk가 성공한 4개 장면 모두와, trunk는 실패했지만 다른 repeat의 Continue가 성공한 장면에서 나왔다.
- 장면별 수치는 `results/stage2a/scene_summary.csv`에 있다. 집계 비율은 block이 많은 장면에 더 큰 가중치를 준다.

### checkpoint 시점의 상태별

checkpoint까지 trunk가 도달한 단계(관찰 라벨)로 나눈 block 수다.

| checkpoint까지의 단계 | block | 장면 | Continue 성공 | Refresh Rescue / Harm | Retreat–Retry Rescue / Harm |
|---|---:|---:|---:|---|---|
| 큐브에 접근 전 | 164 | 24 | 29 | 13 / 8 | 5 / 8 |
| 접근했으나 파지 전 | 20 | 4 | 5 | 3 / 0 | 3 / 1 |
| 파지했으나 들기 전 | 76 | 13 | 18 | 3 / 2 | 10 / 7 |
| 들었으나 정렬 전 | 60 | 11 | 8 | 7 / 0 | 9 / 3 |
| 정렬했으나 성공 전 | 44 | 8 | 7 | 4 / 0 | 6 / 7 |

- checkpoint 시점에 큐브를 쥐고 있던 block(장면 18개)에서 Retreat–Retry는 Rescue 18, Harm 17이고 Refresh는 Rescue 14, Harm 2다.
- "정렬했으나 성공 전" 단계에서 Continue가 성공한 block 7개는 Retreat–Retry에서 7개 모두 실패했다. 이 7개는 장면 2개(10011, 10014)에서 나왔다. 장면 2개의 관측이다.
- 단계는 거리·높이 기준으로 붙인 관찰 라벨이고 실패 원인이 아니다.

### trunk의 실패 위치

| trunk가 도달한 가장 먼 단계 | 장면 수 |
|---|---:|
| 성공 | 4 |
| 정렬했으나 성공 못 함 | 7 |
| 들었으나 정렬 못 함 | 4 |
| 파지했으나 들지 못함 | 4 |
| 접근했으나 파지 못 함 | 2 |
| 큐브에 접근 못 함 | 3 |

실패 20개 중 15개는 큐브를 쥔 뒤에 멈췄다. qualification B의 실패 63개를 다시 실행한 것이 아니고 이 24개 장면의 기록이다.

### Retreat의 실제 동작

364회 모두 목표에 도달했다(`reached`). 사용 step은 23~28(중앙값 25), 실제 변위는 4.50~4.70 cm, 목표와의 오차는 3.1~5.0 mm다. 모든 step은 같은 200-step 예산에서 차감됐고 그리퍼 명령은 유지됐다. 제어 사양을 따랐다는 뜻이고, Retreat가 작업에 무해하다는 뜻이 아니다.

### 영상

`results/stage2a/videos/`에 6개가 있다. 장면 10000과 10001의 첫 유효 checkpoint(step 12), repeat 0의 Continue, Refresh, Retreat–Retry다. 사전에 정한 규칙으로 고른 것이고 결과를 보고 고른 영상은 없다. 두 장면 모두 trunk가 실패한 장면이다.

## 해석의 범위

- trunk 성공 4/24는 qualification의 37/100과 다른 장면에서 나온 작은 표본이다. 새 성공률 추정으로 쓰지 않는다.
- checkpoint에서 선택을 강제한 결과다. 온라인 개입 정책의 성능이나 악화율 `H(g)`가 아니다. 선택기는 학습하지도 평가하지도 않았다.
- 결과는 이 simulator와 선언한 noise 결합 규칙 아래의 것이다.
- 성공 판정은 `success_once_absorbing`이다. 성공 뒤에도 쌓인 상태가 유지되는지는 보지 않았다.
- 초기 배치 대조는 시연 원본 1000개, audit, dev, policy 평가 집합까지다. calibration과 Test는 seed 번호로만 대조했다. 가장 가까운 기존 장면과의 차이는 8.0 mm였다.
- 처음 2개 장면은 프로세스 하나로, 나머지 22개는 6개 프로세스로 동시에 실행했다. 재현 검사는 양쪽 모두 일치했다. 추론 호출당 시간은 동시 실행에서 0.82초로 늘었다(qualification 때 0.17초). 결과가 아니라 처리 시간의 차이다.
- 이번 한 번의 결과로 어느 복구 행동이 유익하다거나 해롭다고 결론짓지 않는다.

## 변경과 실패

변경한 파일과 이유:

- `src/hci/pilot.py` (신규): 전체 실행 상태 비교, 복원 audit, 장면별 block 수집, 정합성 검사, 기술통계.
- `scripts/18_stage2a_pilot.py`, `scripts/stage2a/run.sh` (신규): plan에 묶인 실행기. 기존 `hci collect`·`hci audit`와 Stage 1D 실행기는 수정하지 않았다.
- `scripts/19_stage2a_noise_reference.py` (신규, 결과 확인 후 추가): 저장된 기록만 읽는 참고치 계산.
- `src/hci/simulation.py`: prefix replay 복원 시 환경이 마지막에 돌려준 관측을 `replay_raw`로 남긴다. 실행에 쓰는 값과 복원 절차는 그대로다.
- `tests/test_stage2a_pilot.py` (신규, 15개): 같은 block의 Continue 연결, Continue의 queue·history 무변경, 결과와 무관한 snapshot 기록, 성공 후 개입 거부, 장면의 단일 역할, 일부 실패 시 완료 표시 금지, queue를 잘못 복원한 경우 검출, 환경 관측 차이 검출, 재현 불일치 표시 등.
- `amendments/AMENDMENT_005_stage2a_exploratory_scope.md`, `README.md`, `.gitattributes`.

Codex 검토에서 5건을 받았다. 4건을 반영했다: 재현 검사 실패 시 데이터 집합을 완료로 표시하지 않음, 장면 목록이 plan과 달라지면 수집 거부, `finalize`도 plan의 hash를 다시 대조, 동시 worker의 status 파일 분리. 나머지 1건(실행기 명령 문자열)은 실제로는 동작하는 형태였지만 배열로 바꿨다.

실패한 명령 및 원인: 실험 명령 실패 없음. 인스턴스 생성 응답이 `success: false`였던 원인은 확인하지 못했다(계약은 생성되어 있었고 수동 시작 후 정상 동작).

수행하지 않은 일: 새 학습, 데이터 확대, 체크포인트 재선택, qualification, calibration, 최종 Test, 개입 예측기(Q·도움·악화) 학습, 후보 21개 평가, 5,760회 수집. `RUN_FROZEN_TEST` 토큰은 만들지 않았다.

## 비용

| 항목 | 값 |
|---|---|
| 대여 전 credit | $8.8918 |
| 삭제 후 credit | $8.5101 |
| Stage 2A 사용 | **약 $0.38** (상한 $1.50) |
| 표시 단가 × 가동 시간 | $0.2483/h × 1시간 31분 ≈ $0.38 |
| 업로드 / 다운로드 | 23 MB / 6.4 MB |

- 이번에는 credit 차이와 "단가 × 시간"이 거의 같다. 전송 단가가 낮은 host($0.004/GB)를 골랐고 올리고 내린 양이 작았다.
- Stage 1D 인스턴스에서 약 $0.33이 더 나갔던 것은 그 host의 전송 단가($0.039/GB)와 설치 중 내려받은 수 GB로 설명되는 크기다. 청구 내역으로 확인한 것은 아니고 추정이다.
- Stage 1D 삭제 뒤 credit이 $0.008 더 줄어 있었다(8.8998 → 8.8918). Stage 1D 누적은 약 $1.14다. Stage 2A와는 따로 적는다.
- 유휴 대기는 없었다. 수집 종료 → 묶음 → 다운로드 → 98/98 hash 대조 → 삭제를 이어서 처리했다. 삭제 후 `No instances found`를 확인했다.

## 다음 단계

필요 작업: 한 가지로 좁혀 제안한다. **다음 분기 실험에 "noise만 바꾼 Continue"를 네 번째 대조 선택으로 넣는 것.** 이번 결과에서 Rescue와 Harm이 여러 장면에서 나왔고 시점·상태에 따라 달랐지만, Continue 자체가 noise로 뒤바뀌는 빈도가 같은 크기 범위에 있어서 지금 자료로는 행동의 효과와 noise의 효과를 나눌 수 없다. 이것이 나뉘기 전에는 예측기나 선택기를 학습할 근거가 약하다. 대조 선택은 개입 정의나 정책을 바꾸지 않고 추가할 수 있다.

예상 계산량의 산출 근거: 새 train 장면 24개(seed 10024~)에 선택 4개면 최대 1,536 continuation이다. 이번 측정(1,092회와 audit에 1시간 31분, $0.38)을 비례로 늘리면 약 2시간, $0.5~0.6이다.

동결/승인이 필요한 사항:

- 위 실험을 할지, 한다면 대조 선택의 noise 규칙(어느 repeat의 noise를 쓸지)을 실행 전에 정해야 한다.
- 이번 24개 장면은 결과를 본 개발 표본이다. 다음 실험은 새 장면을 써야 하고, train에는 96개가 남아 있다.
- Retreat 설정을 바꿀지는 별개의 결정이다. 이번 실행 중에는 바꾸지 않았다.
- 추가 실행은 자동으로 시작하지 않는다.

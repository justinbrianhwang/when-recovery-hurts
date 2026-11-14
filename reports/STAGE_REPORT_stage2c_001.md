# Stage 보고서 — Stage 2C 저장 기록의 사후 분석

새 실행 없이 Stage 2B의 저장 기록만 읽었다. Stage 2B 보고서의 집계는 원본 행에서 모두 다시 나왔다. 같은 상태의 다른 난수 반복으로 고른 선택은 Continue보다 장면 평균 0.01~0.03 높았고, 그 차이는 2~4개 장면에서 나왔다. Retreat 구간 동안에는 기록된 파지·들어올림·정렬 flag가 한 번도 꺼지지 않았고, R의 실패는 모두 정책이 제어를 돌려받은 뒤 step 200에서 기록됐다. 결과를 본 뒤 계획한 개발용 분석이며, 원인이나 선택기의 성능을 입증하지 않는다.

실행 ID: `stage2c_001` (보충: `stage2c_001_supplement`)
단계: Stage 2C — 저장 기록의 사후 오프라인 분석
작업/정책 seed: StackCube-v1 / 정책 `stack_1d_s1_d500_clean_v1_i080000` (실행하지 않음, 기록만 읽음)
실제 실행 환경: 로컬 PC, CPU만 사용, Python 3.14 + NumPy. GPU 대여 없음, 비용 $0. 2026-11-14 22:10 UTC부터(명령별 시각은 아래 기록 파일).
코드/upstream 버전: 저장소 `8ef0cf7` + 이 단계의 새 파일(파일 hash를 manifest에 고정). 읽은 기록은 Stage 2B 실행 당시 코드(Stage 2B plan `9bbe7aa5…`에 고정된 hash)가 만든 것이다.
변경 기록: `amendments/AMENDMENT_007_stage2c_offline_analysis.md`

## 상태

```text
base_policy_qualification: BELOW_ORIGINAL_GATE
stage2a: COMPLETED_EXPLORATORY_ONLY
stage2b: COMPLETED_EXPLORATORY_ONLY
stage2c: COMPLETED_POSTHOC_OFFLINE_DIAGNOSTIC
observation_based_selector: NOT_TRAINED
calibration: NOT_RUN
final_test: NOT_RUN
```

simulator rollout 0회, 정책 추론 0회, 학습 0회, 새 장면 0개.

## 요약

- **재검산**: 336개 block을 CSV와 장면 JSON에서 따로 읽어 이었다. 성공 수, 여섯 쌍의 건수, 장면 평균, 빠진 checkpoint, 재현 검사 수가 Stage 2B 보고서와 모두 같다. 원본 파일 72개는 서버 hash 목록과 일치한다.
- **선택 일관성 진단**(repeat 하나를 남기고 나머지 셋으로 arm을 고름): 장면 평균 성공 비율이 항상 C0 0.362, 항상 C1 0.376에 대해 `{C0,F}` 0.375, `{C0,R}` 0.385, `{C0,F,R}` 0.396이다.
- Refresh를 고른 38행 중 남겨둔 repeat에서 C0의 실패를 성공으로 바꾼 것이 9행(3개 장면), 성공을 실패로 바꾼 것이 3행(3개 장면)이다.
- Retreat를 고른 26행(`{C0,R}`) 중 실패→성공이 10행, 성공→실패가 1행이다. 실패→성공은 모두 두 장면(10035, 10043)에서 나왔다.
- **Retreat 구간**(기록된 flag와 위치 기준): 336개 block 어디에서도 Retreat 중에 파지 flag가 꺼지거나, 들어올림·정렬이 사라지거나, 큐브 B가 2 mm 이상 움직이지 않았다.
- **R이 C0보다 나빴던 69개 block**: 모두 step 200까지 가서 끝났다. 66개에서는 제어를 돌려받은 시점에 남은 step이 C0가 checkpoint부터 성공까지 쓴 step보다 많았다.
- 그 69개 중 42개(10개 장면)는 checkpoint에서 큐브를 쥐고 있었다. Retreat 뒤 정책이 gripper를 여는 명령을 냈을 때 큐브는 42개 모두 큐브 B에서 2 cm 넘게 떨어져 있었다(중앙값 7.7 cm). 같은 block의 C0는 42개 모두 큐브 B 위에서 놓았다.
- 영상은 새로 만들지 않았다. 위 내용은 step별 trace에서 읽은 것이다.

## 실제 수행한 명령

| 명령 | 종료 코드 | 로그 경로 | 시작/종료 시각 (UTC) |
|---|---:|---|---|
| `python -m pytest -q` | 0 (157 passed, 3 skipped) | 터미널 | manifest 작성 전 (시각 미기록) |
| `21_stage2c_offline_review.py manifest --out runs/stage2c_001` | 0 | `logs/s2c_manifest.log` | 22:10:54 / 22:10:56 |
| `21_stage2c_offline_review.py verify --label input_hash_check_before --out runs/stage2c_001` | 0 | `logs/s2c_verify…before.log` | 22:10:56 / 22:10:57 |
| `21_stage2c_offline_review.py run --out runs/stage2c_001` | 0 | `logs/s2c_run.log` | 22:10:57 / 22:10:58 |
| `21_stage2c_offline_review.py verify --label input_hash_check_after --out runs/stage2c_001` | 0 | `logs/s2c_verify…after.log` | 22:10:58 / 22:10:59 |
| `22_stage2c_release_reading.py --events … --traces … --scenes … --out runs/stage2c_001_supplement` | 0 | `logs/s2c_supplement.log` | 기록: `results/stage2c/commands_stage2c_001.status` |
| `21_stage2c_offline_review.py verify --label input_hash_check_after_supplement --out runs/stage2c_001` | 0 | 터미널 | 같은 파일 |

분석 명령은 manifest를 쓴 뒤 **한 번** 실행했고 다시 돌리지 않았다. 입력 123개 파일의 hash는 분석 전, 분석 후, 보충 집계 후 모두 manifest와 같다(변경 0, 추가 0).

## 산출물

`runs/stage2c_001/`(git 제외)에 쓰고 `results/stage2c/`에 복사했다.

| 파일 | 용도 | SHA256 (앞 16자) |
|---|---|---|
| `analysis_manifest.json` | 분석 전 고정: 입력·코드 hash, 규칙, 환경 | `a566da0dc8a573ba` |
| `integrity_and_recount.json` | 키·분모·기존 집계와의 대조 | `d94b692e5f11c105` |
| `snapshot_outcome_counts.csv` | snapshot 84개의 arm별 성공 수와 패턴별 repeat | `7a38c09e713842df` |
| `repeat_transfer_rows.csv` | 규칙 × snapshot × 남겨둔 repeat 1,008행 | `7d26d0d1d088a621` |
| `repeat_transfer_scene_summary.csv` | 장면별 요약과 전체 | `1f1868d4b00cd054` |
| `repeat_transfer_summary.json` | 선택 빈도, 도움·악화 건수, `{C0,F,R}`−`{C0,F}` 목록 | `1c271a202022dcab` |
| `retreat_event_rows.csv` | block 336개의 사건·시각·관찰 flag·trace 위치 | `58538427f8997814` |
| `retreat_event_summary.csv` | 집합별 block·snapshot·장면 수와 flag 건수 | `9a573a95ee6e6262` |
| `case_index.csv`, `case_timeseries/*.csv` (8개) | 설명용 사례와 step별 표 | `3645051b0ae912d2` |
| `input_hash_check_{before,after,after_supplement}.json` | 입력 불변 확인 | `ec365cf2…`, `3b149ae8…` |
| `run_status.json` | 완료 상태 | `88aab679e80df275` |
| `supplement/grasp_release_summary.json`, `grasp_release_rows.csv` | 보충: 파지 flag가 꺼진 시점의 명령과 위치 | `b4a8c643fee35e82`, `b663b2544843cf15` |
| `scripts/21_stage2c_offline_review.py` | 분석 코드(manifest에 고정) | `53ca3b9a51aacbdf` |
| `scripts/22_stage2c_release_reading.py` | 보충 집계(주 결과를 본 뒤 추가) | `55d03f36fee6221a` |

## 검증

| 항목 | PASS / FAIL / NOT RUN | 실제 근거 |
|---|---|---|
| CPU tests | PASS | 157 passed, 3 skipped(imageio·mani_skill·gymnasium 없음). Stage 2C 테스트 29개는 합성 기록만 사용 |
| CUDA | NOT RUN | GPU를 쓰지 않음 |
| Vulkan/RGB | NOT RUN | 렌더링 없음 |
| 정책 rollout | NOT RUN | 새 실행 없음 |
| 상태+정책 복원 | NOT RUN | 새 실행 없음(Stage 2B 기록의 검사 수만 다시 셈) |
| 개입 행동 영상 | NOT RUN | 새 영상 없음. 기존 8개 외의 사례는 영상으로 확인하지 않음 |
| seed split | PASS (변경 없음) | Stage 2B 장면 24개만 읽음. 새 장면 0 |
| 데이터 완료 상태 | PASS | `run_status.json`: COMPLETED_POSTHOC_OFFLINE_DIAGNOSTIC, trace를 쓸 수 없는 block 0 |
| 원본 불변 | PASS | 입력 123개 hash가 세 번 모두 manifest와 같음 |
| 과학적 가설 검증 | NOT RUN | 기술통계와 관찰만. 구간·p-value 없음 |

테스트 통과는 분석 코드가 규칙대로 동작한다는 뜻이고, 로봇 연구 가설과는 무관하다.

## 관측 결과 A — 원본에서 다시 센 값

원본 episode 수: 장면 24개. 분기 실행 수: 새 실행 0(읽은 것: block 336, continuation 1,344).

| 항목 | 다시 센 값 | Stage 2B 보고서 |
|---|---|---|
| 장면 / snapshot / block / continuation | 24 / 84 / 336 / 1,344 | 같음 |
| 성공 block (C0 / C1 / F / R) | 93 / 98 / 94 / 39 | 같음 |
| 여섯 쌍의 네 범주 건수 | `pair_summary.csv`와 전부 일치 | 같음 |
| 장면 평균 차이 여섯 개 | `summary.json`과 1e-12 이내 | 같음 |
| 빠진 checkpoint | 12 (step 92: 4, step 132: 8), 모두 trunk 조기 성공 | 같음 |
| trunk 대 C0 재현 / 재실행 / queue 구간 일치 | 84/84, 12/12 (C0 6, C1 6), 336/336 | 같음 |
| 원본 파일 대 서버 hash 목록 | 72개 일치, 장면 JSON 사본 24개도 원본과 같음 | — |
| 저장된 온라인 입력 | 파일 24개, snapshot 84묶음, 구조 일치. 모델에는 쓰지 않음 | — |

R 관련 집합(Rescue/Harm 분자와 분모에 해당):

| 집합 | block | snapshot | 장면 |
|---|---:|---:|---:|
| 1. C0 성공, R 실패 | 69 | 26 | 13 |
| 2. C0 실패, R 성공 | 15 | 9 | 5 |
| 3. C0 성공, F 성공, R 실패 (1의 부분집합) | 53 | 21 | 11 |
| 4. C0 실패, F 실패, R 성공 (2의 부분집합) | 12 | 8 | 5 |
| 5. C0·C1·F 실패, R 성공 (4의 부분집합) | 8 | 5 | 4 |
| 6a. 둘 다 성공 | 24 | 11 | 9 |
| 6b. 둘 다 실패 | 228 | 64 | 18 |

trunk 성공 장면(8개, block 80개)과 위 집합은 같은 것이 아니다.

- 집합 3의 53개 중 48개가 trunk 성공 장면에, 5개가 trunk 실패 장면(10035, 10041, 10043)에 있다.
- trunk 성공 장면의 block 80개 중 C0가 실패한 것이 4개, R이 성공한 것이 19개다.
- 집합 1의 69개는 trunk 성공 장면에 57개(8개 장면), trunk 실패 장면에 12개(5개 장면)다.
- 집합 2의 15개는 모두 trunk 실패 장면에 있다.

## 관측 결과 B — snapshot 단위

snapshot 84개에서 arm별 성공 수(4회 중)의 분포:

| 성공 수 | 0 | 1 | 2 | 3 | 4 |
|---|---:|---:|---:|---:|---:|
| C0 | 54 | 3 | 8 | 2 | 17 |
| C1 | 53 | 5 | 3 | 5 | 18 |
| F | 51 | 5 | 8 | 7 | 13 |
| R | 66 | 8 | 4 | 1 | 5 |

같은 snapshot 안에서 성공 수의 차이(snapshot 수):

| 차이 | −4 | −3 | −2 | −1 | 0 | +1 | +2 | +3 | +4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| R − C0 | 9 | 4 | 7 | 5 | 52 | 3 | 3 | 0 | 1 |
| F − C0 | 0 | 0 | 4 | 7 | 62 | 7 | 3 | 1 | 0 |
| C1 − C0 | 0 | 0 | 0 | 3 | 76 | 3 | 1 | 1 | 0 |

- R이 C0보다 2회 이상 많이 성공한 snapshot은 4개이고 두 장면에 있다: 10035의 step 92(0 대 4), 10043의 step 52(0 대 2), step 92(2 대 4), step 132(2 대 4).
- R이 C0보다 2회 이상 적게 성공한 snapshot은 20개, 11개 장면이다.
- F가 C0보다 2회 이상 많은 snapshot은 4개(3개 장면), 2회 이상 적은 snapshot도 4개(3개 장면)다.
- 0/4나 4/4는 그 상태의 성공 확률이 0이나 1이라는 뜻이 아니다. 네 번의 실현이다.

## 관측 결과 C — 반복을 하나 남기는 선택 일관성 진단

snapshot마다 repeat 하나를 남기고 나머지 세 repeat의 성공 수로 arm을 골라, 남겨둔 repeat에서 그 arm의 결과를 읽었다. 동률은 C0, F, R 순이다. C1은 고를 수 없고 기준으로만 적었다.

이 진단은 과거 분기 결과를 입력으로 쓰므로 실제로 쓸 수 있는 선택기가 아니다. 한 장면의 행은 서로 관련돼 있고 repeat가 여러 번 다시 쓰인다. 규칙당 336행은 독립 표본 수가 아니다.

| 규칙 | 성공 행 (336 중) | 행 평균 | 장면 평균 | − 항상 C0 (장면) | C0보다 높은 / 낮은 / 같은 장면 | − 항상 C1 (장면) | C1보다 높은 / 낮은 / 같은 장면 |
|---|---:|---:|---:|---:|---|---:|---|
| 항상 C0 | 93 | 0.277 | 0.362 | — | — | −0.014 | 1 / 4 / 19 |
| 항상 C1 (대조) | 98 | 0.292 | 0.376 | +0.014 | 4 / 1 / 19 | — | — |
| 항상 F | 94 | 0.280 | 0.350 | −0.012 | 6 / 6 / 12 | −0.026 | 5 / 7 / 12 |
| 항상 R | 39 | 0.116 | 0.127 | −0.235 | 4 / 11 / 9 | −0.249 | 3 / 11 / 10 |
| `{C0, F}` | 99 | 0.295 | 0.375 | +0.013 | 3 / 2 / 19 | −0.001 | 4 / 5 / 15 |
| `{C0, R}` | 102 | 0.304 | 0.385 | +0.023 | 2 / 0 / 22 | +0.010 | 3 / 2 / 19 |
| `{C0, F, R}` | 107 | 0.319 | 0.396 | +0.034 | 4 / 2 / 18 | +0.020 | 5 / 4 / 15 |

C1과의 차이는 "난수를 제거한 효과"가 아니다. C1은 같은 Continue 규칙의 다른 난수 실현일 뿐이다.

각 규칙이 C0가 아닌 arm을 고른 행과, 남겨둔 repeat에서 C0와 비교한 결과:

| 규칙 | 고른 arm | 행 | snapshot | 장면 | 실패→성공 | 성공→실패 | 둘 다 성공 | 둘 다 실패 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `{C0, F}` | F | 38 | 13 | 9 | 9 (장면 10031, 10042, 10043) | 3 (10032, 10040, 10043) | 7 | 19 |
| `{C0, R}` | R | 26 | 8 | 4 | 10 (10035, 10043) | 1 (10043) | 4 | 11 |
| `{C0, F, R}` | F | 36 | 12 | 9 | 8 | 2 | 7 | 19 |
| `{C0, F, R}` | R | 20 | 7 | 4 | 8 (10035, 10043) | 0 | 4 | 8 |

장면별로 C0와 달라진 곳(나머지 18개 장면은 세 규칙 모두 C0와 같다):

| 장면 | 항상 C0 | `{C0,F}` | `{C0,R}` | `{C0,F,R}` |
|---|---:|---:|---:|---:|
| 10031 | 0.000 | 0.125 | 0.000 | 0.125 |
| 10032 | 0.875 | 0.750 | 0.875 | 0.750 |
| 10035 | 0.250 | 0.250 | 0.500 | 0.500 |
| 10040 | 0.063 | 0.000 | 0.063 | 0.000 |
| 10042 | 0.000 | 0.125 | 0.000 | 0.125 |
| 10043 | 0.313 | 0.563 | 0.625 | 0.813 |

`{C0,F,R}` − `{C0,F}`: 선택이 달라진 행은 20개(장면 10031, 10034, 10035, 10043의 snapshot 7개)다. 그중 9행이 더 좋았고 1행이 더 나빴고 10행은 결과가 같았다. 더 좋았던 9행은 10035의 step 92(4행)와 10043의 step 92·132(5행)에서 나왔다.

읽는 법:

- 같은 snapshot의 다른 repeat 결과로 Retreat를 골랐을 때, 남겨둔 repeat에서 실패→성공이 10행, 성공→실패가 1행이었다. 10행은 두 장면의 것이다.
- 같은 방식으로 Refresh를 골랐을 때는 9행 대 3행이었다(각각 3개 장면). `{C0,F}`의 장면 평균은 항상 C1과 0.001 차이다.
- 이 수치는 "같은 상태의 결과를 이미 아는" 규칙의 것이다. 행동할 때 관측만으로 그 상태를 알아볼 수 있는지는 재지 않았다.
- 전체 이득은 24개 장면 중 2~4개 장면에 의존한다.
- 이 진단이 잘 됐다고 해서 관측만으로 그 상태를 알아볼 수 있다는 뜻은 아니다. 잘 안 됐다고 해서 선택기가 불가능하다는 뜻도 아니다.

## 관측 결과 D — Retreat 전후의 사건

336개 block 전부에 같은 추출기를 적용했다. trace를 쓸 수 없는 block은 0개다. Retreat 종료 시각은 block마다 기록된 step 수(23~30)이고, R arm의 첫 추론 step과 336개 모두 일치한다. Retreat 도중에 끝난 block은 없다.

### Retreat 구간에서 일어난 일

| 관찰 (336개 block) | 건수 |
|---|---:|
| checkpoint에서 큐브를 쥐고 있었음 | 108 (16개 장면) |
| 그중 Retreat 중 파지 flag가 꺼짐 | 0 |
| Retreat 중 들어올림 조건을 잃음 | 0 |
| Retreat 중 정렬 조건을 잃음 | 0 |
| Retreat 중 큐브 B가 2 mm 이상 이동 | 0 |
| Retreat 중 접근 조건(TCP–큐브 3 cm)을 잃음 | 4 (1개 장면, 모두 "둘 다 실패") |

쥐고 있던 108개에서는 큐브가 gripper와 함께 약 4.5 cm 올라갔다. 쥐지 않은 경우 큐브는 움직이지 않았다.

### 집합별 (R 대 C0)

| | R이 나쁨 (C0=1, R=0) | R이 좋음 (C0=0, R=1) | 둘 다 성공 | 둘 다 실패 |
|---|---:|---:|---:|---:|
| block / snapshot / 장면 | 69 / 26 / 13 | 15 / 9 / 5 | 24 / 11 / 9 | 228 / 64 / 18 |
| checkpoint: step 12 / 52 / 92 / 132 | 26 / 30 / 13 / 0 | 4 / 3 / 6 / 2 | 11 / 6 / 5 / 2 | 55 / 57 / 56 / 60 |
| checkpoint에서 큐브를 쥠 | 42 | 10 | 13 | 43 |
| 제어 반환 시 남은 step (중앙값) | 123 | 82 | 125 | 84 |
| C0가 checkpoint 이후 쓴 step (중앙값) | 49 | 108 (실패) | 68 | 108 (실패) |
| R의 종료 | 모두 step 200 | 모두 성공 (반환 후 19~160 step) | 모두 성공 | 모두 step 200 |
| 반환 뒤 파지 flag가 꺼짐 | 61 | 15 | 24 | 100 |
| 그때 큐브가 B 위(on flag) | 4 | 12 | 16 | 7 |
| 반환 뒤 한 번도 쥐지 못함 | 8 | 0 | 0 | 108 |
| 반환 뒤 큐브에 다시 접근하지 못함 | 6 | 0 | 0 | 105 |
| 쥔 채로 step 200 | 30 | 0 | 0 | 59 |
| 반환 뒤 정렬 조건까지 갔으나 실패 | 12 | 0 | 0 | 11 |
| 반환 뒤 큐브 B가 2 mm 이상 이동 | 28 | 1 | 0 | 35 |
| 종료 시 큐브 A–B 수평 거리 (중앙값) | 7.9 cm | 1.2 cm | 1.3 cm | 11.4 cm |

큐브 B의 이동은 R arm에서만 셌다. 다른 arm과 비교하지 않았고, 힘이나 접촉 기록이 없으므로 충돌이라고 하지 않는다.

### R이 나빴던 69개의 순서

**남은 시간.** 69개 중 66개(12개 장면)에서, 제어를 돌려받은 시점에 남은 step이 같은 block의 C0가 checkpoint부터 성공까지 쓴 step보다 많았다. 이것은 step 수의 비교일 뿐이다. 서로 다른 분기의 step 수를 견준 것이어서, 시간 경과의 영향이 있는지 없는지는 이 기록으로 가릴 수 없다.

**checkpoint에서 큐브를 쥐고 있던 42개 (10개 장면, step 52에 29개, step 92에 13개).**

1. Retreat 동안 파지가 유지됐고 큐브가 함께 올라갔다(42/42).
2. 제어를 돌려받은 뒤 파지 flag가 꺼졌다(42/42). 꺼진 step의 gripper 명령은 모두 "열기"였다. 반환 후 10~71 step 뒤이고(중앙값 21), 14개는 정확히 10 step 뒤다.
3. 그 직전 큐브는 약 4 cm 떠 있었고, 큐브 B에서 수평으로 2 cm 이내였던 경우는 0개다(중앙값 7.7 cm).
4. 같은 42개 block의 C0는 42개 모두 큐브 B 위에서 flag가 꺼졌다(수평 2 cm 이내 33개, 중앙값 1.2 cm).
5. 그 뒤 25개는 다시 쥔 상태로, 17개는 쥐지 않은 상태로 step 200에 끝났다. 정렬 조건까지 간 것은 7개다.

**checkpoint에서 쥐고 있지 않던 27개 (12개 장면, 26개가 step 12).**

1. step 12는 접근 중이다(TCP–큐브 거리 중앙값 15.6 cm). Retreat는 TCP만 올렸고 큐브는 그대로였다.
2. 반환 뒤 19개는 큐브를 쥐었고(반환 후 17~149 step, 대부분 39~58 step), 8개는 끝까지 쥐지 못했다. 6개는 큐브 3 cm 이내로 다시 가지 못했다.
3. 같은 block의 C0는 checkpoint 이후 70~185 step에 성공했다(대부분 74~94 step).

### R이 좋았던 15개의 순서

- 10개(장면 10035와 10043, step 52·92·132)는 checkpoint에서 큐브를 쥐고 있었다. Retreat 뒤 정책이 gripper를 열었을 때 8개는 큐브 B 위였다(수평 2 cm 이내). 반환 후 18~39 step 뒤다.
- 같은 10개 block의 C0는 flag가 꺼질 때 큐브 B에서 수평 2 cm 이내였던 것이 1개뿐이다(중앙값 2.9 cm). C0는 15개 모두 step 200에 끝났다.
- 나머지 5개(step 12의 4개와 step 52의 1개)는 쥐지 않은 상태였고, 반환 후 92~160 step에 성공했다.

### 사례 (표를 만든 뒤 고정 규칙으로 고름)

집합과 관찰 flag마다 scene/checkpoint/repeat 오름차순 첫 block, 그리고 패턴이 repeat 2개 이상에서 나온 첫 snapshot이다. 목록은 `case_index.csv`, step별 표는 `case_timeseries/`에 있다. 영상은 없다.

| 장면 / step / repeat | 고른 이유 | trace에서 읽은 것 |
|---|---|---|
| 10024 / 52 / 0 | R이 나쁨, 파지 유지된 첫 사례 | C0는 step 85에 큐브 B 위(수평 1.6 cm)에서 놓고 step 87에 성공. R은 step 77에 반환, step 87에 수평 8.2 cm 지점에서 열기 명령, 큐브가 바닥으로 내려감 |
| 10024 / 52 / 3 | R이 나쁨, 쥔 채 step 200 | `case_timeseries/scene10024_t52_r3.csv` |
| 10024 / 12 / 0 | R이 나쁨이 repeat 2개(0, 1)에서 나온 첫 snapshot | `case_timeseries/scene10024_t12_r0.csv` |
| 10032 / 12 / 0, 2 | R이 나쁨, 반환 뒤 쥐지 못함 / 접근하지 못함 | 같은 폴더 |
| 10035 / 92 / 0 | R만 성공이 repeat 4개 모두에서 나온 첫 snapshot | 같은 폴더 |
| 10031 / 12 / 2, 10034 / 52 / 1 | R이 좋음의 첫 사례들 | 같은 폴더 |

## 기존 결과만으로 말할 수 없는 것

- **왜 Retreat 뒤에 큐브 B가 아닌 곳에서 gripper를 열었는가.** 관찰은 "그 위치에서 열기 명령이 나왔다"까지다. 정책이 위치보다 경과 시간이나 동작 순서를 따랐는지, 4.5 cm 옮겨진 상태가 학습 자료에 드물어서인지, 다른 이유인지는 이 기록으로 가릴 수 없다.
- **시간 소모와 위치 변화 중 무엇이 작용했는가.** Retreat는 약 25 step을 쓰고 동시에 4.5 cm를 옮긴다. 둘을 따로 바꾼 실행이 없다.
- **파지 flag가 꺼진 것이 "떨어뜨림"인가.** flag와 명령, 위치만 있다. 영상이나 접촉 기록으로 확인하지 않았다.
- **두 장면(10035, 10043)에서 Retreat가 왜 도움이 됐는가.** C0가 큐브 B에서 어긋난 곳에 놓고 R은 맞는 곳에 놓았다는 관찰까지다.
- **관측만으로 이런 상태를 구별할 수 있는가.** 이 단계는 관측 기반 모델을 만들지 않았다.
- **새 장면에서도 같은가.** 24개 장면, 한 정책(qualification 37/100, 기준 미달), 한 Retreat 설정에서의 결과다.

## 변경과 실패

변경한 파일과 이유:

- `scripts/21_stage2c_offline_review.py` (신규): manifest, 재검산, snapshot 표, 선택 일관성 진단, 사건 추출, 입력 불변 확인.
- `scripts/22_stage2c_release_reading.py` (신규, **주 결과를 본 뒤 추가**): 쥐고 있던 block에서 파지 flag가 처음 꺼진 step의 명령과 큐브 위치. 사전에 정한 분석이 아니다.
- `tests/test_stage2c_offline_review.py` (신규, 29개): 잘못된 join·repeat 누락·중복·잘못된 값의 거부, 남겨둔 결과가 선택에 영향을 주지 않음, 동률 순서와 C1 선택 불가, 결과가 같으면 이득 0, 일관된 우위와 한 번의 우연 구분, 장면 동일 가중, 시간축, 원인 라벨이 자동으로 붙지 않음, 입력 불변.
- `amendments/AMENDMENT_007_stage2c_offline_analysis.md`, `README.md`, `.gitattributes`.
- `src/hci`의 runtime 코드와 Stage 2B 결과 파일은 수정하지 않았다.

Codex 검토에서 3건을 받아 manifest를 쓰기 전에 모두 반영했다: Retreat 도중에 끝난 분기를 "반환 뒤 성공"으로 부르지 않음, trace를 쓸 수 없는 block을 사례로 고르지 않음, 출력 경로가 입력 위치 안이면 거부.

실패한 명령 및 원인: 없음. 테스트 작성 중 한 번, 출력 경로 검사를 넣은 뒤 테스트용 임시 경로가 입력 위치 안으로 잡혀 실패했고 테스트의 경로를 고쳤다(실제 자료를 읽기 전).

수행하지 않은 일: GPU 대여, simulator 실행, 정책 추론, 학습, 렌더링, qualification, calibration, 최종 Test, 예측기·선택기 학습, horizon 연장이나 대기 대조 실험. `RUN_FROZEN_TEST` 토큰은 만들지 않았다.

보고서의 일부 수치(snapshot 차이 분포, "66/69", 쥐지 않은 27개의 세부, 사례 설명)는 산출물 CSV를 읽어 센 것이다. 분석 스크립트의 요약 파일에 그대로 들어 있지는 않다.

## 다음 단계

명세 8절의 **C**에 해당한다고 판단한다(개발 판단이며 통계적 판정이 아님).

- A(Refresh 선택기)가 요구하는 "반복을 바꿔도 여러 장면에서 유지"는 이 진단에서 3개 장면, 9행 대 3행이었다.
- B(다중 개입 선택기)가 요구하는 Retreat의 반복 간 유지는 두 장면(10행 대 1행)에서만 나왔다.
- 두 수치 모두 같은 상태의 결과로 고른 규칙의 것이고, 관측 기반 선택기의 성능이 아니다.
- C의 조건인 "극소수 장면 의존"과 "관측된 특정 부작용"이 둘 다 있다. 부작용은 Retreat 뒤 정책이 큐브 B가 아닌 곳에서 gripper를 연 것(42/42)이다.

필요 작업: 한 가지로 좁혀 제안한다. **"Retreat에서 위치 변화와 시간 경과 중 무엇이 문제인지"를 가르는 최소 대조 실험.** 지금의 Retreat(위로 5 cm)에 두 arm을 더한다.

- 변경 후보 하나: 더 짧은 Retreat(예: 2 cm). 겨냥하는 문제는 옮겨진 위치에서 놓는 것이다.
- 대조: 제자리 유지(이동 0, 같은 step 수만 소비, gripper 유지). 시간 경과만의 효과를 본다.
- 선택기 학습과 기존 Retreat의 재학습은 하지 않는다.

예상 계산량의 산출 근거: 새 train 장면 12개 × checkpoint 4 × repeat 4 × arm 4(C0, 현재 R, 짧은 R, 유지) = 최대 768 continuation. Stage 2B(1,344회와 audit에 1시간 13분, $0.30)에 비례하면 약 45분~1시간, $0.2~0.3이다. 현재 가격의 보장은 아니다.

동결/승인이 필요한 사항:

- 위 실험을 할지, 한다면 짧은 Retreat의 거리와 유지 arm의 step 수 규칙을 실행 전에 정해야 한다. 새 amendment가 필요하다.
- 이 제안은 사후 관찰에서 나왔다. 다음 실험도 개발 실험이다.
- train에는 쓰지 않은 장면이 72개 남아 있다.
- 추가 실행, GPU 대여, 학습은 자동으로 시작하지 않았다.

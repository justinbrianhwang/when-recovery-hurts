# 설정/프로토콜 변경 기록 004 — Stage 1D 재실행 (결함 궤적 전체 제외·대체, D500-clean)

기존 버전: v0.3 + AMENDMENT_001~003 (저장소 HEAD `d562701`)
새 버전: v0.3 + AMENDMENT_001~004
변경 시점: 2026-11-14 (UTC). D500-clean 정책의 어떤 평가 결과도 나오기 전. GPU 대여 전.
변경 사유: `stage1d_001`에서 학습 궤적 500개 중 4개가 "직전 원본 episode의 실패한 재생 + reset 항목(dummy action 1개) + 채택된 episode"로 이어 붙어 있음을 확인하고 학습을 중단했다.
사용자가 `prompts/STAGE1D_CLEAN_REBUILD_AND_RETRAIN_AGENT_PROMPT_v0_2.md`로 **결함 궤적 전체 제외 후 대체**를 결정했다. 앞부분 절단이나 부분 복구는 하지 않는다.
허용 오차, 모델, horizon, 성공 판정, Retreat, 학습·선택·qualification 규칙은 바꾸지 않는다.

## 1. AMENDMENT_003에서 대체되는 것

AMENDMENT_003과 `STAGE_REPORT_stage1d_001.md`의 본문은 고치지 않는다. 다음 세 가지만 이 문서가 대체한다.

| AMENDMENT_003 | 이 문서 |
|---|---|
| D100 100개를 전부 포함 | 결함 1개(`traj_30`, 원본 31)를 제외하고 대체 |
| `traj_30`은 "horizon을 넘는 시연"으로 보고 유지 | 이어 붙은 궤적으로 확인되어 제외 |
| 변경 변수는 학습 시연 집합(수)뿐 | **데이터 수와 데이터 구성이 함께 바뀐다.** D100 대비 "시연 수만 바꾼 실험"이라고 부르지 않는다 |

## 2. 기존 결과의 상태 표기 (수치는 바꾸지 않음)

- Stage 1B qualification 4/100: 그대로. 표기는 "알려진 결함 시연 1개가 포함된 D100으로 학습한 legacy 정책".
- Stage 1C 진단: 그 정책에 대한 기록으로 남긴다. 데이터 부족만이 원인이라는 설명은 붙이지 않는다.
- `stage1d_001`: `ABORTED_AFTER_DATA_INTEGRITY_FAILURE` / `INCOMPLETE`. 학습은 약 2천 update 실행되었으므로 NOT RUN이 아니다. 후보 평가·qualification·진단·audit은 NOT RUN.
- 오염된 D500 파일, D100, 새 변환 파일, 로그, manifest는 삭제하지 않는다 (3.29 GB 묶음 해시 `66692866…`).

## 3. 내용 무결성 검사 (검사 추가)

`src/hci/integrity.py`, `scripts/17_stage1d_clean.py audit`. 하나의 학습 궤적은 하나의 원본 episode에서 와야 한다. shape와 마지막 success만으로는 통과시키지 않는다.

| 신호 | 종류 | 뜻 |
|---|---|---|
| `internal_terminal` | 확정 | success 없이 terminated가 켜진 step (recorder가 reset 항목에 남기는 표시) |
| `truncated_restart` | 확정 | 시간 제한 flag가 true에서 false로 돌아감 (step 카운터가 초기화됨) |
| `first_obs_not_source` | 확정 | 첫 관측의 관절 각도가 선언된 원본 episode의 첫 상태와 다름 |
| `first_state_not_source` | 확정 | 저장된 첫 env state가 선언된 원본 episode의 것이 아님 |
| `kinematic_jump` | 선별 | 한 step에 큐브 B 2 cm 또는 관절 0.3 rad 초과 |
| `gripper_not_binary` | 선별 | ±1이 아닌 그리퍼 명령 (원본 전문가 명령이 ±1뿐임을 원본 1000개에서 확인한 경우에만 사용) |
| `length_not_source` | 선별 | 길이가 원본과 다름 (현재 변환 방식에서 관측된 기대 관계는 "같음") |
| `cube_path_not_source` | 선별 | 같은 step에서 큐브 A가 원본 episode의 위치와 5 cm 넘게 다름 |

- `cube_path_not_source`는 Codex 검토에서 "flag와 길이가 정상으로 보이는 접합"을 놓칠 수 있다는 지적을 받고 추가했다. 5 cm는 보유한 545개에서 잰 값 사이에 둔 선별 기준이다 (reset이 없는 궤적의 최대 3.4 cm, reset이 있는 궤적의 최소 15.7 cm). 물리 법칙이 아니다.
- 구성 단계는 저장된 검사 결과를 그대로 믿지 않는다. 입력에서 검사를 다시 계산해 저장된 결과 파일과 byte 단위로 같아야 진행한다.
- 판정: 확정 신호가 하나라도 있으면 `FAIL`, 선별 신호만 있으면 `QUARANTINE`, 없으면 `PASS`. `FAIL`과 `QUARANTINE`은 학습에서 제외한다.
- 알려진 4개 ID를 blacklist로 쓰지 않는다. 제외는 검사 결과로만 정해진다.
- 내용의 출처도 따로 본다: 저장된 첫 상태가 원본 1000개 중 어느 episode와 일치하는지, reset 직후 장면이 어느 episode의 첫 상태인지.
- `env_states[0]`이 원본 두 번째 상태라는 기존 사항은 metadata anomaly로 따로 둔다. 그래서 `first_state_not_source`는 원본 **두 번째** 상태와 비교한다. 초기 장면 기준은 계속 원본 첫 상태다.
- **success flicker는 결함으로 보지 않는다** (정보로만 기록). 성공 판정에 정지 조건이 있어, 같은 episode 안에서 success가 켜졌다 꺼졌다 다시 켜질 수 있다. 처음 구현은 이것을 확정 신호로 잡아 정상 궤적 1개(옛 학습 ID 339)를 FAIL로 분류했다. 그 궤적은 나머지 신호가 모두 깨끗하고 큐브 위치도 연속이어서 규칙을 고쳤다. 정책 결과가 나오기 전의 변경이다.
- `PASS`와 `clean`은 "선언한 검사를 통과했다"는 뜻이다. 결함이 전혀 없다는 보장이 아니다.

## 4. 데이터 구성 (데이터 변경)

- 검사 대상: 보유한 변환 궤적 전체 545개 (기존 변환 129 + 새 변환 416).
- 학습 집합 `train_d500_clean_v1`: `stage1d_001`의 500개 중 PASS인 궤적은 보존된 변환 파일에서 그대로 복사하고, 제외된 자리에는 대체 궤적을 같은 ID에 넣는다.
- 대체 후보: 새 변환에서 채택하지 않았던 성공분 16개 중 PASS이고 보호 집합이 아닌 것. **원본 episode ID 오름차순**으로 쓴다. 첫 번째(r1)는 D100의 결함 자리, 나머지는 새 시연의 결함 자리(옛 학습 ID 오름차순).
- `D100_clean_v1` = D500-clean의 ID 0~99 (기존 99 + r1). manifest로만 정의한다. **D100-clean 정책은 이번에 학습하지 않으며 그 성능을 주장하지 않는다.**
- 후보가 모자라면 절단·복제·보호 집합 전용·결함 재포함으로 채우지 않고 중단한다.
- 보호 집합 29개는 판정과 무관하게 계속 학습 금지다.
- 유지되는 궤적은 dataset별 hash가 `stage1d_001` 기록과 같아야 한다. 다시 렌더링하지 않는다.

## 5. 학습 시작 차단 (실행기 변경)

`scripts/stage1d/run_clean.sh`와 `17_stage1d_clean.py gate`. 다섯 층을 따로 기록한다: `structural_validation`, `source_and_split_validation`, `episode_content_integrity`, `sampled_action_replay`, `policy_io_parity`.

학습 직전에 다음을 다시 읽고, 하나라도 어긋나면 학습 프로세스를 시작하지 않는다.

- 검증 보고서의 HDF5·JSON·manifest·검사기 소스 hash가 실제 입력과 같은가
- 500개 모두 내용 검사 PASS이고 미해결 후보가 없는가
- 다섯 층이 모두 PASS인가 (exit code 0이나 마지막 success는 이를 대신하지 않는다)
- `policy_qualify_B` ledger가 비어 있는가
- 예산: 지금까지 쓴 금액 + 단가 × 학습 시간 상한 + 백업 여유분이 누적 상한 안인가

중단은 trainer의 process group에만 신호를 보낸다 (`stage1d_001`에서 중단 명령이 SSH 세션까지 끊은 일의 재발 방지). start/stop/reason/exit를 status에 남긴다. 중단된 run, probe 가중치, D100 가중치에서 재개하지 않는다.

서버 재검사: 전문가 action replay 14개(대체 전부 + 유지된 기존 5 + 유지된 새 5, 원본 ID 순 등간격), 새 dataset 경로의 관측·target 정렬(대체 전부 + 기존 3 + 새 3).

## 6. 변환기 (경고와 guard만 추가, 수정은 분리)

- `scripts/02_prepare_demos.sh`와 `scripts/stage1d/run.sh data`는 `HCI_CONVERSION_ACK=content-integrity-check-required` 없이는 실행을 거부한다. 변환 결과는 내용 검사를 거치기 전에는 학습에 쓰지 않는다.
- upstream 수정이나 재변환은 이번에 하지 않는다. 이번 학습 입력은 보존된 정상 payload를 골라 재구성한 것이다.
- 고정 commit 코드에서 읽은 경로(정적 확인, 수정 후 재실행으로 검증한 것은 아님): `replay_trajectory.py`가 `save_on_reset=False`로 recorder를 만들고, 실패한 재생에서는 `flush_video(save=False)`만 호출한다.

## 7. 예산

Stage 1D 누적 상한 US$2.00을 유지한다. `stage1d_001` 사용분 약 $0.24를 포함하므로 남은 한도는 약 $1.76이다. 새로 $2를 배정하는 것이 아니다.

## 8. 이 단계로 할 수 없는 결론

- 약 1%의 오염이 4/100 실패의 원인이다.
- 정제 효과와 시연 수 확대 효과를 각각 입증했다.
- 남은 데이터에 결함이 전혀 없다.
- qualification이 최종 Test다.

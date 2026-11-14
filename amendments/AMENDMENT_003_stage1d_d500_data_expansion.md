# 설정/프로토콜 변경 기록 003 — Stage 1D (시연 500개 확장, 기본 정책 재확보)

기존 버전: v0.3 + AMENDMENT_001 + AMENDMENT_002 (저장소 HEAD `ae2f721`)
새 버전: v0.3 + AMENDMENT_001 + AMENDMENT_002 + 이 문서
변경 시점: 2026-11-14 (UTC). D500 정책의 어떤 평가 결과도 나오기 전. GPU 대여 전.
변경 사유: Stage 1B 정책은 qualification 4/100이었고, Stage 1C 진단에서 학습 장면 6/10 대 학습에 쓰지 않은 장면 0/10이 관측되었다.
사용자가 `prompts/STAGE1D_D500_CONTROLLED_TRAINING_AGENT_PROMPT_v0_1.md`로 시연 확대 단일 재학습을 결정했다.
허용 오차(`state_atol`, `action_atol`, `observation_atol`), 모델, horizon, 성공 판정, Retreat는 바꾸지 않는다.

## 1. 사전 규칙과의 관계 (규칙 변경, 숨기지 않음)

- AMENDMENT_002의 운영 기준은 "qualify 10~49면 시연 500개 조건부 두 번째 학습, 0~9면 확대 없이 진단"이었다.
  Stage 1B 결과 4/100은 0~9 구간이므로 그 규칙만으로는 500개 학습이 시작되지 않는다. 규칙대로 Stage 1C 진단을 먼저 수행했다.
- 이번 500개 학습은 **진단 결과를 본 뒤 사용자가 새로 내린 결정**이다. 사전 규칙의 자동 발동이 아니다.
- `policy_qualify_B`(50300~50399)는 "두 번째 학습의 선택 후보 하나를 한 번 평가"하는 용도로 선언되어 있었고 아직 사용하지 않았다. 이번 학습이 그 두 번째 학습이다.
- 데이터 부족이 실패의 유일한 원인이라고 확정한 것은 아니다. update 수를 100,000으로 고정하므로 시연당 평균 재노출 횟수는 약 1/5이 된다. 순수한 단일 변수 인과 실험이라고 부르지 않는다. 학습 seed는 1개다.

## 2. 학습 데이터 구성 (데이터 변경)

| 집합 | 내용 | 수 |
|---|---|---:|
| D100 (`old_train`) | Stage 1B가 실제로 학습한 변환 궤적 `traj_0..99`. 보존된 변환 파일에서 복사한다. 다시 렌더링하지 않는다. | 100 |
| Dnew400 (`new_train`) | 원본 파일 순서로 Stage 1B 변환이 읽은 마지막 episode 바로 다음(원본 위치 130)부터 420개를 같은 명령·옵션으로 변환하고, 재생 성공 필터를 통과한 처음 400개 | 400 |
| D500 | D100 ∪ Dnew400. 학습 전용 파일의 `traj_0..499` | 500 |
| V20 (`protected_dev`) | Stage 1B held-out `traj_100..119`. 학습 제외 | 20 |
| `protected_remainder` | Stage 1B에서 변환했지만 쓰지 않은 `traj_120..128`. 학습 제외 | 9 |

- "앞에서 500개"를 읽지 않는다. 그렇게 하면 V20이 학습에 들어간다. upstream loader는 `traj_<id>`를 숫자 순으로 정렬해 앞의 N개를 읽으므로, **학습할 500개만 담은 파일**을 만들어 `--num-demos 500`으로 넘긴다.
- 동일성은 파일 내부 번호가 아니라 원본 episode ID·seed·reset 인자·원본 첫 상태 fingerprint·dataset별 payload hash로 판단한다 (`scripts/16_stage1d_data.py`).
- 창 420개에서 400개가 안 나오면 창을 조용히 늘리지 않고 중단한다. 400개를 넘는 성공분은 `unused`로 기록하고 쓰지 않는다. 정책 성능을 본 뒤 시연을 추가·교체하지 않는다.
- 이미 기록된 중복: restore audit seed 100~104는 Stage 1B 변환 장면이다 (100, 101은 D100, 102~104는 V20). 이 split은 결정론 검사용이며 성능 평가가 아니다. 삭제하지 않고 그대로 둔다. 새 시연이 선언된 어떤 split과 겹치면 중단한다.

## 3. Stage 1C에서 발견한 두 데이터 사항의 처리

- **첫 env state**: 고정 commit의 변환기는 변환 파일의 `env_states[0]`에 원본 두 번째 상태를 쓰고, 첫 관측은 원본 첫 상태에서 렌더링한다. 학습에 쓰는 관측·행동 배열은 바꾸지 않는다. 초기 장면 fingerprint와 진단의 초기화에는 **원본 파일의 첫 상태**를 쓴다 (`initial_state_source` 기록, 회귀 테스트 `tests/test_stage1d_data.py`). upstream 변환기 수정이나 state 재작성은 하지 않는다.
- **horizon을 넘는 시연**: D100의 `traj_30`(첫 성공 201 step)을 삭제하지 않는다. horizon 200과 성공 필터를 유지한다. 새 시연에서 첫 성공이 200을 넘는 수·비율·분포를 학습 전에 manifest에 기록한다.

## 4. 학습·선택·qualification (값은 Stage 1B와 같음)

- StackCube-v1, 기존 RGB Diffusion Policy와 관측 whitelist, 새 초기화(warm start 없음, 프로브 모델 폐기), 학습 seed 1, update 100,000, batch 256, lr 1e-4, obs/act/pred 2/8/16, DDPM 100 step, horizon 200, `success_once_absorbing`. run ID `stack_1d_s1_d500`.
- 후보: iteration tag 60000 / 80000 / 100000의 EMA. `policy_select` 50개에서 `success_once` 최대, 동률이면 더 뒤 iteration. `success_at_end`는 보조 지표.
- Stage 1B와 다른 운영 사항: 세 후보를 **순차** 실행한다 (Stage 1B는 동시 실행). 성공 판정에는 영향이 없고 추론 시간 기록의 조건만 달라진다.
- 선택된 후보의 checkpoint↔export tensor 검사와 새 가중치 I/O parity를 **qualification 전에** 수행한다. 실패하면 qualification을 시작하지 않는다.
- 새 dataset 경로의 관측·target 정렬은 학습 전에 기존 3개 + 새 3개 시연으로 검사한다 (`io_parity_sample`, 위치로만 고름).
- `policy_qualify_B`는 선택된 후보 하나로 한 번만 실행한다. ledger가 두 번째 실행을 거부한다. qualify_A 4/100은 그대로 둔다. A와 B는 서로 다른 장면이므로 합치거나 paired 검정을 하지 않는다.
- 종료 분기: ≥50이면 새 정책으로 복원 audit를 다시 하고 통과 시 원본 2개 분기만 수집. 10~49와 0~9는 추가 학습·재시험 없이 종료.

## 5. 진단 스크립트 (옵션 추가, 기존 동작 변경 없음)

- `scripts/14_scene_rollouts.py`, `scripts/15_offline_action_diag.py`에 `--other-policy` 추가. Stage 1C manifest에 고정된 장면·이력·noise seed는 그대로 쓰고, manifest에 고정된 정책이 아닌 다른 정책을 넣을 수 있게 한다. 시연 파일 hash 검사는 유지한다. 옵션이 없으면 기존과 같다.
- D4(RGB 민감도)는 반복하지 않는다. 관측 경로를 바꾸지 않았기 때문이다.

## 6. Stage 1C 보고서 정정

- `reports/STAGE_REPORT_stage1c_001.md`는 "F-policy와 R-policy 40쌍 중 39쌍의 성공 여부가 같다"고 썼다. 단위가 틀렸다. 정책 실행은 F 20회 + R 20회이고 장면 쌍은 **20쌍**이다. `results/stage1c/scene_rollouts.csv`로 다시 세면 **20쌍 중 19쌍**이 같고 `traj_55`만 다르다.
- CSV와 수치는 바뀌지 않는다. 기존 보고서는 덮어쓰지 않고 이 문서와 Stage 1D 보고서에 정정을 남긴다.

## 7. 예산과 중단

- 상한 US$2.00. 실제 대여 단가로 다시 계산하고, 부족하면 조용히 넘기지 않는다.
- 중단 사유: D100 보존 또는 보호 집합 비중복 검사 실패, 관측·행동 의미 불일치, 새 I/O parity 실패, NaN/Inf, 설명할 수 없는 expert replay 실패, VRAM·디스크·예산 초과 예상, ledger 충돌, 백업 불가.
- 중간 성공률이 낮다는 이유로 100k 일정을 바꾸지 않는다.

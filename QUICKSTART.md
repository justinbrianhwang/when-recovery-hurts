# 실제 실행 순서

모든 명령은 저장소 루트에서 실행합니다.
파일명 `001`은 예시 실행 ID입니다. 기존 폴더를 지우지 말고 재실행 시 `002`처럼 새 ID를 사용합니다.

## A. 서버 준비

Vast 인스턴스를 생성할 때 NVIDIA 드라이버의 `compute,utility,graphics` 기능이 컨테이너에 노출되어야 합니다.
`NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics`는 **컨테이너 생성 시** 설정합니다.
이미 실행 중인 컨테이너 안에서 `export`만 해서는 누락된 호스트 라이브러리가 새로 마운트되지 않습니다.[S6]

```bash
cd when-recovery-hurts
# root로 접속한 경우. 일반 계정이면 필요한 경우에만 sudo를 붙입니다.
bash scripts/00_system_packages.sh
bash scripts/01_bootstrap_vast.sh
source .venv/bin/activate
export MS_SKIP_ASSET_DOWNLOAD_PROMPT=1
export WANDB_MODE=disabled
python -m hci doctor --out runs/doctor_001
bash scripts/06_validate_local.sh validation/server_core_001
```

doctor가 실패하면 멈춥니다. 커널 드라이버를 컨테이너에서 설치하지 마십시오.
`.venv`가 이미 있으면 bootstrap은 중단합니다. 기존 환경을 확인하고 수동 복구하거나 새 작업 폴더를 사용합니다.

## B. 실제 RGB 렌더링과 물리 복원 확인

```bash
python -m hci smoke --config configs/pickcube.yaml --out runs/smoke_pick_001
python -m hci smoke --config configs/stackcube.yaml --out runs/smoke_stack_001
```

`smoke.json`의 `pass`, 카메라 크기, 상태·관측 복원 오차를 확인합니다.
이것은 무작위 작은 행동 검사입니다. 정책이 작업을 해결했다는 결과가 아닙니다.

## C. 시연 다운로드와 RGB 변환

```bash
bash scripts/02_prepare_demos.sh StackCube-v1 20
export DEMO_ROOT="$(python -c 'from mani_skill import DEMO_DIR; print(DEMO_DIR)')"
export STACK_DEMO="$DEMO_ROOT/StackCube-v1/motionplanning/trajectory.rgb.pd_ee_delta_pos.physx_cpu.h5"
```

원본 시연은 관측 영상을 포함하지 않을 수 있으므로 RGB 재생성 단계가 필요합니다.[S2,S5]
20개 변환 요청이 20개 유효 시연을 보장하지는 않습니다. 변환 로그의 성공 수와 HDF5 실제 trajectory 수를 확인합니다.
실패한 변환을 성공으로 표시하지 마십시오. 기존 출력 데이터는 덮어쓰지 않습니다.

## D. Diffusion Policy 짧은 학습 → 실제 학습

먼저 명령만 확인합니다.

```bash
python scripts/03_train_dp.py --config configs/stackcube.yaml \
  --demo "$STACK_DEMO" --run stack_plumbing_s1 --num-demos 20 --total-iters 500
```

같은 명령에 `--execute`를 붙이면 학습을 시작합니다.

```bash
python scripts/03_train_dp.py --config configs/stackcube.yaml \
  --demo "$STACK_DEMO" --run stack_plumbing_s1 --num-demos 20 --total-iters 500 --execute
```

500 iteration은 설치·배치·학습 동작 확인용입니다. 이 정책의 성공률을 본 실험 성능으로 취급하지 않습니다.
성공률을 확인하며 개발 단계에서 학습량을 정합니다. 다음은 20개 시연에서 학습을 늘리는 명령 예시이며 최적 설정 주장이 아닙니다.

```bash
python scripts/03_train_dp.py --config configs/stackcube.yaml \
  --demo "$STACK_DEMO" --run stack_s1 --num-demos 20 --total-iters 30000 \
  --eval-freq 1000 --execute
```

100개 시연으로 늘릴 때는 기존 20개 RGB 파일을 삭제·덮어쓰지 말고 별도 데이터 위치/실행 ID를 사용합니다.
공식 학습기는 시연을 메모리에 올립니다. VRAM/RAM을 측정한 다음 규모를 늘립니다.[S3]

체크포인트 파일을 확인한 뒤 내보냅니다.

```bash
find runs/policy/stack_s1/checkpoints -maxdepth 1 -type f
python scripts/04_export_policy.py --run runs/policy/stack_s1 \
  --checkpoint runs/policy/stack_s1/checkpoints/best_eval_success_once.pt \
  --out artifacts/policies/stack_s1
```

해당 체크포인트가 없다면 학습·평가 로그를 확인하십시오. 이름만 맞춘 가짜 파일을 만들면 안 됩니다.
공식 DP의 초기화·EMA 등은 재사용하지만 **관측 필드는 RGB+qpos+qvel+TCP pose로 명시적으로 제한**했습니다.
따라서 다른 관측 스키마로 학습된 공개 체크포인트를 아무 검증 없이 대신 로드할 수 없습니다.

## E. 학습 정책을 포함한 분기 복원 audit

```bash
python -m hci audit --config configs/stackcube.yaml \
  --policy artifacts/policies/stack_s1 --out runs/audit_stack_001
```

`runs/audit_stack_001/audit.json`에서 전체 `pass: true`여야 합니다.
같은 상태에서 같은 정책을 재실행했을 때와 다른 개입을 실행한 뒤 원래 상태로 돌아왔을 때를 모두 비교합니다.
허용 오차를 근거 없이 키워 통과시키면 안 됩니다.

## F. 2개 원본 실행으로 데이터 파이프라인 확인

```bash
python -m hci collect --config configs/stackcube.yaml \
  --policy artifacts/policies/stack_s1 --audit runs/audit_stack_001/audit.json \
  --episodes 2 --out runs/branches_smoke_001
```

`dataset.json`에서 완료 여부, 실제 체크포인트 수, 분기 수, 처리 시간을 확인합니다.
개입의 움직임과 성공 판정은 GPU 호스트에서 `scripts/record_episode.py`로 확인합니다. 기본 수집기는 디스크 절약을 위해 영상을 자동 저장하지 않습니다.
실제 영상 기록 명령은 `README.md`의 *Recording intervention videos* 절에 있습니다.

```bash
python scripts/inspect_branches.py --data runs/branches_smoke_001
```

## G. 파일럿 분기 수집 및 예측기 학습

```bash
python -m hci collect --config configs/stackcube.yaml \
  --policy artifacts/policies/stack_s1 --audit runs/audit_stack_001/audit.json \
  --episodes 120 --out runs/branches_train_001
python -m hci fit-heads --data runs/branches_train_001 \
  --out artifacts/heads/stack_001 --steps 2000
```

상한은 120 × 4 체크포인트 × 3 선택지 × 4 반복 = 5,760개 분기 실행입니다.
작업이 일찍 끝나면 적어집니다. 기본 정책 학습, 전체 에피소드 후보 평가 비용은 이 수에 포함되지 않습니다.
먼저 소규모 측정으로 비용을 계산하세요.

```bash
python scripts/estimate_budget.py --seconds-per-rollout 3 --hourly-usd 0.5
```

위 3초와 0.5달러는 **사용법을 위한 가정값**입니다. 실제 측정과 실제 대여 요금으로 바꾸십시오.

## H. 개발 평가와 후보 정책 고정

```bash
python -m hci evaluate --config configs/stackcube.yaml \
  --policy artifacts/policies/stack_s1 --heads artifacts/heads/stack_001/heads.pt \
  --audit runs/audit_stack_001/audit.json --split dev --episodes 20 --out runs/dev_smoke_001
python -m hci summarize --run runs/dev_smoke_001 --out runs/dev_smoke_report_001
```

작업이 정상이라면 별도 실행 ID로 개발 split 전체를 평가합니다.

```bash
python -m hci evaluate --config configs/stackcube.yaml \
  --policy artifacts/policies/stack_s1 --heads artifacts/heads/stack_001/heads.pt \
  --audit runs/audit_stack_001/audit.json --split dev --out runs/dev_full_001
```

후보 목록은 `configs/candidates.json`입니다. 개발 중 바꿀 수 있지만 **보정 데이터를 보기 전** 확정해야 합니다.

## I. 보정 — 개발 설정을 확정한 뒤에만

```bash
python -m hci evaluate --config configs/stackcube.yaml \
  --policy artifacts/policies/stack_s1 --heads artifacts/heads/stack_001/heads.pt \
  --audit runs/audit_stack_001/audit.json --split calibration --out runs/calibration_001
python -m hci certify --dev runs/dev_full_001 --calibration runs/calibration_001 \
  --epsilon .05 --delta .05 --out artifacts/selection/stack_001
```

표본이 부족하면 어떤 개입도 통과하지 않아 `never`가 선택될 수 있습니다. 오류나 성능 개선 실패를 숨기지 마십시오.
보정 결과를 보고 예측기·후보를 바꾸면 새 보정 split을 선언해야 합니다.

## J. 동결 — 여기까지 실행하고 Test는 대기

```bash
python -m hci freeze --root . --out frozen/stack_001 \
  --config configs/stackcube.yaml --policy artifacts/policies/stack_s1 \
  --heads artifacts/heads/stack_001/heads.pt --registry configs/candidates.json \
  --splits configs/splits.json --selection artifacts/selection/stack_001/selection.json \
  --audit runs/audit_stack_001/audit.json
python -m hci verify-freeze --frozen frozen/stack_001
```

최종 Test 명령은 `docs/06_FREEZE_AND_TEST.md`에만 별도로 적었습니다.
동결 보고서를 확인하기 전에는 실행하지 않습니다. 임의 승인 토큰을 에이전트가 만들어 실행하면 안 됩니다.

## K. PegInsertionSide와 다른 정책 seed

StackCube 파일럿이 정상화된 뒤 `configs/peg_insertion.yaml`로 동일한 절차를 별도 폴더에서 반복합니다.
Peg는 `pd_ee_delta_pose`, horizon 300, 과거 TCP 위치 방향으로의 제한된 후퇴를 사용합니다.
정책/heads/보정/동결 파일을 작업 간 재사용하지 않습니다. 정책 seed 반복도 독립 폴더와 provenance를 남깁니다.
본 논문 규모는 파일럿 측정 뒤 `docs/05_STAGE_GATES.md`의 본 실험 항목을 확정합니다.

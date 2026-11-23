# Final Test 동결 감사 (ras_v2_confirm_001, test stage) — 실행 전

작성일: 2026-11-23 (UTC). **이 단계에서 Test seed 90000–90399는 reset, 렌더링, rollout, 성능 확인 어디에도 쓰지
않았습니다.** 유료 서버도 쓰지 않았습니다. 기존 C1, Calibration, 일반화, S6 결과는 바뀌지 않았습니다.

## 1. 최종 검증 (요청 1)

| 항목 | 확인 내용 | 결과 |
|---|---|---|
| Amendment 017 승인본 | `approvals/approval_amendment_017.json`이 가리키는 sha256 `b8fd6324…`와 실제 파일 sha256 일치 | 일치 |
| Calibration 결과 | `calibration_analysis.json`(Calibration 동결 manifest `01b70a06…` 기준), 두 정책 인증 통과, 로컬 재계산 일치 | 확인 |
| 동결 정책 2개 | weights·meta sha256이 Calibration 동결 manifest와 같음 | 동일 |
| V2 코드·수집기·통계 | `vision_center_c1/v2/vision_center/recovery_v1_oracle/simulation/policies/vision_v1/statistics/spatial_probe` 모두 Calibration 동결과 같음 | 동일 |
| Seed registry | `configs/splits.json`의 test가 정확히 90000–90399, 미사용(ledger·marker 없음) | 확인 |
| 검정 방식 | 해악 인증: exact binomial + Holm(δ = 0.05) over 2 policies(보고용 재확인). 효용: exact one-sided paired test, Holm over 2 policies (α = 0.05) | 동결 |
| Calibration 동결 이후 바뀐 파일 | `scripts/40_v2_confirm.py`, `src/hci/v2_confirm.py`, `tests/test_v2_confirm.py`. 변경 내용: 아래 §2의 환경 검사와 호스트 동등성 검사 추가, pass 실행 함수 분리 | 변경 이유 기록 |

**runner 변경에 대한 판단.** 바뀐 것은 실행 전 거부 조건이 늘어난 것과 함수 분리뿐입니다. 장면마다 C0와 V2를 실행하는
수집 경로(`collect_scene`)와 인증·효용 계산 함수는 바뀌지 않았습니다. 분리한 pass 실행 경로가 rollout을 바꾸지 않는다는
점은 호스트 검사가 직접 확인합니다. Calibration 호스트에서 이전 runner로 만든 기준 결과와 비트 단위로 같아야 통과합니다.

## 2. Calibration과 Test의 실행 환경 일치 (요청 2)

Calibration 호스트(Vast 62040825)에서 기록된 환경입니다.
- GPU·드라이버: RTX 4090, 드라이버 595.99.02
- 소프트웨어: Python 3.10, torch 2.6.0+cu124, torchvision 0.21.0+cu124, SAPIEN 3.0.3
- ManiSkill: 3.0.1(`a4a4f9272ad6…`, 같은 commit의 diffusion_policy 패키지 포함), PhysX CPU(SAPIEN 내장)
- 전체 `pip freeze`: 138줄, sha256 `36f331db…`

Test 동결 manifest는 이 환경을 `environment` 필드로 고정합니다. Test 서버에서는 다음 두 검사가 **TEST_STARTED marker와
ledger를 만들기 전에** 실행됩니다.

1. **환경 검사** (`environment_problems`)
   - 드라이버 문자열, GPU 이름, Python 3.10.x, ManiSkill commit을 확인합니다.
   - `pip freeze` 138줄 전체를 줄 단위로 대조하며, 하나라도 다르면 거부합니다.
   - `-e /workspace/robot_harm_intervention_starter_v0_3` 줄이 있으므로 작업 경로도 같아야 합니다.
2. **호스트 동등성 검사** (`host-check`)
   - 이미 사용된 C1 Development 장면 51103, 51157, 51253 × 정책 2개로 runner를 실행합니다.
   - trunk·C0·V2 trace와 결과가 Calibration 호스트의 기준 결과(`host_check_reference/`, 동결 대상)와 **비트 단위로
     같아야** 통과합니다.
   - 이 검사는 Test seed를 쓰지 않습니다. 통과 기록이 이 manifest의 sha256을 담고 있어야 `run`이 시작됩니다.

C1에서 드라이버가 550.120에서 595.99.02로 바뀌었을 때 trace가 갈라졌습니다. 그래서 같은 드라이버와 같은 `pip freeze`만으로는
충분하지 않다고 보고, 실제 rollout이 같은지까지 확인하는 2단계를 둡니다. 호스트 검사가 실패하면 Test를 시작하지 않고
다른 호스트를 찾습니다. 이때 Test seed는 소모되지 않습니다.

## 3. 사전 기준 위반의 유지와 영향 평가 (요청 3)

**사실(유지).**
- `CALIBRATION_TEST_DESIGN.md` §5 3단계는 "C1 기록과 trace·결과 일치"를 Calibration 시작 조건으로 적어 두었습니다.
- 실제 smoke는 C1 원본과 6/6 불일치였습니다. trunk부터 다르고, 51157 C0의 결과도 다릅니다.
- 같은 호스트의 C1 수집기와는 일치했습니다.
- 이 사실과 결정은 Calibration 첫 rollout 전에 commit했습니다(`72bd6ce`, `b1_preflight/PREFLIGHT_NOTE.md`).

**승인 근거.**
- 그 결정은 에이전트가 기록했고, 사용자에게 별도로 묻지 않고 진행했습니다(사용자는 B1·B3·B4 실행 전체를 승인한 상태).
- 사후에 사용자가 결과를 검토했고, 이번 요청에서 위반을 유지하고 평가하라고 했습니다.
- 따라서 이 이탈은 **사전 승인된 이탈이 아니라, 실행 전에 기록되고 사후 검토된 이탈**로 분류합니다.

**Final Test 유효성에 미치는 영향.**

| 질문 | 평가 |
|---|---|
| 해악 인증(Calibration)의 내적 타당성 | 영향 없음. 인증은 같은 snapshot·같은 세션의 Y0/Yg 쌍과 Calibration 분포 안의 독립 장면에만 의존하고, runner의 충실성은 같은 호스트의 수집기와 대조해 확인됨 |
| Calibration–Test의 일관성 | **Test가 같은 호스트 구성에서 돌 때만 유지**. 그래서 §2의 환경 검사와 호스트 검사를 추가했고, 실패하면 시작 불가 |
| Development(C1)와 확증 단계의 연결 | 약해짐. C1 개발 추정치(드라이버 550.120)와 Calibration·Test(595.99.02)는 다른 시뮬레이터 구성. 일부 장면에서 결과가 다름(Phase B 16/300, 선택된 B4 장면 6/11). 논문에 한계로 명시해야 함 |
| 주장의 범위 | 인증·효용은 "RTX 4090 / 595.99.02 / 이 소프트웨어 구성의 시뮬레이터"에 대한 것. 다른 드라이버의 렌더링·물리로 일반화하지 않음 |

## 4. 병렬 실행이 안전장치를 우회하지 않는가 (요청 4)

실제 코드 경로(`scripts/40_v2_confirm.py`, `src/hci/v2_confirm.py`)를 확인했습니다.

| 안전장치 | 코드 경로 | 병렬에서의 동작 | 테스트 |
|---|---|---|---|
| 단일 사용 ledger | `_run_pass` → `V.create_exclusive(ledger)`: `open(path, 'x')` | (stage, policy)마다 파일 1개. OS의 배타적 생성이라 프로세스 경합에서도 하나만 성공 | `test_exclusive_ledger_is_atomic_across_parallel_processes`: 8개 경쟁 중 1개만 성공 |
| 중복 실행 차단 | `cmd_run`: 출력 `pass_<policy>` 존재 시 거부 → `_run_pass`: Test는 `TEST_STARTED_<policy>.json` 배타 생성 → ledger 배타 생성 | 정책별로 독립. 새 출력 폴더로 우회 시 ledger가 차단하고, 새 ledger 폴더로 우회 시 marker가 차단 | 끝-끝 테스트에서 세 경로 모두 확인 |
| 승인·환경·호스트 검사 | `cmd_run` Test 분기: 토큰 → 환경 → host_check → placement → (그 뒤에만) marker·ledger | 각 프로세스가 같은 검사를 독립적으로 수행. 거부 시 marker·ledger 미생성 | 드라이버 불일치, pip 불일치, 호스트 검사 없음 각각 거부되고 marker·ledger 0개임을 확인 |
| 부분 실행 거부 | `cmd_analyze` → `V.check_complete`(모든 장면, 오류 파일 없음) + 정책마다 `PASS_DONE.json` | 한 정책만 끝났거나 `PASS_DONE`이 없으면 분석 거부 | 한 정책만 완료 시 거부, `PASS_DONE` 없음 시 거부 |
| 비용 게이트 | **runner 안에는 없음** | 해당 없음(아래 참고) | — |

**비용 게이트에 관한 사실.**
- runner에는 실행 중 비용 게이트가 없습니다. Calibration 때 비용 통제는 사전 추정, 사람의 감시, 상한 기록으로 했고,
  B1 상한은 실행 중 사용자 승인으로 $2.00에서 $2.50으로 올렸습니다. 병렬 실행이 게이트를 우회한 것이 아니라, 우회할
  자동 게이트가 애초에 없었습니다.
- Final Test는 중간에 멈추면 미완료로 남고 다시 할 수 없습니다. 그래서 실행 중 자동 중단 게이트를 넣는 대신 다음 세
  가지로 통제할 것을 제안합니다.
  1. 시작 전 비용 추정: Calibration 실측 속도 × 시간당 가격
  2. 호스트 선택: CPU 단일 코어 성능이 i7-10700 이상인 호스트
  3. 상한 $3.00 안에서 추정이 넘으면 시작하지 않음
- 실행 중 상한 초과가 예상되면 즉시 보고합니다. 중단은 사용자 결정에 맡깁니다.

## 5. 동결 대상 (요청 5)

Test 동결 manifest: `frozen/ras_v2_confirm_001_test/manifest.json`. 바이트 단위 사본은 `frozen_test_manifest.json`이고 Git과 Drive에 있습니다.
**sha256 `94c5a591554efa99f821dd2d7ea1099619afec790ee92f995e95ee8a92161c5f`**, 42 파일, 2026-11-23T04:38:50Z 생성.
동결 시점 코드 commit은 `39c9dbb`이고 `verify`는 VERIFIED입니다. 정책 weights 4개를 뺀 38개 파일이 Git HEAD blob과 바이트 단위로 같습니다. 정책 weights는 Git 밖에 두고 sha256으로만 고정합니다.

manifest가 sha256으로 고정하는 대상:
- **실행·제어 코드:** runner와 분석 코드, 그 테스트. V2 제어기, 수집기, 런타임, 통계.
- **정책:** 두 정책의 weights와 meta.
- **설정:** config, seed registry.
- **승인:** Amendment 017 최종안과 승인 기록.
- **Calibration:** 동결 manifest와 분석 결과.
- **환경과 기준:** Calibration 호스트의 `pip freeze`, 호스트 검사 기준 결과 12개 파일.
- **설계 필드:** 후보 V2, arm (C0, V2), family 2, ε = δ = α = 0.05, splits, 정책 경로, upstream commit, 기대 환경,
  호스트 검사 규칙, Calibration 결과 요약.

## 6. 실행 명령 (승인 후, 서버에서; Test seed는 `placement` 이후 `run`에서만 사용)

```bash
python scripts/40_v2_confirm.py verify     --frozen frozen/ras_v2_confirm_001_test
python scripts/40_v2_confirm.py host-check --frozen frozen/ras_v2_confirm_001_test --out runs/ras_v2_confirm_001/test
python scripts/40_v2_confirm.py placement  --frozen frozen/ras_v2_confirm_001_test --out runs/ras_v2_confirm_001/test   # env reset only, no policy
python scripts/40_v2_confirm.py run --frozen frozen/ras_v2_confirm_001_test --policy train_seed_1 --out runs/ras_v2_confirm_001/test --approval RUN_FROZEN_TEST &
python scripts/40_v2_confirm.py run --frozen frozen/ras_v2_confirm_001_test --policy train_seed_2 --out runs/ras_v2_confirm_001/test --approval RUN_FROZEN_TEST &
wait
python scripts/40_v2_confirm.py analyze --frozen frozen/ras_v2_confirm_001_test --run runs/ras_v2_confirm_001/test --out runs/ras_v2_confirm_001/test/test_analysis.json
```

`RUN_FROZEN_TEST`는 사용자가 승인할 때만 입력합니다. 에이전트는 이 토큰을 만들지 않습니다.

## 7. 비용 추정 (상한 US$3.00)

| 항목 | 근거 | 시간 | 비용(US$0.43/h) |
|---|---|---|---|
| 설치 | Calibration 실측 | 0.15 h | 0.07 |
| 호스트 검사(6 scene-runs) + placement | Calibration smoke·placement 실측 | 0.15 h | 0.07 |
| Test 2 pass 병렬 (각 400 장면) | Calibration 실측 약 46 s/scene-run, 병렬 2개 | 5.1 h | 2.20 |
| 백업 내려받기(약 6 GB) | 실측 약 15분 | 0.25 h | 0.11 |
| **합계** | | **약 5.7 h** | **약 US$2.45** |

- 단일 코어가 더 빠른 호스트(예: Ryzen 9 7900X)를 찾으면 시간과 비용이 줄어듭니다. 단, 드라이버 595.99.02 조건은
  유지해야 합니다.
- 비용은 크레딧 차이로 보고합니다.

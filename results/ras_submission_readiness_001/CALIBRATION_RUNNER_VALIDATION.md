# Calibration / Test runner 검증 (Task A2)

## 1. 구현

| 파일 | 역할 |
|---|---|
| `src/hci/v2_confirm.py` | split 확인, family, 해악 인증(exact binomial + Holm), 효용 검정(exact paired + Holm), 검정력 표, 완결성 검사, manifest·검증, 승인 기록 확인, 배타적 ledger/marker, Test 토큰 확인 |
| `scripts/40_v2_confirm.py` | `freeze`(calibration/test 단계), `verify`, `placement`(정책 없이 초기 배치), `run`(C0+V2, 한 snapshot), `analyze` |
| `tests/test_v2_confirm.py` | 단위 6개 + mock 끝-끝 1개 |
| `scripts/41_gen_arch_plainconv.py`, `tests/test_gen_arch_plainconv.py` | 일반화 실험 runner와 테스트 |

재사용(변경 없음): `hci.vision_center_c1.collect_scene`(C0와 V2를 같은 snapshot에서 실행, 기권 시 C0 궤적 동일성 검사,
특권 접근 차단 ArmView), `validate_scene`, `scripts/33`의 `save_scene`(trace·frame SHA256 기록), `hci.statistics`.

## 2. 요구 기능 대비

| 요구 | 구현 | 테스트 |
|---|---|---|
| 동결 정책·V2만 사용 | freeze manifest에 두 정책 weights/meta와 제어 코드 해시, 실행 전 `verify_manifest` | 변경 파일 거부 |
| C0와 V2 같은 snapshot·같은 세션 | `collect_scene(arms=(C0, V2))` | mock 끝-끝(2 정책 × 2 장면) |
| seed 중복·범위 검사 | `split_seeds`(정확히 30000–30399 / 90000–90399), `seeds_disjoint` | 실제 config로 검사, 개발 집합과 교집합 0 |
| 원본 해시 검증 | 분석 시 각 trace SHA256을 기록 값과 대조, 기록 파일 해시 저장 | — |
| 성공·해악 계산 | `outcomes` → `certificate` | 경계값(330, k=9/10), Holm family 2, n₊=0 |
| exact paired 효용 검정 | `efficacy` (Test 단계만) | 수식 대조(16 vs 1) |
| exact binomial 해악 검정 + Holm | `certificate` | readiness 표(family 1/2/4 → k 9/8/7) 재현 |
| 결과·trace SHA256 기록 | `record_sha256`, 장면 기록의 `trace_sha256` | 분석 결과에 4개 기록 해시 |
| 중단과 부분 실행 구분 | 오류 → `<seed>.error.json` + INCOMPLETE, `PASS_DONE.json` 없으면 분석 거부 | 누락·오류 파일 거부 |
| 중복 실행 차단 | 출력 폴더 존재 시 거부, (stage, policy)별 배타적 ledger, Test 정책별 `TEST_STARTED` marker | 새 출력 폴더로 우회 → ledger가 차단; 새 ledger 폴더로 우회 → marker가 차단 |
| Test 승인 토큰 | `RUN_FROZEN_TEST` 없으면 PermissionError, marker 생성 전 거부 | 토큰 없이 실행 시 marker 미생성 확인 |
| 재튜닝 금지 | 분석은 동결된 family·ε·δ만 읽음; 설정 인자 없음 | — |
| 승인 없는 동결 금지 | 승인 기록이 개정안 SHA256을 명시해야 함 | 없음·불일치 거부 |
| calibration/test seed로 smoke 금지 | `--seeds`가 두 범위를 이름하면 거부 | 거부 확인 |

## 3. 테스트 결과

결과 로그: `logs/pytest_runner_and_related.log`(신규 runner + 기존 `test_statistics.py`, `test_vision_center_c1.py`),
`logs/pytest_gen_arch.log`. 결과 요약은 `START_HERE.md`에 적습니다.

## 4. 아직 검증하지 않은 것

- **실제 시뮬레이터·정책에서의 실행**: 로컬에는 GPU 렌더링이 없어 mock 런타임으로만 검증했습니다. 승인 후 서버에서
  이미 사용된 Development 장면(예: 51103, 51157, 51253) × 2 정책으로 smoke를 돌려 C1 기록과 C0·V2 trace·결과가 같은지
  확인한 뒤 Calibration을 시작합니다(B1의 첫 단계).
- `placement` 명령은 실제 ManiSkill이 필요하며, C1과 같은 참조 집합에 C1 dev 200개를 추가해 비교합니다(코드 검토만).
- 전체 `pip freeze`와 driver 기록은 실행 호스트에서 생성됩니다.

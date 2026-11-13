# 설정/프로토콜 변경 기록 001 — 상태 복원 방식

기존 버전: v0.3 초기 패키지
새 버전: v0.3 + 서버 통합 수정 (아래 파일)
변경 시점: 2026-11-13 (UTC), Stage 0 smoke 단계. 정책 학습·audit·분기 수집·dev/calibration/test 평가 이전.
변경 사유: 실제 GPU 호스트(대여 RTX 3090, ManiSkill a4a4f927, sapien 3.0.3, physx_cpu)에서 발견한 문제 3건.

## 1. 환경 RNG 저장 실패 (인터페이스 오류)

- 증상: `runs/smoke_pick_001`, `runs/smoke_stack_001` exit 1. `copy.deepcopy(env._batched_main_rng)`에서 `ValueError`.
- 원인: ManiSkill `BatchedRNG.__getattribute__`가 `__reduce_ex__`까지 구성 RNG로 전달한다.
- 변경: `src/hci/simulation.py`에 `rng_state` / `set_rng_state` 추가. 객체 복사 대신 `get_state()` 값을 저장하고 기존 객체에 `set_state()`로 되돌린다 (`_episode_rng` 별칭 유지).
- 회귀 테스트: `tests/test_rng_restore.py` (로컬 replica + 서버에서는 실제 `BatchedRNG`).

## 2. 학습 패치 import 실패 (인터페이스 오류)

- 증상: `runs/policy/stack_plumbing_s1` 학습 시작 전 exit 1. `ImportError: cannot import name 'wrappers' from 'mani_skill.envs.utils'`.
- 원인: 고정 commit에서 `mani_skill.utils` 속성이 `mani_skill.envs.utils`로 가려져 `import a.b.c as x` 형태가 실패한다.
- 변경: `src/hci/observations.py`에서 `importlib.import_module`로 대상 모듈을 직접 가져온다. 패치 내용 자체는 그대로.
- 회귀 테스트: `tests/test_training_patches.py`.

## 3. 상태 복원 방식 변경 (프로토콜 변경, docs/01_PROTOCOL.md 6절의 사전 명시 대안)

- 증상: RNG 수정 후 `runs/smoke_pick_002`, `runs/smoke_stack_002` exit 2. 복원 후 재실행 오차 2.7e-3 / 6.3e-3 (기준 `state_atol` 1e-6).
- 진단 (서버 실측, StackCube/PickCube):
  - `reset(seed)` + 같은 행동열 2회: 차이 0.0
  - 복원 후 실행 vs 복원 후 실행: 차이 0.0
  - 복원 없이 계속 vs 복원 후 실행: 1 step 뒤 7e-5, 8 step 뒤 6e-3
  - 로봇(articulation)만 `set_state`: 차이 0.0 / 강체 actor(큐브)만 `set_state`: 7.3e-5
  - drive target, qf, qacc, link 속도·pose는 복원 전후 동일
- 원인: `set_state_dict`가 강체 actor 상태를 다시 써넣을 때 PhysX 내부 접촉·솔버 상태가 달라진다. 이 상태는 `get_state_dict`에 없다.
- 변경: `restore_env`가 `set_state_dict`를 쓰지 않고 **prefix action replay**로 복원한다.
  `env.reset(seed)` → 기록된 행동열 재실행 → 재생된 상태를 스냅샷 상태와 `state_atol`로 비교(불일치 시 `RuntimeError`) → RNG 복원 → 정책 큐·이력 복원.
  `Runtime`은 실행한 모든 행동(`self.actions`)을 기록하고 스냅샷에 episode seed와 함께 저장한다. 다른 episode의 스냅샷은 거부한다.
- **허용 오차는 변경하지 않았다** (`state_atol` 1e-6, `action_atol` 1e-6, `observation_atol` 1.0 그대로).
- 회귀 테스트: `tests/test_replay_restore.py` (state dict에 없는 숨은 상태를 가진 mock 환경).
- 재검증: `runs/smoke_pick_003`, `runs/smoke_stack_003` PASS, 상태·RGB·성공 차이 모두 0.0.

영향받는 파일/설정: `src/hci/simulation.py`, `src/hci/smoke.py`, `src/hci/observations.py`, 신규 테스트 3개. YAML 설정은 변경 없음.
변경 전에 이미 확인한 train/dev/calibration/test 결과: 없음.
기존 보정의 유효성: 해당 없음 (보정 미수행).
새로 생성해야 하는 데이터: 없음 (분기 데이터 수집 전).
재실행할 검사: CPU 테스트, smoke 2종 (완료). 정책 포함 audit는 이 방식으로 처음 실행한다.
기존 결과 보존 경로: 서버 `runs/smoke_*_001`, `runs/smoke_*_002`, `runs/policy/stack_plumbing_s1` (삭제하지 않음).
승인 범위: 개발 단계 수정. 비용 영향 — 분기 1회마다 checkpoint 시점까지의 물리·렌더 step이 추가된다 (정책 추론은 없음). 실측은 2개 seed 수집에서 보고한다.
한계: 재생 일치는 검사한 seed·상태에서의 결과이며 시뮬레이터 전역 결정성의 증명이 아니다. 초기 패키지의 해시 목록과 위 3개 소스 파일은 일치하지 않는다.

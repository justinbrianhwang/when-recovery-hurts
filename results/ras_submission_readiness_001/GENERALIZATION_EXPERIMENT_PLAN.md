# 일반화 실험 계획: gen_arch_plainconv_001 (Task C3, 준비 완료·미실행)

## 1. 질문

동결된 다른 아키텍처의 정책(Stage 1D 공식 PlainConv DP)에 **수정하지 않은** Vision-CENTER V2를 적용해도 C0보다
더 많은 장면에서 성공하는가? C1의 두 정책은 같은 ResNet-18 아키텍처·같은 시연에서 seed만 달랐습니다.

## 2. 설계 (실행 전 동결: `scripts/41_gen_arch_plainconv.py plan`이 해시를 기록)

| 항목 | 값 |
|---|---|
| 실험 ID | `gen_arch_plainconv_001` (C1과 별개) |
| 정책 | `artifacts/policies/stack_1d_s1_d500_clean_v1_i080000` (동결, 변경 없음) |
| 장면 | 새 Development seed 51300–51499 (200개, 미사용 확인; Calibration·Test 아님) |
| Arms | C0, V2_SINGLE_FRAME, VISION_CENTER_V2, CENTER_ORACLE(특권 참조) — 한 snapshot, 같은 세션 |
| 제어기 | `hci.vision_center_c1.collect_scene`과 V2 코드 그대로 |
| 결과 | `success_once_absorbing`, 200 step |
| 주 가설 | **H_arch**: V2 성공 장면 > C0 (exact one-sided paired test, α = 0.05, 단일 검정 → 다중비교 없음) |
| 기술 보고 | 성공(Wilson 95%), arm별 rescue/harm/net, 트리거율·개입률·기권 사유, oracle rescue 중 V2 회복 수, V2 vs single frame(검정 없음), 목표 밀림, C1 개발 안전 기준 |
| 성공/실패 기준 | H_arch 귀무가설 기각 여부만 주 판단. 기각 실패는 "효과 없음"의 증거가 아님(기회 수를 함께 보고) |
| 중단 조건 | 장면 오류 시 INCOMPLETE로 중단(재개 없음); 비용 상한 도달 시 중단 |
| 비용 | 200 scene-runs × 4 arms, ≈ 14 s/scene-run 가정 → ≈ 0.8 h + 설치 0.5 h ≈ US$0.5; 상한 US$1.5 |
| 저장 | ≈ 1–1.5 GB (C1 기준 비례) |

## 3. 검정력과 해석 범위

- 효과 크기는 이 정책의 실패 유형에 달려 있습니다. 해제 직전 오정렬 실패가 적으면 rescue 기회가 적어 기각이 어렵습니다.
  그래서 oracle rescue(같은 시점·같은 이동, 특권 위치)를 함께 보고해 "기회가 없었는지"와 "V2가 못 잡았는지"를 구분합니다.
- 불일치 장면 기준 기각 조건: V2만 성공 b, C0만 성공 c일 때 c = 0이면 b ≥ 5, c = 1이면 b ≥ 7(α = 0.05 단측).
- 결과가 지지하는 것: V2의 이득이 ResNet-18 정책에 국한되지 않는다는 것(같은 태스크·같은 시연 분포 안에서).
  지지하지 않는 것: 다른 태스크, 실물, 해악률 인증.

## 4. 원래 V2의 직접 일반화와의 구분

이 실험은 V2를 한 줄도 바꾸지 않습니다. 다른 태스크로 옮기며 규칙을 바꾸는 실험(StackPyramid 등)은 "원리 이식"으로
별도 ID와 별도 개발·검증 단계를 가집니다.

## 5. 실행 절차 (승인 후)

1. `python scripts/41_gen_arch_plainconv.py plan --run runs/gen_arch_plainconv_001` (로컬) → plan SHA256 보고.
2. 사용자: `authorization.json` 작성(plan SHA256, 상한).
3. 서버: 설치 → placement 검사(51300–51499, 정책 없이; C1 방식) → `run` → 백업·해시 확인 → 인스턴스 삭제.
4. 로컬: `analyze` → 기존 C1 mechanism 분석 파이프라인으로 기술 통계 보완.

## 6. Runner 검증 (완료)

`tests/test_gen_arch_plainconv.py`: 설계 상수, plan 중복 거부, 승인 없음·plan 불일치 거부, calibration/test seed 거부,
ledger 단일 사용, 재실행 거부, 부분 실행 분석 거부, mock 런타임 끝-끝 실행(합성 seed 7, 8). 결과는 `logs/pytest_gen_arch.log`.

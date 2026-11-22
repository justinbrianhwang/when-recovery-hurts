# 승인이 필요한 작업 (한 번에 정리)

아래 항목은 **모두 실행하지 않았습니다.** 각 항목은 서로 독립적으로 승인할 수 있습니다. 비용은 Vast RTX 4090
(≈ US$0.35/h, 전송비 포함 추정) 기준이며, 실행 전 처음 몇 장면의 실측으로 다시 확인합니다. 이전 실측: C1 400 scene-runs
(9 arms) US$1.46, rgb_replay_001 US$0.06.

## A. 프로토콜 (무료, 결정 필요)

| ID | 승인 대상 | 기본 권고 | 승인 방식 |
|---|---|---|---|
| A0 | `AMENDMENT_017_FINAL_PROPOSAL.md` 채택 | 채택 | 사용자가 `approval_amendment_017.json`에 이 파일의 SHA256을 적어 작성(코드는 쓰지 않음) |
| A0-D1 | 후보 | V2 단독 | A0에 포함 |
| A0-D2 | 정책 범위 | 두 정책, Holm family 2, 공동 주장은 둘 다 통과 시 | A0에 포함 |
| A0-D3 | Calibration 크기 | 400 장면 | A0에 포함 |
| A0-D4 | 효용 검정 | Final Test에서만, 정책별 Holm | A0에 포함 |
| A0-D5 | 메모리 오류 | 보고만 | A0에 포함 |

## B. 유료 실행 (각각 승인 필요)

| ID | 작업 | 데이터 | rollouts | 예상 시간 | 예상 비용 | 상한 제안 | 전제 |
|---|---|---|---|---|---|---|---|
| B1 | Calibration (설치 + runner 실환경 smoke 6 scene-runs + placement + 1회 보정) | 30000–30399 × 2 정책 | 800 scene-runs (1,600 branch) | ≈ 2.8 h | ≈ US$1.05 | US$2.00 | A0 승인, 동결 |
| B2 | Final Test | 90000–90399 × 2 정책 | 800 scene-runs | ≈ 2.6 h | ≈ US$0.95 | US$2.00 | B1 완료 후 Test 동결, **사용자의 `RUN_FROZEN_TEST` 제공** |
| B3 | 일반화 실험 gen_arch_plainconv_001 | 새 dev 51300–51499 | 200 scene-runs × 4 arms | ≈ 1.3 h | ≈ US$0.5 | US$1.50 | plan SHA256에 대한 authorization.json |
| B4 | S6 substep 계측 replay | 사용된 C1 dev 장면 7개(사전 고정) | ≈ 7 scene-runs, 필요한 arm만 | ≈ 0.5 h | ≈ US$0.2 | US$1.00 | 로거 ON/OFF 동일성 검사 포함 |

B1, B3, B4는 한 인스턴스에서 순서대로 묶을 수 있습니다(설치 1회 절약, 합계 ≈ US$1.6, 상한 합 US$4.5).
B2는 B1 결과 보고와 별도 승인 후에만 실행합니다.

## C. 학습 (현재 권고하지 않음)

| ID | 작업 | 비용 | 판단 |
|---|---|---|---|
| C1 | ResNet-18 정책 seed 3, 4 학습 | seed당 학습 수 시간 GPU, ≈ US$2–4 | B3 결과를 본 뒤 결정 권고 |
| C2 | 새 태스크(StackPyramid 등) 시연 변환·학습 + V2 원리 이식 | ≈ US$5+ | 원래 V2의 일반화가 아니므로 별도 연구 버전 |

## 승인 후 지켜지는 규칙

- 결과를 보고 후보·규칙·임계값·검정을 바꾸지 않음. Calibration·Test는 각 1회.
- `RUN_FROZEN_TEST`는 사용자가 직접 제공(에이전트가 만들거나 입력하지 않음).
- 키·토큰은 코드·로그·Git에 넣지 않음. 결과 백업·해시 확인 후 인스턴스 삭제, 실제 비용은 크레딧 차이로 보고.

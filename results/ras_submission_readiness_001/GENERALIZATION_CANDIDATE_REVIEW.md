# 일반화 후보 검토 (Task C1–C2)

목표: 개입 효과가 특정 장면과 특정 정책 가중치에만 의존하는지 검증. 태스크 수를 늘리는 것 자체가 목표가 아닙니다.
조사 근거: ManiSkill a4a4f92 소스(`mani_skill/envs/tasks/tabletop/*`, `examples/baselines/diffusion_policy/baselines.sh`),
저장소의 동결 정책 `artifacts/policies/*`와 `policy_meta.json`, V2 코드(`vision_center_v2.py`, `vision_center_c1.py`).

## 1. V2가 전제하는 StackCube 고유 가정

| 가정 | 코드상 근거 | 다른 태스크에서 |
|---|---|---|
| 붉은 큐브(잡은 것)와 초록 큐브(목표), 4 cm 정육면체 | 색 분할·사영, 4 cm 면적으로 가림 판단 | 색·형상이 다르면 추정기 재설계 필요 |
| 목표 위 XY 정렬이 성공 조건의 핵심 | 오프셋 ≥ 1.5 cm일 때 이동, 잔차 10 mm | 삽입·밀기 태스크에는 의미 없음 |
| 트리거: 닫힘 명령 ≥ 3 후 열림 명령 대기 | `recovery_v1_oracle.trigger_rule` | 집어서 놓는 태스크에만 성립 |
| 고정 카메라 상수(base camera 자세, 내부 파라미터) | V2 base 추정 | 카메라가 바뀌면 상수 재측정 |
| `pd_ee_delta_pos` 수평 이동 명령 | 이동 단계 ≤ 5 mm | delta_pose 컨트롤러는 회전 성분 처리 필요 |

## 2. 후보 평가

| 순위 | 후보 | 재학습 | V2 변경 | 코드 변경량 | 비용(추정) | 공정성 | 논문 가치 |
|---|---|---|---|---|---|---|---|
| **1** | **같은 StackCube, 다른 아키텍처의 동결 정책: Stage 1D 공식 PlainConv DP (500 clean demos, seed 1, 80k)** | 없음 | **없음** | 새 runner 1개(완료), 수집·제어 코드 재사용 | ≈ US$0.6, 상한 1.5 | 같은 장면 분포, 같은 C0 비교, 정책만 교체 | 높음: "V2는 ResNet-18 정책 두 개에만 맞춘 것인가?"에 직접 답함 |
| 2 | 같은 정책, 다른 조건(조명·텍스처·초기 배치 분포 변경) | 없음 | 없음(카메라 변경 시 상수 재측정 필요) | env wrapper + 교란 schedule | ≈ US$0.6 | 정책도 OOD가 되어 해석이 섞임 | 중간 |
| 3 | 3·4번째 ResNet-18 학습 seed | **필요** (seed당 학습 ≈ 수 시간 GPU) | 없음 | 학습 스크립트 재사용 | 학습 ≈ US$2–4/seed + 평가 | 같은 아키텍처 | 중간: seed 분산 추정은 가능하나 아키텍처 일반화는 아님 |
| 4 | StackPyramid-v1 (세 번째 큐브를 두 큐브 위에) | 필요(RGB 시연 변환·학습) | 목표 추정(두 큐브 중심)·트리거 단계 수정 → **원래 V2의 직접 일반화가 아님** | 큼 | 학습 + 평가 ≈ US$5+ | 새 태스크 | 높지만 비용·해석 위험 큼 |
| 5 | PlaceSphere-v1 (공을 bin에) | 필요 | 추정기(구·bin 분할) 재작성 | 큼 | 큼 | 다른 제어기 | 원리 이식 실험(V2 아님) |
| 제외 | PegInsertionSide, PushCube, PushT, PullCube 등 | — | 정렬 후 놓기 구조가 없음 | — | — | — | 개입 원리 부적합 |

ManiSkill은 이 commit에서 사전학습 정책 checkpoint를 제공하지 않습니다(baseline은 학습 스크립트만 제공). 따라서
"다른 아키텍처의 동결 checkpoint"는 이 프로젝트가 이미 동결해 둔 Stage 1D 정책이 유일한 무학습 선택지입니다.

## 3. 선택: 후보 1 (gen_arch_plainconv_001)

- 정책: `stack_1d_s1_d500_clean_v1_i080000` (공식 RGB DP baseline의 PlainConv encoder). Vision-V1 자격 평가 장면에서
  28/100 성공(새 ResNet-18 정책 79/100), Stage 1D 자격 37/100. 성공률이 낮은 정책이므로:
  - C0 실패가 많아 rescue 기회는 많을 수 있으나, 실패가 해제 직전 오정렬이 아닌 다른 단계(접근·파지)에서 나면 트리거가
    아예 불리지 않거나 V2가 기권합니다. **oracle rescue 수가 기회의 상한**을 알려 줍니다.
  - C0 성공(해악 분모)이 약 60–75개로 작아 해악은 기술 통계로만 보고합니다(인증 아님).
- 확인 완료: 같은 env_id·control mode·관측 구성·horizon(2/16/8). 런타임은 `vision_v1.runtime`의 아키텍처 분기로 이미 지원.
- V2 규칙·상수·트리거는 C1 코드 그대로(해시로 확인). 바꾸면 이 실험의 범주를 벗어남.

## 4. 이번 작업에서 하지 않은 것

새 정책 학습, 새 장면 rollout, 유료 서버 — 모두 승인 필요(`APPROVAL_REQUIRED.md`). Runner와 테스트는 mock 런타임으로 검증.

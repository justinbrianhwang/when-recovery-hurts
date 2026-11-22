# RAS 원고 구성안 (Robotics and Autonomous Systems, subscription 경로)

형식: Elsevier LaTeX(`elsarticle`), 본문 PDF와 Supplementary PDF 분리. Highlights(3–5개, 각 85자 이내 관례),
Graphical abstract(선택), CRediT, Declaration of competing interest, Data/Code availability, 생성형 AI 사용 고지.
분량·심사 방식·APC 정책은 Guide for Authors(이번 세션에서 접근 거부, 403)를 투고 전에 직접 확인해야 합니다.

**원칙: 실행하지 않은 결과는 쓰지 않습니다.** 아래 각 절의 상태는 `있음` / `대기(승인 필요)` / `계획`으로 표시합니다.

## 가제

"When Recovery Hurts: Certifying the Outcome Risk of an Observation-Only Recovery Intervention for a Frozen Visuomotor Policy"

## Highlights (초안)

- An observation-only centring move, triggered at release, rescues failures of a frozen diffusion policy.
- Outcome harm, a success turned into a failure, is defined and bounded per policy with an exact test.
- Temporal memory of earlier camera views carries most of the gain (paired ablation on two policies).
- Physical validity of reconstructed contacts is audited, and its failure is reported and explained.
- (조건부) Calibration certificate and held-out Test result — 실행 후에만 포함.

## 본문 구성

| # | 절 | 내용 | 근거 자료 | 상태 |
|---|---|---|---|---|
| 1 | Introduction | 동결 정책 위 개입이 오히려 해를 끼칠 수 있다는 문제, 결과 악화(harm) 정의, 기여 3–4개 | — | 작성 가능 |
| 2 | Related Work | 복구·재계획(Recovery chaining, Back to the manifold), 인간/자율 개입(HG-DAgger, ThriftyDAgger, Sentinel, FIPER), 실패 감지(REFLECT, AHA), 위험 제어(Learn-then-Test), Diffusion Policy | Drive `Paper/pdf/` 18편 + 추가 3편 | 작성 가능 |
| 3 | Problem Formulation | Y0, Yg, rescue/harm 4분할, H = P(Yg=0\|Y0=1), 성공 정의 success_once_absorbing, 1회 개입, 특권 입력 금지 | docs/00, 01, 03 | 작성 가능 |
| 4 | Vision-CENTER Method | 트리거, 관측(두 RGB, TCP, 손가락), 메모리(손·base·hold), 결정 규칙(기권 사유), 제한된 이동, hand-back | `vision_center_v2.py`, C1 코드 | 작성 가능 |
| 5 | Experimental Protocol | ManiSkill3 StackCube, 정책 2개, 같은 snapshot의 arm 분기, split, 사전 등록(H_gen, H_mem), 통계 | C1 plan, SAP | 작성 가능 |
| 6 | Development Results | C1 9 arms × 2 정책, rescue/harm, H_gen p=1.5e-5, H_mem p=0.0078, Holm | C1 report, result.json | **있음** |
| 7 | Calibration and Final Test | 해악 인증(Holm family 2), 효용 검정 | — | **대기 (Amendment 017 승인 + 유료 실행 + Test 승인)** |
| 8 | Mechanistic Analysis | 관측→메모리→결정→이동→정렬→재개→결과 사슬, ablation, 놓친 oracle rescue, 해악 사례(51157, 51253) | mechanism_c1_001 | **있음** |
| 9 | Physical Measurement Validity | S4/S5/S6 정의, S6 FAIL과 분해(목록 거리, substep 시점), 통제 두 상자 실험, hull 민감도, S4 1건·S5 2사건 | posthoc_physics_001, hull_001, s6_static_validation | **있음**(B3 계측은 대기) |
| 10 | Generalization Evaluation | 다른 아키텍처 동결 정책(PlainConv)에 V2 그대로 | gen_arch_plainconv_001 | **대기 (실행 승인)** |
| 11 | Discussion | 메모리의 이득과 오래된 메모리 위험(51009), 정렬 부족(명령 대비 실제 이동), 시간 예산 손실형 해악 | — | 작성 가능 |
| 12 | Limitations and Conclusion | 단일 태스크·시뮬레이터 이미지, seed 2개, H는 작업 결과 악화이지 물리 안전 아님, 연속시간 충돌 보장 없음 | — | 작성 가능 |

## 영문 주장 초안 (현재 근거로 쓸 수 있는 것만)

- "On 200 development scenes, the unchanged V2 controller raised success from 171 to 184 (train seed 1) and from 164 to
  180 (train seed 2), with 1 and 0 harms among the scenes the policy solved alone."
- "Pre-registered paired tests rejected the null hypotheses that V2 does not improve on the uninterrupted policy with the
  second policy (p = 1.5×10⁻⁵) and that memory does not improve on a single-frame controller with both policies
  (intersection–union p = 0.0078; Holm over two hypotheses)."
- "These are development results; they do not certify a harm rate." (Calibration 전까지 필수 문장)

쓰면 안 되는 주장: 해악률 인증, 다른 태스크·실물로의 일반화, 물리적 안전, S6 통과, 연속시간 충돌 회피.

## 표 (본문)

- T1 Arms와 입력(특권 여부) 정의
- T2 C1 정책별 arm 성공·rescue·harm·net (Wilson 95%)
- T3 사전 등록 검정(H_gen, H_mem)과 Holm
- T4 S4/S5/S6 요약(공식 판정, 분해 결과, hull 민감도)
- T5 (대기) Calibration 인증과 Test 효용
- T6 (대기) 일반화 실험

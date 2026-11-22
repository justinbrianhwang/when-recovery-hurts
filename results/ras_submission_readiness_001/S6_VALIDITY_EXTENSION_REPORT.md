# S6 물리 측정 타당성 확장 (Task B)

라벨: `POST_HOC_VALIDATION`. **공식 S6는 두 정책 모두 FAIL로 유지**합니다. 아래의 어떤 결과도 S6를 PASS로 재분류하거나
사전 등록 검정에 소급 적용하지 않습니다.

## 1. 실행한 것과 실행하지 않은 것

| 항목 | 상태 | 근거 |
|---|---|---|
| B1 기존 자료·문서·API 확인 | 완료 | 사후 물리 진단 001, hull 민감도 001, `spatial_probe/links/geometry.py`, `s6_validity.py`, PhysX 문서, ManiSkill a4a4f92 소스 |
| B2 독립 정적·통제 두 상자 실험 | **완료 (로컬 CPU, 무료, C1 자료와 완전히 분리)** | `s6_static_validation/` (59 cases, 192 contact points) |
| B3 substep 계측 replay | **준비만** | 별도 시뮬레이션 실행 승인 필요(`APPROVAL_REQUIRED.md`) |

## 2. B1 — 문서와 API가 보장하는 것만

- PhysX `PxContactPairPoint::separation`: "The separation of the shapes at the contact point. A negative separation
  denotes a penetration." (NVIDIA PhysX API reference) → 점 단위 값이며 "두 형상의 최소 거리"라고 정의되어 있지 않음.
- PhysX guide: contactOffset이 접촉 생성 시작을 제어하며, 양수이면 형상이 관통하기 전부터 접촉이 생성될 수 있음.
  "두 shape의 offset 합" 규칙은 PhysX 문서에서 직접 찾지 못했고(Unity 문서에만 같은 서술), **B2에서 실험으로 확인**.
- ManiSkill a4a4f92 `SceneConfig`: contact_offset 0.02, rest_offset 0, PCM on, TGS on, 15/1 iterations, sleep 0.005,
  100 Hz physics, 20 Hz control(제어 step당 5 substep). `scene.get_contacts()`는 `px.get_contacts()`를 그대로 반환.
- C1 로깅: 접촉 목록은 제어 step의 마지막 substep 뒤에 읽고, 자세(link_pose, 큐브 pose)도 같은 시점에 읽음.
  처음 4개 substep의 접촉·자세, solver 내부 상태, PCM manifold는 기록되지 않음.

## 3. B2 — 통제 실험 결과 (ManiSkill과 같은 PhysX 설정, SAPIEN 3.0.3, dt = 0.01 s)

| 실험 | 경우 수 | separation − (스텝 **이전** 자세의 해석적 거리) 최대 | separation − (스텝 **이후** 자세의 거리) 최대 | 판단 |
|---|---|---|---|---|
| G 정렬된 면–면, 정지, 간격 −3 ~ +39.9 mm | 13 | 0.000000 mm | 3.0 mm(관통 경우: solver가 한 스텝에 해소) | 정지 상태에서 separation = 해석적 거리 |
| R 회전(모서리–면, 꼭짓점–면, 손가락형 얇은 상자, 10° 기울임), 0.5–20 mm | 20 | 0.000003 mm | 0.000003 mm | 상자 쌍에서 점 separation 최솟값 = 최소 거리 |
| G 목록 경계 | 5 | 39.9 mm는 목록에 있음, 40.1·41·45 mm는 없음 | — | **두 shape의 contact_offset 합(0.04 m) 규칙 확인** |
| M 일정 속도 접근 0.05–0.5 m/s, 1·5 step | 8 | 0.000003 mm | v·dt (0.5–5.0 mm) | **separation은 마지막 물리 step 시작 시점 자세의 거리** |
| P step 사이 순간 이동 | 3 | 0 | 0 | 이동 후 다음 step 시작 자세 기준으로 재생성 |
| S 느린 이동·회전 50 step, PCM on/off | 8 | 깨어 있는 step: ≤ 0.00002 mm | 잠든(sleep) 뒤: 마지막 이동량(0.008–0.05 mm)만큼 고정 | **sleep 중에는 직전 separation이 갱신되지 않음**; PCM on/off 차이 없음 |
| W 중력, 테이블 위 정지 접촉 | 4 | 0.0004 mm | 0.98 mm(낙하 첫 step) | 정지 접촉 −0.0005 mm 수준 |

## 4. S6 FAIL 분해 — 세 갈래로 구분

### (1) 측정 정의 차이로 설명되는 부분 — **확정(통제 실험)**

- 접촉 목록에는 두 shape 거리가 **0.04 m 미만인 모든 pair**가 들어갑니다(B2 G). C1의 최대 separation 0.03999999 m와 일치.
  따라서 목록에 오른 pair 대부분은 접촉이 아니라 근접(proximity)이며, C1 비교 행의 99%가 impulse 0인 것과 맞습니다.
- 상자 쌍에서 PhysX 점 separation의 최솟값은 해석적 최소 거리와 같습니다(B2 R, 3e-6 mm 이내). 즉 **상자 쌍에서는**
  "점 separation ≠ 최소 거리"가 S6 불일치의 원인이 아닙니다. 이전 진단의 "다른 물리량" 결론은 점별 값(최솟값이 아닌
  개별 점)과 메시 형상에 한정해서 읽어야 합니다.
- 메시(팔 링크·손): 기록된 삼각형 분할이 꼭짓점의 볼록 껍질이 아님 → 거리 최대 0.24 mm 차이(hull 민감도 001). 판정 영향 0.

### (2) 시간 정렬로 설명되는 부분 — **메커니즘 확정, C1 행 단위 적용은 연관**

- 통제 실험: separation은 **마지막 물리 substep 시작 시점의 자세**로 계산됩니다(B2 M, v·dt 차이가 정확히 재현).
  C1은 step 끝 자세를 기록했으므로, 상대 속도 v인 pair는 약 v × 0.01 s만큼 어긋납니다.
- 사후 진단 001의 f = 0.8(한 substep 이전) 재계산이 1 mm 초과 표본 행의 69%(최적 f 79%)를 1 mm 이내로 만든 것은
  이 메커니즘과 정량적으로 일치합니다. C1의 개별 행에 대해서는 substep 자세가 기록되지 않아 연관 수준으로 남습니다.
- S5 FAIL 2개 사건(빠른 큐브 112–189 mm/s)도 같은 메커니즘과 맞습니다(51143: f = 0.8에서 0.095 mm = PhysX 0.096 mm).

### (3) 여전히 설명되지 않는 부분 — **미해결**

- 상대 속도 < 1 mm/s인 pair에서 1 mm 초과 4,201행(예: 51248 seed 1, 14.79 vs 8.58 mm). 통제 실험에서 정지 상자 쌍은
  오차 0이고, sleep 고정 효과는 최대 0.05 mm 수준이라 mm 단위 잔차를 설명하지 못합니다.
- 느린 손가락–큐브 pair 잔차 중앙값 1.8 mm(대부분 손가락 box #2, PhysX 쪽이 큼), 카메라–큐브 pair.
- 후보 가설(검증 안 됨): 관절 링크(articulation)의 내부 자세와 기록된 link_pose의 차이, 쥔 상태의 지속 접촉 manifold,
  기록된 shape local pose와 실제 shape pose의 차이. 모두 B3 계측 없이는 판단 불가.

## 5. B3 계측 계획 (승인 시)

- 대상: 이미 사용된 C1 Development 장면 중 사전 고정 목록(정지 잔차 대표 51248 seed 1, 51157 seed 2 C0, 느린 손가락 잔차
  상위 3장면, S5 사건 51143 seed 2·51270 seed 1) — 결과를 보고 장면을 추가하지 않음.
- 방법: 같은 GPU·드라이버(RTX 4090, 595.99.02)에서 C1 runner로 재실행하며, `px.step`을 감싸 **모든 substep 전후의
  자세·접촉 목록(점, 법선, impulse, separation)**, shape별 offset, controller target(qpos target), qpos/qvel/TCP를 기록.
- 필수 검사: 로거 ON/OFF에서 trace·RNG·결과가 비트 단위로 같아야 함(C1 probe-check 방식). 원본 C1 trace와도 비교하고,
  다르면 "같은 실행 재현"이라고 주장하지 않음.
- 산출: 정렬된 지표 `S6_ALIGNED_POST_HOC`(separation vs substep 시작 자세의 재구성 거리) — 별도 라벨, 공식 S6 대체 아님.
- 비용: 장면 7개 × 해당 arm, ≈ 0.5 h 포함 설치, ≈ US$0.3, 상한 US$1.0.

## 6. 논문에 쓸 수 있는 문장 (영문 초안)

- "In controlled two-box experiments with the same PhysX configuration, the reported contact-point separation equalled the
  analytic distance of the boxes at the pose at the start of the last physics sub-step (|error| ≤ 3e-6 mm), and pairs were
  listed whenever their distance was below the sum of the two contact offsets (0.04 m)."
- "The C1 measurement-validity item S6 failed as pre-registered. Most disagreements are explained by listing distance and a
  one-sub-step timing offset between the contact list and the logged pose; a residual for slowly moving finger–cube and
  static pairs remains unexplained."
- 쓰지 않을 문장: "S6 passes after correction", "contacts are collision-free in continuous time".

## 7. 파일

`s6_static_validation/cases.csv`, `contact_points.csv`, `slow_motion_steps.csv`, `run.json`, `run.log`;
코드 `research_diagnostics/s6_static_validation_001/static_two_box.py`(WSL `~/c1_sim/venv`, sapien 3.0.3, Python 3.11.13).

# RAS 그림과 Supplementary 계획

원칙: 실제 데이터에서 만든 그림만 결과로 씁니다. 본문 그림은 벡터 PDF로 다시 그리고(같은 저장 데이터에서),
축·단위·범례·출처(데이터 파일과 SHA256)를 그림 캡션 또는 Supplementary 표에 적습니다. 기존 PNG는 그림집
(`Paper/figure_atlas/FIGURE_ATLAS_when_recovery_hurts.pdf`, 146개)에 원본 그대로 보관합니다.

토폴로지 관련: 이 프로젝트에는 위상수학적 데이터 분석(TDA)을 한 기록이 없습니다(scope lock이 TDA를 제외).
"공간 점유(occupancy) XY/3D 그림"과 "3D 궤적"은 있으므로 그 이름으로 씁니다. TDA 그림은 넣지 않습니다.

## 본문 그림 (벡터 재작성 대상)

| # | 그림 | 데이터 | 상태 |
|---|---|---|---|
| F1 | 장면과 두 카메라 입력, 트리거와 개입 타임라인 개념도 | assets(개념도), C1 frames | 재작성 필요 |
| F2 | 4분할 결과(rescue/harm)와 H 정의 | — (개념도) | 기존 `fig_outcomes.png` 재작성 |
| F3 | C1 정책별 arm 성공률과 rescue/harm 막대(Wilson 구간) | C1 result.json | 신규 작성 |
| F4 | Ablation: 메모리 출처별 결정 변화와 성공 차 | mechanism_c1_001 tables | 신규 작성 |
| F5 | 대표 사례: rescue(51103)·harm(51157) 시간축(오프셋·손가락·접촉) | trace, case figures | 기존 timeline 기반 재작성 |
| F6 | 공간 점유 XY/3D: 이동 중 손·손가락·큐브 footprint | case_figures occupancy | 기존 PNG → 벡터 재작성 |
| F7 | S6 분해: separation vs 재구성 거리, substep 시점 효과(통제 실험 M과 C1 f=0.8) | s6_static_validation, posthoc_physics_001 | 신규 작성 |
| F8 | (대기) Calibration 인증과 Test 효용 | — | 실행 후 |
| F9 | (대기) 일반화 실험 | — | 실행 후 |

## Supplementary PDF 구성

1. 전체 실험군 결과: 9 arms × 2 정책의 모든 셀, 조건부 표(트리거·이동 조건부)
2. 통계 검정 상세: exact paired test 유도, 교집합-합집합, Holm, 사전 시뮬레이션(FWER ≤ 3.2%, 검정력)
3. 기억 구조와 Ablation: 메모리 갱신 규칙, 확인 횟수, 무효화, leave-one-out 결과
4. Rescue/Harm 사례: 사례 지수 규칙, 14개 대표 사례 그림(그림집에서)
5. 물리적 접촉과 거리: 접촉 목록 구조, impulse 기반 접촉 판정
6. S4/S5/S6: 정의, 공식 판정, 전체 branch 재계산(S4 1, S5 11), 실패 상세
7. Check 4: 첫 step 방향 실패, STEP2PLUS 0/715, 직전 운동 모형
8. Convex hull 검증: 19개 닫힌형 사례, 메시 삼각분할 오차, S4/S5 판정 변화 0
9. 통제 두 상자 실험: 59 cases 전체 표
10. (대기) 일반화 실험 전체 결과
11. 재현성과 해시: plan·archive·코드·결과 SHA256, 같은 GPU 재실행의 trace·프레임 완전 일치(rgb_replay_001)
12. 실패 사례와 미해결 문제: S6 잔차, 51009 오래된 메모리
13. 2D/3D 궤적 그림, 타임라인, 이미지 분석(그림집 섹션 4–5)
14. 수식과 알고리즘: V2 결정 규칙 의사코드, 메모리 갱신, 이동 제한

## 작업 순서 제안

1. F3, F4, F7을 저장 데이터에서 벡터로 작성(무료, 승인 불필요).
2. Supplementary 1–9, 11–14 초안(무료).
3. Calibration·Test·일반화 결과는 승인·실행 후 추가.

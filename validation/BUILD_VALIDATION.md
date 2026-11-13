# 제작 환경 검증 기록

기록 시각(UTC): 2026-11-13T02:23:30.608171+00:00

이 기록은 **ZIP에 포함된 Python 코드의 코어/모의 환경 검사**입니다.
실제 ManiSkill GPU 실험의 결과가 아닙니다.

## 실제 실행한 검사

| 검사 | 결과 |
|---|---|
| pytest 단위·모의 환경 테스트 | **43 passed**, 실패·건너뛰기 없음 |
| Python compileall (`src`, `scripts`, `tests`) | 통과 |
| 모든 제공 Bash 스크립트 `bash -n` | 통과 |
| CLI 메인 도움말 및 11개 하위 명령 도움말 | 종료 코드 0 |
| 작업별 YAML 유효성 검사 | 통과 |
| 원본 episode seed 분할의 중복 검사 | 통과 |
| 제공 JSON 후보와 코드 기본 후보의 일치 | 통과: 총 20개, 무개입을 제외한 보정 후보 19개 |
| 표본 수·비용 계산 도구 실행 | 통과; 비용 입력은 사용법 예시값 |

정확한 pytest 출력은 `pytest.txt`, 기계 판독 결과는 `pytest.xml`에 있습니다.
`environment.json`은 설치 버전과 명령별 결과를 담습니다.

## 검사에 포함된 핵심 동작

이항 위험 검정, Holm 다중검정, 조건부 지표 분모, 원본 episode bootstrap,
분할 누출 검사, 결과 덮어쓰기 거부, 후보별 의사결정,
최대 1회 개입, 원래 horizon 내 후퇴 동작, 상태·정책 큐 모의 복원,
불완전 평가 거부, 합성 자료의 정식 보정 거부,
작은 Q/Joint 모델의 실제 CPU 학습·저장·재로딩을 검사했습니다.

## 제작 환경

- Python: 3.13.5 (main, Jul 15 2026, 20:25:40) [GCC 14.2.0]
- PyTorch: 2.10.0+cpu
- CUDA 사용 가능: **아니오**
- ManiSkill / SAPIEN / gymnasium: **미설치**
- 실제 GPU 서버의 목표 Python: 3.10 또는 3.11
- 실제 GPU 서버의 초기 dependency 조합: `requirements/constraints-linux-py311.txt`

따라서 이 코어 테스트는 GPU 서버의 제안 dependency 조합 전체를 검증한 것이 아닙니다.
서버 설치 후 같은 테스트와 실제 렌더링·정책·복원 검사를 다시 실행해야 합니다.

## 실행하지 않은 항목

1. Vast 인스턴스 대여·결제·접속.
2. CUDA / Vulkan / SAPIEN 실제 렌더링과 물리 실행.
3. 공식 시연 데이터 다운로드·RGB 변환.
4. 기본 Diffusion Policy GPU 학습 및 실제 체크포인트 생성.
5. 실제 StackCube/PegInsertionSide 분기 데이터 수집 및 작업 성공률 측정.
6. 실제 에피소드를 이용한 위험 보정·최종 Test.
7. Docker 이미지 build/run.

외부 소스는 공식 문서 및 고정 commit을 기준으로 연결했지만, 제작 컨테이너에서
해당 GPU 통합 경로를 실행하지 못했습니다. 서버 통합 중 API/드라이버 문제는
원인과 최소 수정 내역을 기록하고 회귀 검사를 추가해야 합니다.

## 연구자가 다음에 확인할 사항

`QUICKSTART.md`의 doctor → RGB smoke → 짧은 정책 학습 → 정책 export → audit 순서를 따릅니다.
audit가 통과하기 전에는 정식 분기 데이터 수집을 시작하지 않습니다.
단위 테스트 통과를 논문 성공률 또는 안전성 보장으로 보고하지 않습니다.

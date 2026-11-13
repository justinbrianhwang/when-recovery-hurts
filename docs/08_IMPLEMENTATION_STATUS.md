# 구현 상태

## 제공한 실행 경로

- 버전 고정 설치 스크립트와 시스템 패키지 스크립트.
- GPU/Vulkan 진단과 실제 렌더링 스모크 명령.
- 공식 RGB DP 학습 launcher, strict observation whitelist, EMA export.
- 단일 환경 CPU 물리 runner, GPU 정책 adapter, 관측·행동 버퍼.
- 상태/정책 복원 audit와 3-way counterfactual 데이터 수집.
- Q/Joint 예측기 학습 및 동일한 전체 episode 후보 평가.
- 지표/신뢰구간 보고, exact binomial + Holm 위험 보정.
- 개발 성공률 기반 후보 선택, 동결 manifest, 최종 Test 전용 잠금.

## 검증 완료 여부

코어·mock 검사 실행 결과는 `validation/BUILD_VALIDATION.md`를 기준으로 합니다.
GPU 렌더링, 실제 ManiSkill 정책 rollout, 실제 DP 학습, 현실 로봇 실행은 제작 환경에서 하지 않았습니다.
Dockerfile도 생성했지만 이미지를 build/run하지 않았습니다.

## 아직 제공하지 않는 완성품

학습된 policy/model, 실제 성공률·악화율 데이터, 논문용 전체 영상(개별 개발 episode 기록 도구는 제공),
다중 GPU/단일 프로세스 vectorized physics, 자동 장애 복구/재개,
OOD 교란 데이터 수집기, 여러 정책 seed의 종합 통계,
FPR 보정 대조 실험, 데이터 예산 sweep과 관측 ablation 자동화입니다.
이들은 후속 구현 범위입니다.

현재 패키지는 “연구를 실행하기 위한 코드와 작업 명세”입니다.
모든 실험이 끝난 완성 연구나, 임의의 Vast 호스트에서 무수정 동작을 보장한 배포판은 아닙니다.

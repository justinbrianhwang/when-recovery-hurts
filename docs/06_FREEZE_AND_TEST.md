# 동결과 최종 Test

## 동결 대상

작업/센서/컨트롤러/성공 정의/horizon, 기본 정책과 metadata, 모형과 정규화,
개입 동작/횟수/후보 registry, split manifest, epsilon/delta, 위험 보정 선택 결과,
실행 코드, 주요 의존성, 실제 pip freeze, upstream commit/diff를 기록합니다.

`hci freeze`는 파일 hash와 provenance를 검사합니다. 선택기의 code/config/heads/정책을 바꾸면
`verify-freeze`가 거부합니다. 보안 제품이나 접근 권한 통제 시스템이 아니라 실험 오염을 막는 운영 장치입니다.
동결 디렉터리와 원시 로그는 외부에도 백업합니다.

## 최종 Test 승인 전 보고

선택한 family별 후보, calibration 분모 n+, Harm 건수, p-value/다중검정 결과,
무개입 fallback 여부, 성공 판정, Test seed 수, 추정 서버 시간, 미확정 설정을 보고합니다.
한 항목이라도 바꿀 예정이면 아직 최종 동결이 아닙니다.

## 명시 승인 후에만 실행

```bash
python -m hci verify-freeze --frozen frozen/stack_001
python -m hci frozen-test --frozen frozen/stack_001 \
  --out runs/final_test_stack_001 --approval RUN_FROZEN_TEST
python -m hci summarize --run runs/final_test_stack_001 \
  --out runs/final_test_stack_report_001
```

이 문서에 명령이 적혀 있다는 사실은 사용자 승인과 다릅니다.
MASTER 프롬프트의 기본 작업 범위에는 Test가 포함되지 않습니다.

## Test 실행 이후

`TEST_STARTED.json`이 생성되어 같은 동결본의 무심코 재실행을 막습니다.
중단·호스트 오류로 복구가 필요하면 원인, 이미 관측한 결과, 수정 범위를 amendment로 기록하고
사용자 승인 아래 별도 실행을 설계합니다. 기존 marker를 몰래 지우지 않습니다.

최종 Test에서 실패한 설정을 고쳐 다시 같은 Test를 돌리고 마지막 결과만 보고하지 않습니다.
추가 개발이 필요하면 새 연구 버전과 별도 held-out 평가를 선언합니다.

## 평가 결과 해석

Test에서 관측한 악화율이 epsilon을 조금 넘었다고 바로 이론의 반례라고 단정하지 않습니다.
보장은 모집단 위험과 calibration 랜덤성에 관한 것이며 Test 자체도 표본오차를 갖습니다.
반대로 경험적 악화율이 낮다고 OOD/실물/각 상태의 안전이 증명되었다고 쓰지 않습니다.

# 문제 해결

| 증상 | 먼저 볼 것 | 하지 말아야 할 것 |
|---|---|---|
| CUDA 되지만 RGB 실패 | Vulkan driver, graphics capability, 실제 nvidia ICD, 컨테이너 생성 옵션 | CUDA 성공만으로 렌더링 검증 완료 선언 |
| `vulkaninfo` 없음 | `vulkan-tools` 설치 | 드라이버 전체 재설치 |
| `.venv` 이미 있음 | 기존 Python/pip 상태, 새 작업 폴더 | 환경 삭제 후 기존 결과 provenance 유지한 척하기 |
| SAPIEN wheel 설치 실패 | Python 3.10/3.11, Linux wheel, pip index 접근 | 호환되지 않는 Python에서 무작정 최신 패키지 혼합 |
| Diffusion Policy import 실패 | upstream commit, editable baseline 설치, diffusers/torchvision 호환 | import 오류를 무시하고 무작위 정책으로 성능 보고 |
| 학습 OOM | 실제 GPU에 올라가는 시연 수, batch size, 평가 env 수 | 데이터 축소 사실 누락 |
| 시연 control mode 불일치 | RGB 재생성 control과 config의 일치 | action dimension을 잘라 맞추기 |
| policy shape mismatch | RGB camera 수·순서, whitelist, 학습 args | 일부 가중치만 무시하고 로드하기 |
| checkpoint 없음 | 공식 `best_eval_*` 및 주기 저장 파일 실제 목록 | 경로만 맞춘 빈 체크포인트 만들기 |
| Refresh=Continue | 남은 queue, 자연 replanning 시점, 단일-step 정책 여부 | 차이가 없는 개입을 다른 방법으로 주장 |
| 복원 audit 실패 | controller, queue, frame history, RNG, solver, scene 구성 | atol만 임의로 키우기 |
| 후보 평가 baseline 불일치 | 동일 seed 재생성·PRNG·동기화 | mismatch를 결과에서 제외 |
| calibration 모두 never | n+, 사건 수, 후보 수, 실제 개입 악화 | 위험 수준을 Test 보고 올리기 |
| freeze mismatch | 코드/환경/패키지/metadata 변경 내역 | manifest의 hash를 수동으로 바꾸기 |

## 초기 코드에서 예상할 수 있는 통합 수정 지점

이 ZIP은 실GPU 환경에서 실행하지 않았습니다. pinned 소스와 API를 대조했지만,
호스트 드라이버와 wheel 해결 결과에 따라 환경 설정 수정이 필요할 수 있습니다.

1. `observations.py`의 whitelist와 task raw observation shape를 실제 출력으로 확인합니다.
2. `policies.py`에서 공식 EMA weight key·학습 args·camera channel 수를 확인합니다.
3. `simulation.py`의 controller state와 environment RNG 복원이 실제 동작하는지 확인합니다.
4. 후퇴는 고정 Panda root와 normalized EE delta control만 지원합니다. 다른 controller는 중단이 정상입니다.
5. 그래픽 라이브러리 문제와 정책 코드 문제를 분리합니다. library mount 문제를 알고리즘 변경으로 해결하지 않습니다.

문제를 수정할 때 기존 결과를 보존하고 새 run ID로 재검증합니다. `templates/AMENDMENT.md`에 원인을 남깁니다.

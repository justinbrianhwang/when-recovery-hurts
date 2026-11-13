# Vast.ai 운영과 환경

## 제안 시작 환경

NVIDIA RTX 3090/4090 계열 한 장, Linux, Python 3.11, 충분한 CPU/RAM/디스크를 가정합니다.
24GB VRAM, RAM 32~64GB, 저장 공간 100GB 이상은 초기 운영을 위한 제안이지 성능 보장이 아닙니다.
최초에는 RGB 시연 20개, 물리 환경 1개, 짧은 학습으로 실제 메모리와 처리량을 측정합니다.
가격이나 특정 호스트 가용성은 이 패키지에 고정하지 않았습니다.

## 컨테이너와 호스트

Vast 환경은 컨테이너와 호스트의 GPU 드라이버를 구분해서 다뤄야 합니다.[S7]
CUDA 연산 성공이 Vulkan 렌더링 성공을 보장하지 않습니다. NVIDIA Container Toolkit의 graphics capability가
필요한 라이브러리 마운트에 관여합니다.[S6]

호스트에서 필요한 라이브러리가 노출되지 않으면 적합한 템플릿/호스트로 바꾸는 편이 맞습니다.
대여 컨테이너 안에서 호스트 커널 드라이버를 업그레이드하거나 재부팅하지 마십시오.
실행 중 export만으로 해결되지 않는 문제를 driver reinstall로 덮지 않습니다.

## 버전

ManiSkill 소스는 `a4a4f9272ad64b1564035874b605ceb687b63ed8`(v3.0.1)로 고정합니다.[S8]
제안 PyTorch 조합은 torch 2.6.0 / torchvision 0.21.0 / CUDA 12.4 wheel입니다.[S9]
이는 최신 버전 주장이나 전체 조합의 실호스트 검증 완료 주장이 아닙니다.
나머지 주요 제약은 `requirements/constraints-linux-py311.txt`에 있고, 실제 설치 결과는 pip freeze로 기록합니다.
최신 GPU 아키텍처를 쓰려면 PyTorch/CUDA 지원을 따로 확인하고 새 환경 버전으로 관리합니다.

## 실행 검사

`nvidia-smi` → `torch.cuda.is_available()` → `vulkaninfo --summary` → 실제 `smoke`.
통과한 각 검사 이름을 기록합니다. doctor만 통과하고 render-tested라고 쓰지 않습니다.

## 원격 장시간 실행

수동으로 tmux를 사용할 수 있지만, 먼저 작은 명령을 foreground로 실행해 실패 로그를 확인합니다.
서버 중단에 대비해 episode별 산출물과 모델 체크포인트를 외부 저장소로 복사합니다.
이 패키지는 Vast API를 호출하거나 자동으로 임대를 시작/연장/종료하지 않습니다.

`backup_results.sh`는 **로컬 백업 아카이브만** 만듭니다. 대여 서버 밖으로 전송하는 작업은 별도입니다.
실제 전송·검증이 끝나기 전에는 인스턴스를 삭제하지 마십시오.
현재 포함된 bootstrap은 실험 출력 폴더를 삭제하지 않습니다.

## 계산량 산정

분기 학습: 원본 수 × checkpoint 수 × 선택지 수 × 반복 수.
후보 평가: episode 수 × (후보 수 + 무개입 재실행 대조군 수).
개입으로 일찍 성공하면 수행 step이 줄어들지만, 가격 계산은 실측 wall time으로 합니다.
보정 후보 20개 × 수백 episode는 파일럿 분기 수집보다 더 비쌀 수 있습니다.
개발 자료에서 후보를 줄일 수 있으나 calibration을 보고 후보를 정하면 안 됩니다.

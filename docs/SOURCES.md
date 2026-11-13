# 기술 출처와 확인 범위

확인일: 2026-11-12. 아래는 설치/API/기본 알고리즘 연결에 사용한 1차 자료입니다.
외부 코드·데이터는 ZIP에 복제하지 않았으며 설치 시 원저자 저장소에서 받아 각 라이선스를 따릅니다.
기존 대화에서 언급한 모든 2026년 논문을 다시 검증한 문헌 보고서는 이 패키지에 포함하지 않았습니다.
연구 문제와 범위는 사용자와 확정한 v0.3 계획을 바탕으로 합니다. 소프트웨어 선택은 아래 공식 자료와 구분하여 기록했습니다.

| ID | 자료 | 사용 범위 |
|---|---|---|
| S1 | ManiSkill Installation | NVIDIA/Vulkan 렌더링과 설치 요구 구분 |
| S2 | Learning from Demonstrations: Setup | 시연 변환, 평가 지표, backend 주의 |
| S3a | Official Diffusion Policy README | 공식 RGB 학습 entrypoint |
| S3b | Pinned `train_rgbd.py` | Agent shape, EMA 체크포인트, args, 메모리 적재 |
| S4 | Reproducibility and RNG | 상태에 포함되지 않는 scene/설정과 재현성 |
| S5 | Pinned replay tool | 시연 변환 CLI와 옵션 |
| S6 | NVIDIA Container Toolkit | graphics/compute/utility capability |
| S7 | Vast Docker Execution Environment | 호스트와 컨테이너 구분 |
| S8 | ManiSkill v3.0.1 commit | immutable source revision |
| S9 | PyTorch previous versions | torch 2.6.0 / torchvision 0.21.0 / cu124 조합 |
| S10 | Learn then Test | 유한 후보 위험 검정 원리 |
| S11 | SciPy distribution docs | binomial CDF와 beta quantile API |
| S12 | ManiSkill observation/controller APIs | 관측 구분, state/queue와 controller 검사 |
| S13 | Task source files | Panda wristcam, 작업 관측/성공 조건 |
| S14 | Diffusion Policy paper/project | 기본 policy 알고리즘 출처 |

## 정확한 URL

S1: https://maniskill.readthedocs.io/en/latest/user_guide/getting_started/installation.html

S2: https://maniskill.readthedocs.io/en/latest/user_guide/learning_from_demos/setup.html

S3a: https://raw.githubusercontent.com/mani-skill/ManiSkill/v3.0.1/examples/baselines/diffusion_policy/README.md

S3b: https://raw.githubusercontent.com/mani-skill/ManiSkill/a4a4f9272ad64b1564035874b605ceb687b63ed8/examples/baselines/diffusion_policy/train_rgbd.py

S4: https://maniskill.readthedocs.io/en/latest/user_guide/concepts/rng.html

S5: https://raw.githubusercontent.com/mani-skill/ManiSkill/v3.0.1/mani_skill/trajectory/replay_trajectory.py

S6: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/docker-specialized.html

S7: https://docs.vast.ai/guides/instances/docker-environment

S8: https://github.com/mani-skill/ManiSkill/commit/a4a4f9272ad64b1564035874b605ceb687b63ed8

S9: https://pytorch.org/get-started/previous-versions/

S10: https://arxiv.org/abs/2110.01052 — Angelopoulos et al., Learn then Test: Calibrating Predictive Algorithms to Achieve Risk Control.

S11a: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binom.html

S11b: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.beta.html

S12a: https://maniskill.readthedocs.io/en/latest/user_guide/concepts/observation.html

S12b: https://raw.githubusercontent.com/mani-skill/ManiSkill/v3.0.1/mani_skill/agents/controllers/pd_ee_pose.py

S12c: https://raw.githubusercontent.com/mani-skill/ManiSkill/v3.0.1/mani_skill/envs/sapien_env.py

S13a: https://raw.githubusercontent.com/mani-skill/ManiSkill/v3.0.1/mani_skill/envs/tasks/tabletop/stack_cube.py

S13b: https://raw.githubusercontent.com/mani-skill/ManiSkill/v3.0.1/mani_skill/envs/tasks/tabletop/peg_insertion_side.py

S14: https://arxiv.org/abs/2303.04137 — Chi et al., Diffusion Policy: Visuomotor Policy Learning via Action Diffusion.

## 출처와 설계 결정의 차이

`CPU physics 1env`, 최초 20 demonstrations, 4개의 checkpoint, 120개 원본, 5% 악화 목표,
한 번만 개입, MLP 폭 128, paired evaluation split은 이 패키지의 시작 설계입니다.
공식 문서에서 논문에 충분하다고 보증한 설정이 아닙니다.

`S3`라고 줄여 쓴 부분은 S3a/S3b를 뜻합니다.

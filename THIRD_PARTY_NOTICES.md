# Third-party notices

The code written for this project (`src/`, `scripts/`, `tests/`, `configs/`, documentation and result tables) is
licensed under the Apache License 2.0; see [`LICENSE`](LICENSE). That license covers this project's own work only.

**Nothing from a third party is relicensed here.** This repository does not contain a copy of any upstream
source tree, model weight, demonstration dataset, 3D asset or font. The install script downloads them from
their original locations, and each remains under its own terms:

| Component | How it is used | Where it comes from | Terms |
|---|---|---|---|
| ManiSkill3 (simulator, tasks, controllers), pinned commit `a4a4f9272ad64b1564035874b605ceb687b63ed8` | installed by `scripts/01_bootstrap_vast.sh`; imported, never copied or edited on disk | https://github.com/mani-skill/ManiSkill | Apache-2.0 (upstream `LICENSE`) |
| Diffusion Policy baseline inside ManiSkill (`examples/baselines/diffusion_policy`) | run as the training entry point; observation paths are replaced in memory only (`src/hci/observations.py`, `src/hci/training.py`) | same repository | same repository terms |
| Demonstration trajectories (`StackCube-v1` motion planning) | downloaded by ManiSkill's `download_demo`; converted locally; never committed | ManiSkill demo hosting | terms of the dataset host; check before redistributing |
| Robot and scene assets | downloaded by ManiSkill on first use | ManiSkill asset hosting and the original asset authors | per-asset terms; check before redistributing |
| PyTorch, diffusers, SAPIEN, Gymnasium, NumPy, SciPy and other packages | installed from PyPI / the PyTorch index | respective projects | respective licenses |

Images under `assets/` that show the simulated scene are renders produced by running ManiSkill; the robot and
scene models visible in them belong to their original authors. Trained policy checkpoints produced by this
project are derived from the downloaded demonstrations and are not published in this repository.

If you use this work, please cite ManiSkill3, Diffusion Policy and Learn then Test; references are in
[`docs/SOURCES.md`](docs/SOURCES.md). Citation is a request of academic practice, not an additional license
condition. The observation whitelist and the intervention/evaluation code differ from the official baseline;
state that difference in any methods section. Do not redistribute access tokens or downloaded assets.

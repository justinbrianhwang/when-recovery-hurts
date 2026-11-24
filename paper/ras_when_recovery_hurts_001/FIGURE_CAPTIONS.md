# Figure and table captions (extracted from the LaTeX sources)

### main — figure `fig:problem` (F1_problem_rescue_vs_harm.pdf)

Rescue and harm. (a) Paired outcome matrix of the Final Test, both policies. (b) A development rescue (scene 51103, policy seed 1): the policy alone fails; after the V2 move the policy's own release places the cube and the episode succeeds. (c) The only development harm of V2 for policy seed 1 (scene 51157): the policy alone drops the cube, regrasps it and succeeds at step 190; with V2 it fails. Each frame shows the stored base-camera (left half) and wrist-camera (right half) RGB images without processing.

### main — table `tab:related` (inline)

Positioning against related work (qualitative; based on each paper's own description).

### main — figure `fig:method` (F2_method.pdf)

Vision-CENTER V2. At the policy's first gripper-open command while a cube is held, the controller combines current base- and wrist-camera estimates with a confirmed memory of the target cube and of the held cube. It abstains unless both estimates exist, the memory is not contradicted by the current image, and the estimated offset is at least 15\,mm and exceeds its reading bound by at least 10\,mm; otherwise it makes a bounded horizontal move and returns control to the unchanged policy. Every arm is branched from the same simulator snapshot.

### main — table `tab:dev` (T2_development_arms.tex)

Development stage (C1, seeds 51100--51299): successes of 200 scenes, rescues and harms against C0, and number of scenes in which the arm moved. The trigger fired in 195 (seed 1) and 191 (seed 2) scenes. For the oracle the column counts the scenes in which it entered its control loop (every trigger); it actually moved in 93 and 102.

### main — table `tab:devtests` (T3_development_tests.tex)

Pre-registered development hypotheses (exact one-sided paired tests; Holm over the two hypotheses at $\alpha=0.05$, i.e.\ the smaller p-value is compared with 0.025 and the larger with 0.05; IU = intersection--union).

### main — figure `fig:memory` (F4_memory_and_ablation.pdf)

Memory and ablation on the development scenes. (a) Source of the target estimate at the first V2 reading in triggered scenes: in about one third of them the target came from the wrist-camera memory, because the held cube hid it at the trigger. (b) Successes of all arms; labels give rescues/harms against C0. The single-frame variant behaves like the wrist-only V1, while removing one memory at a time loses less.

### main — figure `fig:stats` (F3_statistics_calibration_vs_test.pdf)

Confirmatory stages. (a) Calibration (400 scenes per policy): observed conditional harm with the one-sided 95\% Clopper--Pearson upper bound (descriptive) against the pre-registered level $\varepsilon=5\%$ (marked $h^\ast$); the p-values are those of Eq.~\eqref{eq:cert}. (b) Final Test (400 held-out scenes per policy, run once): task success of C0 and V2. (c) Final Test discordant scenes and the p-values of Eq.~\eqref{eq:eff}.

### main — table `tab:confirm` (T4_calibration_and_test.tex)

Calibration certificate and Final Test. Trig.: scenes with a trigger; Acted: scenes in which V2 moved; C0, V2: successes of 400. Calibration tests $h(g)\le 0.05$ (Eq.~\ref{eq:cert}); the Final Test tests efficacy (Eq.~\ref{eq:eff}); both with Holm over the two policies at level 0.05.

### main — figure `fig:control` (F6_control_check4.pdf)

Commanded versus actual motion. (a,b) Scene 51157, V2: commanded and actual horizontal TCP displacement per move step and their direction cosine. (c) All 715 acted branches in development (post hoc): the first step often deviates in direction because the arm is still moving with the policy's previous command (110 branches below the pre-declared 0.9 threshold); from the second step on, no branch falls below it.

### main — figure `fig:spatial` (F5_spatial_51157.pdf)

The harm case 51157 (policy seed 1, development). (a) Footprints of the held cube A (blue) and target cube B (grey) at the trigger (solid) and at the end of the V2 move (dashed), the TCP paths and the target estimate. (b) Three-dimensional view at the trigger: cube boxes from the logged poses, TCP paths of C0 (grey) and V2 (orange), and the robot link frames. (c) Heights of the TCP and of cube A after the trigger.

### main — table `tab:physics` (T5_physics_validity.tex)

Measurement-validity checks on the development record (post hoc unless stated). S4--S6 were declared before the analysis; their official status is kept.

### main — figure `fig:s6` (F7_physics_s6_b4.pdf)

Contact-separation check S6. (a) Rows over 1\,mm by pair kind (official FAIL). (b) Logging every physics sub-step on 11 used scene-runs: comparing with the pose before the last sub-step removes most disagreements of fast pairs. (c) Static boxes with known geometry: the PhysX separation equals the analytic gap before the last sub-step within 0.05\,mm.

### main — figure `fig:hull` (F8_hull_sensitivity.pdf)

Collision-geometry sensitivity. (a) S4 minimum robot--target distance on convex hulls for all 715 acted branches; the change with respect to the logged meshes is zero in every branch, and the only negative value is an oracle branch. (b) Volume excess of the convex hull over the logged mesh per link (at most 0.3\%).

### main — figure `fig:gen` (F9_generalization_plainconv.pdf)

Unchanged V2 on a frozen policy of a different architecture (PlainConv encoder; development seeds 51300--51499). (a) Successes of 200 scenes. (b) Rescues (up) and harms (down, hatched) against C0 for the development policies and the PlainConv policy.

### main — table `tab:gen` (T6_generalization_plainconv.tex)

PlainConv policy, development level (200 scenes; 95\% Clopper--Pearson intervals).

### supplementary — table `tab:s-desc` (TS2_descriptive_posthoc.tex)

Post hoc descriptive quantities for calibration and Final Test. ``Moved among $Y_0{=}1$'': baseline successes in which V2 made a move.

### supplementary — table `tab:s-const` (inline)

Constants of Vision-CENTER V2 (frozen before calibration; source \texttt{src/hci/vision\_center\_v2.py}, \texttt{vision\_center.py}, \texttt{recovery\_v1\_oracle.py}).

### supplementary — table `tab:s-dec` (TS1_decisions_by_stage.tex)

V2 decision at the trigger in every scene of every stage (raw scene records). s1, s2: policy seeds.

### supplementary — figure `fig:s-dec` (S2_decisions_by_stage.pdf)

Share of each V2 decision per stage and policy (log scale). The proportions are stable from development to calibration and test: V2 acts in 16--21\% of scenes, and about two thirds of scenes are already aligned at the trigger.

### supplementary — figure `fig:s-frames` (S1_frames_51157.pdf)

Scene 51157, policy seed 1. Rows 1--2: all 13 stored frames of the V2 move (trigger at step 111). Rows 3--4: frames after the hand-back for C0 and V2. Each frame is base camera (left) and wrist camera (right), unprocessed. C0 regrasps the dropped cube and succeeds at step 190; with V2 the cube lies beside the target and the episode ends at the horizon.

### supplementary — figure `fig:s-mem` (S3_memory_timeline_51157.pdf)

Accepted memory observations before the trigger in scene 51157 (hand 18, base 89, hold 14; no conflict). The target was seen by the wrist camera early in the episode and by the base camera until step 106.

### supplementary — table `tab:s-hash` (inline)

Frozen artefacts.

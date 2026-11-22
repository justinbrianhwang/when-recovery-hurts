"""Figure atlas: every figure already made from real project data (repository assets/results and the Drive project
folder), one per page, grouped by stage, with its source path, sha256 and a caption of what it shows. Nothing is
re-plotted or generated; the PNG files are embedded as they are (raster). Test/mock demo figures are excluded.

usage: python -P paper_tools/make_figure_atlas.py --out <pdf>   (the Drive project folder is found under G:\\<first entry>)
"""
from __future__ import annotations

import argparse, hashlib, json, os, re, sys
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def drive_root():
    g = Path('G:/'); top = g / '내 드라이브'   # 'My Drive' in the Korean Drive client
    p = top / 'Independent Research' / 'Robotics' / 'Robotics and Autonomous Systems' / 'Robot Harm' if top else None
    return p if p and p.exists() else None


sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()

# (section, title, pattern on the posix relative path, caption)
SECTIONS = [
    ('1. Task, policy and intervention setup', [
        ('fig_sim_scene', 'Schematic of the simulated scene: robot arm with a wrist camera, a fixed base camera, two cubes on a table, and the two camera views as insets.'),
        ('stackcube_scene', 'StackCube-v1 in ManiSkill3: Panda arm, red cube (to be stacked), green cube (target).'),
        ('stackcube_policy_inputs', 'The two 128x128 RGB inputs of the policy and selectors (base camera, wrist camera).'),
        ('stackcube_filmstrip', 'Five frames of the StackCube task: approach, grasp, lift, move, place.'),
        ('fig_three_choices', 'From one checkpoint state the episode branches into Continue, Refresh and Retreat-Retry (v0.3 protocol).'),
        ('fig_outcomes', 'Paired outcomes: both succeed, harm, rescue, both fail. Harm rate = harm / baseline successes.'),
        ('training_eval', 'Training-time evaluation: with 20 demonstrations the policy never succeeds in 31 evaluations of 5 episodes; with 99 demonstrations it succeeds once at 14k, once at 22k and twice at 24k iterations.'),
        ('retreat_tracking', 'The configured retreat path reaches 5 cm after 10 steps; the measured upward TCP displacement reaches only 1.9 cm after 12 steps.'),
        ('retreat_closed_loop', 'Measured upward TCP displacement per closed-loop retreat step at 30 states: about 1.8 mm per step, stopping inside the 5 mm arrival band after 24-28 steps, before the 40-step budget.'),
        ('restore_divergence', 'After a set_state restore the state difference grows from about 1e-4 (one step) to 6e-3 (eight steps), far above the 1e-6 audit tolerance; prefix-replay restore gives exactly zero.'),
        ('banner', 'Project banner (illustration only).'),
    ]),
    ('2. Stage 1C: input diagnostics (file replay vs live RGB)', [('stage1c/images/', 'Per-pixel difference maps and examples between demonstration-file RGB and live re-rendered RGB (x8/x16 enlarged); F = failed, R = replayed, Fdemo = demonstration scenes.')]),
    ('3. Vision-CENTER v2 (Phase B) preparation and results', [
        ('fig1_hand_view_before_trigger', 'Hand-camera view before the trigger: what the v2 memory sees.'),
        ('fig2_observation_timeline', 'Observation timeline of the video episodes: when target and held cube are visible.'),
        ('fig3_base_camera_error', 'Base-camera target estimate error (used to set the v2 bound).'),
        ('fig4_carry_height', 'Carry height and pushed-target cases behind the LOW_CARRY / side-contact rules.'),
        ('fig_b2_results', 'Phase B2 (pilot, new scenes): outcomes per arm.')]),
    ('4. C1 representative cases: occupancy (XY / 3D) and physical timelines', [
        ('occupancy_xy_3d', 'B: XY occupancy at true scale (footprints of hand, fingers, cubes); C: 3D collision shapes at the end of the move. POST-HOC GROUND TRUTH from the logged simulator state.'),
        ('case_figures_001/.*timeline', 'D: physical behaviour per step: filled = observed contact (nonzero impulse), hollow = listed PhysX pair only; distances, forces, cube motion.')]),
    ('5. C1 trajectory audit (first scenes of each policy)', [
        ('3d_trajectory', '3D trajectories of TCP and both cubes for every arm (POST-HOC GROUND TRUTH).'),
        ('xy_alignment', 'XY alignment of the held cube over the target per arm.'),
        ('timeline_actions_gripper', 'Per-step actions and gripper commands per arm; trigger and control phase marked.'),
        ('after_trigger_outcome', 'Outcome after the trigger per arm.'),
        ('rgb_with_posthoc_ground_truth', 'Recorded RGB at the trigger with post-hoc ground-truth overlay.'),
        ('rgb_recorded', 'Recorded RGB at the trigger (as stored).')]),
    ('6. C1 post-hoc physics diagnostics (S6, Check 4)', [
        ('s6_by_separation_and_motion', 'S6 disagreement by PhysX separation and relative motion.'),
        ('s6_time_shift_sensitivity', 'S6: reconstruction at poses shifted toward the previous row (time-offset association; explanatory only).'),
        ('s6_extreme_cases', 'S6 extreme cases.'),
        ('check4_v2_stepwise', 'Check 4 V2 per step: direction cosine and progress.'),
        ('check4_first_step_vectors', 'Check 4 first-step command vs achieved motion vectors.'),
        ('check4_first_step_model', 'Check 4 first-step models (LOSO) with prior motion.')]),
    ('7. C1 S4/S5 convex-hull sensitivity', [
        ('s4_logged_vs_hull', 'S4 minimum distance per acted branch: logged mesh vs convex hull (identical).'),
        ('s5_observed_pairs', 'S5: reconstructed distance of observed contact pairs; mesh pairs logged vs hull.'),
        ('mesh_shapes_vs_hull', 'Logged PhysX meshes vs the convex hull of their vertices.')]),
    ('8. README research GIF key frames (re-run RGB, trajectory identical to the record)', [('research_gifs_full_rgb/[^/]*\\.png', 'Key frame of a README GIF: RGB of the re-run (rgb_replay_001), top view, offset and finger timeline, decision record.')]),
]
EXCLUDE = re.compile(r'(runner_mock_demo|independent_review_bundle|/Paper/|/research_gifs/(?!.*full)|upstream_geometry|superseded)')


def collect():
    files = {}
    for base, tag in ((REPO, 'repo'), (drive_root(), 'drive')):
        if base is None: continue
        for p in base.rglob('*.png'):
            rel = p.relative_to(base).as_posix()
            if EXCLUDE.search('/' + rel): continue
            h = sha(p); rank = 1 if rel.startswith('runs/') else 0            # a duplicate keeps the committed/Drive path, not the scratch copy under runs/
            if h not in files or rank < files[h][3]: files[h] = (tag, rel, p, rank)
    return files


def main():
    a = argparse.ArgumentParser(); a.add_argument('--out', required=True); a = a.parse_args(); out = Path(a.out)
    if out.exists(): raise SystemExit(f'{out} exists')
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt; from matplotlib.backends.backend_pdf import PdfPages; from PIL import Image
    files = collect(); used = set(); plan = []
    for sec, items in SECTIONS:
        rows = []
        for pat, cap in items:
            for h, (tag, rel, p, _r) in sorted(files.items(), key=lambda x: x[1][1]):
                if h in used or not re.search(pat, rel): continue
                used.add(h); rows.append((tag, rel, p, h, cap))
        plan.append((sec, rows))
    rest = [(tag, rel, p, h, 'Other project figure from real data (policy-input audit or Stage 1C probe; scratch folder runs/, not in Git).' if rel.startswith('runs/') else 'Other project figure.') for h, (tag, rel, p, _r) in sorted(files.items(), key=lambda x: x[1][1]) if h not in used]
    if rest: plan.append(('9. Other figures', rest))
    index = []
    with PdfPages(out) as pdf:
        fig = plt.figure(figsize=(11.69, 8.27)); fig.text(0.05, 0.9, 'When Recovery Hurts: figure atlas', fontsize=22, weight='bold')
        fig.text(0.05, 0.84, f'Every figure made from real project data so far, as stored (no re-plotting). Built {date.today().isoformat()}.', fontsize=11)
        fig.text(0.05, 0.80, 'POST-HOC GROUND TRUTH panels come from logged simulator state and were never a controller input. C1 results are Development data (not Calibration or Test).', fontsize=9, color='#a00000')
        y = 0.72
        for sec, rows in plan:
            fig.text(0.07, y, f'{sec}  ({len(rows)} figures)', fontsize=11); y -= 0.04
        pdf.savefig(fig); plt.close(fig)
        page = 2
        for sec, rows in plan:
            fig = plt.figure(figsize=(11.69, 8.27)); fig.text(0.05, 0.5, sec, fontsize=20, weight='bold'); pdf.savefig(fig); plt.close(fig); page += 1
            for tag, rel, p, h, cap in rows:
                im = Image.open(p).convert('RGB'); w, hh = im.size
                fig = plt.figure(figsize=(11.69, 8.27)); ax = fig.add_axes([0.03, 0.12, 0.94, 0.82]); ax.imshow(im); ax.axis('off')
                fig.text(0.03, 0.965, f'{sec}  |  {Path(rel).name}', fontsize=9, weight='bold')
                fig.text(0.03, 0.075, cap, fontsize=8.5, wrap=True)
                fig.text(0.03, 0.035, f"source: {'DRIVE:Robot Harm/' if tag == 'drive' else 'repo:'}{rel}   ({w}x{hh} px)   sha256 {h[:16]}", fontsize=7, color='#444')
                fig.text(0.97, 0.01, str(page), fontsize=7, ha='right'); pdf.savefig(fig, dpi=200); plt.close(fig)
                index.append({'page': page, 'section': sec, 'source': ('drive:' if tag == 'drive' else 'repo:') + rel, 'sha256': h, 'caption': cap}); page += 1
    Path(str(out) + '.index.json').write_text(json.dumps({'pdf': out.name, 'figures': index}, indent=1), encoding='utf-8')
    print(len(index), 'figures', out.stat().st_size // 1_000_000, 'MB')


if __name__ == '__main__':
    main()

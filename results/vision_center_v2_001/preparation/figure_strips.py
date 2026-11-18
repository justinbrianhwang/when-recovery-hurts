"""Vision-CENTER v2, Phase A, figure: what the hand camera saw of the target before the trigger in the four named scenes.
Frames are RECONSTRUCTED from logged ground-truth states (reconstruct.py; offline only), except the last column, which is
the real stored trigger frame. A green outline marks steps where the unchanged v1 target fit accepts the reconstructed frame.
Usage (repository root): python results/vision_center_v2_001/preparation/figure_strips.py <diagnostic frames dir> <out.png>"""
import gzip, json, os, sys
import numpy as np
from PIL import Image, ImageDraw
sys.path.insert(0, "src"); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hci import vision_center as V1
import reconstruct as RC, reconstruction_audit as RA
R = "results/vision_center_001/run2/pilot/"; SC = 2; W = V1.IMG * SC; PAD = 6; TOP = 20; LEFT = 150
rm = RA.robot_mask(sys.argv[1]); rows = []
for s in (50813, 50872, 50817, 50846):
    r = json.load(open(R + f"scenes/{s}.json", encoding="utf-8")); T = r["trigger"]["step"]; v = [c for c in r["cells"] if c["arm"] == "VISION_CENTER"][0]["control"]["decision"]
    with gzip.open(R + f"traces/trace_{s}.json.gz", "rt", encoding="utf-8") as fh: st = [r["initial_state"]] + json.load(fh)["traces"]["trunk"]
    g = next(i for i in range(T + 1) if st[i]["is_cubeA_grasped"]); steps = [0, 8, 16, g, (g + T) // 2, T - 20, T - 10, T - 4, T]; cells = []
    for i in steps:
        x = st[i]; img = RC.render_hand(x["tcp"], x["tcp_q"], x["cubeA"], x["cubeA_q"], x["cubeB"], x["cubeB_q"], rm.get(sum(x["finger_qpos"])))
        f = V1.target_partial_fit(img, V1.cam_hand(np.array(x["tcp"] + x["tcp_q"], float)), "hand_camera"); cells.append((i, img, bool(f and f["valid"]), "grasp" if i == g else ("trigger" if i == T else "")))
    cells.append((T, np.load(R + f"frames/hand_{s}.npz")["frames"][0], None, "trigger, real frame")); rows.append((s, v, cells))
n = len(rows[0][2]); H = len(rows) * (W + TOP + PAD) + PAD; canvas = Image.new("RGB", (LEFT + n * (W + PAD) + PAD, H), (252, 252, 250)); d = ImageDraw.Draw(canvas)
for j, (s, v, cells) in enumerate(rows):
    y = PAD + j * (W + TOP + PAD); d.text((8, y + TOP + 40), f"scene {s}", fill=(20, 20, 20)); d.text((8, y + TOP + 58), "v1 at the trigger:", fill=(90, 90, 90)); d.text((8, y + TOP + 74), v.replace("_ESTIMATE", "").replace("_", " ").lower(), fill=(90, 90, 90))
    for k, (i, img, ok, tag) in enumerate(cells):
        x = LEFT + k * (W + PAD); canvas.paste(Image.fromarray(img).resize((W, W), Image.NEAREST), (x, y + TOP)); d.text((x + 2, y + 4), f"step {i}" + (f" ({tag})" if tag else ""), fill=(20, 20, 20))
        if ok: d.rectangle([x - 2, y + TOP - 2, x + W + 1, y + TOP + W + 1], outline=(0, 140, 60), width=3)
canvas.save(sys.argv[2]); print("saved", sys.argv[2], canvas.size)

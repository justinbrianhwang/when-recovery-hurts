"""Vision-CENTER v2, Phase A: reconstruction of the hand-camera class image from a logged simulator state.
OFFLINE ANALYSIS ONLY: it reads ground-truth cube poses and is never imported by a controller. Two cubes drawn as
flat-colour faces (the most upward face lit), robot pixels from an empirical mask indexed by finger width. It answers
"what would the unchanged estimator have seen of the TARGET at step t" for scenes without stored video. The held cube is
NOT reproduced well enough for the held-cube fit (see reconstruction_audit.json) and is used only as an occluder."""
import numpy as np, sys
sys.path.insert(0, 'src')
from hci import vision_center as VC
BG = (200, 120, 70); COL = {('A', 1): (230, 0, 0), ('A', 0): (150, 0, 0), ('B', 1): (0, 230, 0), ('B', 0): (0, 150, 0)}; ROBOT = (128, 128, 128)
_C = np.array([[a, b, c] for c in (-1, 1) for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1))], float) * VC.CUBE_HALF_M
_F = ((0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7))          # outward-facing order


def faces(p, q, name):
    R = VC.quat_R(q); W = np.asarray(p, float) + _C @ R.T; out = []
    for f in _F:
        P = W[list(f)]; n = np.cross(P[1] - P[0], P[2] - P[1]); n = n / np.linalg.norm(n); out.append((P, n, name))
    return out


def render_hand(tcp, tcp_q, A, A_q, B, B_q, robot_mask=None):
    cam = VC.cam_hand(np.array(list(tcp) + list(tcp_q), float)); Rc, pc = cam; img = np.empty((VC.IMG, VC.IMG, 3), np.uint8); img[:] = BG; todo = []
    fs = faces(A, A_q, 'A') + faces(B, B_q, 'B'); lit = {}
    for nm in 'AB':                                                           # the face whose normal is most upward is the lit one
        zs = [n[2] for P, n, m in fs if m == nm]; lit[nm] = max(zs)
    for P, n, nm in fs:
        c = P.mean(0)
        if np.dot(n, c - pc) >= 0: continue                                    # faces away from the camera
        X = (P - pc) @ Rc
        if (X[:, 2] <= 1e-3).any(): continue
        uv = np.stack([VC.FOCAL_PX * X[:, 0] / X[:, 2] + VC.CENTRE_PX, VC.FOCAL_PX * X[:, 1] / X[:, 2] + VC.CENTRE_PX], 1)
        todo.append((float(np.linalg.norm(c - pc)), uv, COL[(nm, int(n[2] == lit[nm] and n[2] > 0.7))]))
    for d, uv, col in sorted(todo, key=lambda x: -x[0]): img[VC.raster(uv)] = col
    if robot_mask is not None: img[robot_mask] = ROBOT
    return img


class RobotMask:
    """P(robot pixel | finger width) learnt from real frames: a pixel is robot when it is grey in most frames of that width
    in which it is not covered by a cube colour."""
    def __init__(self, step=0.004): self.step = step; self.g = {}; self.n = {}

    def add(self, img, width):
        k = VC.classes(img); b = int(round(width / self.step)); free = ~(k['red'] | k['green'])
        self.g[b] = self.g.get(b, 0) + (k['grey'] & free); self.n[b] = self.n.get(b, 0) + free

    def get(self, width):
        b = int(round(width / self.step)); b = min(self.g, key=lambda x: abs(x - b)); return self.g[b] / np.maximum(self.n[b], 1) > 0.5

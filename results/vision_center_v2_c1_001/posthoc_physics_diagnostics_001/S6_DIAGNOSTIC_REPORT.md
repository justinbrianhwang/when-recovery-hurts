# S6 measurement validity — post-hoc diagnosis (POST_HOC_EXPLORATORY)

**The official S6 result stays FAIL for both policies.** Nothing here changes it, the 1 mm tolerance, or any C1 result.
Every number below was computed from the stored C1 Development records by `research_diagnostics/c1_physics_001/`; no
simulation was run.

## 1. What S6 compares (from the code, not assumed)

- `physx_sep`: the **minimum over the PhysX contact points** of one listed shape pair of the point `separation`, read
  from the contact list of the **last of the five PhysX sub-steps** of a control step (`hci.spatial_probe`, `scene.get_contacts()`).
- `recon`: the exact **minimum signed distance** between the two logged collision shapes posed at the **logged poses of
  that control step's row t** (`hci.spatial_links.gap` → `hci.spatial_geometry.signed_distance`).
- Population: every listed pair with `physx_sep > 0`, all arms and phases; FAIL if any |physx_sep − recon| > 1 mm.

These are **not the same physical quantity by construction**: a PhysX point separation is a per-manifold-point value
(measured along the contact normal) produced by the narrow phase at some moment of the sub-step, while `recon` is a
minimum distance at the end-of-step pose. S6 treated them as equal within 1 mm; the data show when they are and when not.

## 2. Reproduction (A1)

| Policy | Pair-rows | |diff| > 1 mm | Official |
|---|---|---|---|
| train_seed_1 | 293,614 | 27,376 | FAIL |
| train_seed_2 | 286,925 | 26,891 | FAIL |

Reproduced exactly from `mechanism_c1_001/tables/s6_pair_rows.csv.gz` (sha256 checked against the analysis manifest).
Row mapping back to the raw trace: the existing reconstruction recomputed at the raw row equals the stored value in all
8,394 recomputed rows (max error 1e-16 m), and the minimum over the pair's raw PhysX points equals the stored `physx_sep`
in all 8,394.

## 3. Distribution and conditions (all 580,539 rows, both policies; rows are not independent — descriptive)

| Group | n | > 1 mm | fraction | |diff| median / p99 / max (mm) |
|---|---|---|---|---|
| all | 580,539 | 54,267 | 0.094 | 0.057 / 3.13 / 16.5 |
| recorded nonzero impulse | 7,277 | 612 | 0.084 (scene bootstrap 0.063–0.110) | 0.004 / 4.83 / 9.5 |
| zero impulse (proximity-only listing) | 573,262 | 53,655 | 0.094 (0.087–0.100) | 0.059 / 3.11 / 16.5 |
| impulse flag missing | 0 | – | – | – |
| PhysX sep ≤ 1 mm (near contact) | 15,811 | 205 | 0.013 (0.009–0.018) | 0.015 / 1.16 / 4.5 |
| PhysX sep 1–5 mm | 50,394 | 2,156 | 0.043 | 0.003 / 1.90 / 5.7 |
| PhysX sep 30–40 mm | 116,883 | 18,906 | 0.162 | 0.163 / 3.58 / 11.7 |
| relative speed < 1 mm/s (static) | 154,513 | 4,201 | 0.027 | 0.0007 / 1.45 / 9.7 |
| relative speed ≥ 100 mm/s | 83,940 | 32,994 | 0.393 (0.375–0.409) | 0.81 / 4.86 / 16.5 |
| |(v_a−v_b)·n·dt_sub| < 0.01 mm | 230,875 | 7,228 | 0.031 | 0.002 / 1.77 / 9.8 |
| (v_a−v_b)·n·dt_sub < −0.5 mm | 45,882 | 13,043 | 0.284 | signed median **+0.78** |
| (v_a−v_b)·n·dt_sub > +0.5 mm | 49,630 | 20,150 | 0.406 | signed median **−0.81** |
| angular speed of body a ≥ 1 rad/s | 2,657 | 725 | 0.273 | 0.26 / 5.93 / 13.5 |
| pair already listed & separated at t−1 | 529,197 | 43,957 | 0.083 | 0.048 / 3.05 / 16.5 |
| pair newly listed (not at t−1) | 51,342 | 10,310 | 0.201 | 0.39 / 3.70 / 13.5 |

By kind (scene-bootstrap 95% intervals in `S6_DIAGNOSTIC_SUMMARY.json`): fingers–cubeA 23,192 / 270,022 (0.086),
fingers–cubeB 15,368 / 129,124 (0.119), fingers–table 8,449 / 72,121 (0.117), cubeA–table 2,523 / 15,255 (0.165),
cubeA–cubeB 2,012 / 34,834 (0.058), hand–cubeA 1,612 / 46,579 (0.035), hand–cubeB 798 / 9,880 (0.081),
camera–cubeA 182 / 609, cubeB–table 131 / 1,486, hand–table 0 / 621, camera–cubeB 0 / 8. Full tables:
`S6_ERRORS_BY_GROUP.csv` (policy × grouping, with the worst row of each group and its scene/arm/step/shape pair).

**Readings.** (i) 82.6% of the rows over 1 mm are pairs PhysX lists 10–40 mm apart (within the 0.04 m listing distance),
i.e. proximity listings, not contacts; near contact (≤ 1 mm) only 1.3% of rows exceed 1 mm. (ii) The sign of the
difference follows the relative normal motion within one sub-step, and the over-tolerance fraction rises with relative
speed. (iii) A recorded nonzero impulse does not make the agreement better or worse on average.

## 4. Contact listing distance

The largest listed separation is 0.03999999 m (p99.9 0.0399975 m), consistent with `contact_offset = 0.02` per shape
(two shapes: 0.04) in the pinned ManiSkill `SceneConfig` (a4a4f92: contact_offset 0.02, rest_offset 0, PCM on, TGS on,
15 position iterations). **Not logged by the run**: the actual per-shape offsets, PCM manifold state, contact caching.

## 5. Geometry of the reconstruction (independent check, A2/C)

Independent reconstruction (own posing with scipy rotations; GJK / Qhull Minkowski facets; `geometry_reference.py`) on
the 8,394 recomputed rows: identical to the existing one (≤ 3e-13 mm) for every box–box kind. **It differs for pairs with
the hand** (hand–cubeA max 0.24 mm, hand–cubeB max 0.22 mm; 7 and 4 sampled over-tolerance rows would be within 1 mm with
the reference; no sign disagreements). Cause, found in the data: the PhysX-cooked convex meshes are logged as vertices plus
a triangulation that is **not the convex hull of those vertices** (hand: 93 vertices, 182 triangles, mesh volume 0.13%
below the hull; same for every arm link mesh), while `hci.spatial_geometry` enumerates the logged triangles
(`GEOMETRY_VALIDATION_REPORT.md`). This is a real reconstruction error, small (≤ 0.24 mm) and limited to the hand among S6
pairs; it is not the main source of the S6 FAIL.

## 6. Time-axis sensitivity (A2, EXPLANATORY_ONLY)

Rule-selected subset: per policy × kind × over/within-tolerance, a fixed-seed sample of 250 rows (all if fewer) plus the
extremes; 8,394 rows in 400 scene-runs. The same existing reconstruction was repeated with every moving body posed at a
fraction f of the control step between rows t−1 (f = 0) and t (f = 1) (linear position + slerp: a constant-velocity
**assumption**), and with a one-sub-step back-step using the logged cube velocities.

Stratified rows over 1 mm at t (n = 3,813): within 1 mm at **f = 0.8** (one sub-step = the start of the last sub-step)
**69%**; at the best f **79%**. Rows within 1 mm at t (n = 4,522): 99% stay within at f = 0.8.

| Kind (rows > 1 mm at t) | median |diff| f=0 / **0.8** / 1 (mm) | within 1 mm at f=0.8 |
|---|---|---|
| fingers–table | 4.91 / **0.05** / 1.21 | 100% |
| cubeA–table | 5.88 / **0.07** / 1.55 | 87% |
| cubeA–cubeB | 6.20 / **0.12** / 1.52 | 85% |
| hand–cubeB | 4.61 / **0.08** / 1.24 | 85% |
| hand–cubeA | 5.06 / **0.15** / 1.23 | 75% |
| cubeB–table | 5.44 / **0.08** / 1.59 | 67% |
| fingers–cubeA | 4.04 / **1.13** / 1.46 | 42% |
| fingers–cubeB | 3.02 / **1.61** / 1.84 | 34% |
| camera–cubeA | 1.54 / **1.35** / 1.40 | 6% |

Some cube–table extremes are reproduced to 1e-7 m by the back-step with the logged cube velocity (e.g. seed 1 scene
51143 CENTER_ORACLE t 183 cubeA–table: PhysX 10.45 mm, recon at t 1.26 mm, back-step 10.45 mm), but in general the
back-step does no better than f = 0.8 (69% within).

**Reading (association, not identification).** For most pair kinds, the PhysX separations agree far better with poses
about one sub-step before the logged pose than with the logged pose itself. That is consistent with the contact list being
generated before the last sub-step's position update while the probe logs the poses after it. It is **not** shown that
this is PhysX's internal order (not observed), and it does not explain everything.

## 7. Counter-examples and what stays unexplained

- **Static pairs.** 4,201 rows over 1 mm have relative speed < 1 mm/s; a time offset cannot produce them. Extreme: seed 1
  51248 trunk t 65, right finger box #0 – cubeA, PhysX 14.79 mm vs reconstruction 8.58 mm at 0.9 mm/s; seed 2 51157 C0
  t 168, the same shape pair, 14.32 vs 8.70 mm at 0.4 mm/s.
- **Slow finger–cube rows.** Of the sampled finger rows over 1 mm with relative speed < 0.1 m/s (545 rows; 481 involve
  finger box #2, the 30°-rolled "diagonal" box), the residual at f = 0.8 is still median 1.8 mm, and PhysX reports the
  larger value in 95%.
- **Definition test** (`S6_NORMAL_RAY_SUMMARY.json`, 9,058 PhysX points): each point lies on one shape (median 0.002 mm).
  For points of the slow finger rows, the point separation equals neither the Euclidean distance to the other shape
  (median difference 1.8 mm) nor the distance along the normal to its supporting plane (3.4 mm); it equals the ray distance
  along the normal (0.05 mm) only for the 56% of points whose ray hits the other shape. For other rows at f = 0.8, the
  Euclidean distance matches within 0.05 mm for about half of the points.
- `camera–cubeA` (182 rows over 1 mm) is not explained by any tested shift.

## 8. Hypotheses and their status

| Hypothesis | Status | Evidence / falsifier |
|---|---|---|
| H-S6-1 Contact points are computed about one sub-step before the logged pose (time offset) | **Supported as an association** for most kinds; **not the only cause** | Median residual drops from ~5 mm to < 0.2 mm at f = 0.8 for 6 of 9 kinds; falsified for static rows, slow finger rows and camera rows |
| H-S6-2 The reconstruction geometry is wrong (shape, local pose, quaternion order, scaling, index) | **Refuted for boxes; confirmed small error for convex meshes** | 19 closed-form cases + transform tests pass to 1e-9 m; box rows match an independent path to 3e-13 mm; hand mesh differs ≤ 0.24 mm |
| H-S6-3 S6 compares two different quantities (manifold point separation vs minimum distance) | **Supported in part** | Separations of slow finger rows are not minimum distances; some match a ray along the normal; the definitional difference was not settled for every point |
| H-S6-4 PCM contact caching / persistent manifolds keep stale separations | **UNKNOWN** (not observable) | Newly listed pairs disagree more (0.20) than persisting ones (0.08), which argues against stale persistence as the main cause, but the manifold state is not logged |
| H-S6-5 The disagreement concerns physical contacts | **Mostly refuted** | 99% of rows have zero impulse; 83% of over-tolerance rows are 10–40 mm apart |

What the data **cannot** answer: PhysX's call order inside a sub-step, contact-offset/rest-offset values actually used,
PCM manifold contents and refresh rule, and the sub-step states. These need instrumentation (`FUTURE_DIAGNOSTIC_OPTIONS.md`).

## 9. Coverage note

The S6 population uses finite differences between consecutive rows of the same arm list, so the first row of each arm
after the trigger (t = trigger + 1) is not a pair-row (the trunk covers t ≤ trigger). This is how the existing S6 was
defined; it was not changed.

## 10. Files

`S6_ERRORS_BY_GROUP.csv`, `S6_DIAGNOSTIC_SUMMARY.json`, `S6_RECOMPUTED_SUBSET.csv.gz` (8,394 rows with record/trace sha256),
`S6_TIME_SHIFT_BY_KIND.csv`, `S6_EXTREME_CASES.csv` (10 rule-selected cases), `S6_NORMAL_RAY_CHECK.csv`,
`S6_NORMAL_RAY_SUMMARY.json`, figures `s6_by_separation_and_motion.png`, `s6_time_shift_sensitivity.png`,
`s6_extreme_cases.png`.

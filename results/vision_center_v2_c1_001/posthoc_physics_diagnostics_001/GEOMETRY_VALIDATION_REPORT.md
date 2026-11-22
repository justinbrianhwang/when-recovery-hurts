# Geometry validation — known truth and an independent path (POST_HOC_EXPLORATORY verification)

## 1. What the existing calculator does (read from `src/hci/spatial_geometry.py`)

- Separated shapes: exact minimum over vertex–triangle distances both ways (Ericson 5.1.5) and edge–edge distances
  (Ericson 5.1.9), over the shape's **triangles and edges as given**.
- Overlapping shapes: separating-axis test over the shapes' triangle normals and all edge-direction cross products; the
  smallest overlap is returned as the penetration depth (negative signed distance).
- Not surface point sampling, and not GJK/EPA. Poses: logged quaternions (w, x, y, z) via `quat_to_R`; world shape =
  body pose × logged local pose (`spatial_links.posed`); convex meshes scaled vertex-wise; shapes named `body#index` in
  the logged order.

## 2. Independent path (`research_diagnostics/c1_physics_001/geometry_reference.py`)

Shares no code with the existing calculator: GJK on the Minkowski difference with an exhaustive sub-simplex projection
(witness points returned) for separated shapes; for overlap, the exact minimum translation distance from the facets of the
convex hull of all pairwise vertex differences computed by Qhull (`scipy.spatial.ConvexHull`); rotations from
`scipy.spatial.transform.Rotation`. **EPA is NOT_IMPLEMENTED**; the Qhull facet method is used instead and is not called EPA.

## 3. Known-truth cases (closed form, derived by hand; none taken from `signed_distance`)

| Group | Cases | Truth | Existing max error | Reference max error |
|---|---|---|---|---|
| G1 axis-aligned cubes, gap 0 / 0.5 / 1 / 5 mm along x, 0.5 / 1 mm stacked in z, parallel vertical edges (√(3²+4²) = 5 mm), corner–corner (√3 mm) | 8 | g | 2.6e-18 m | 2.6e-18 m |
| G2 axis-aligned penetration 0.5 / 1 / 3 mm; unequal boxes 2 mm in z | 4 | −p | 2.6e-18 m | 2.6e-18 m |
| G3 vertical edge of a box rotated 30° / 45° about z to a face (gap 2 mm = x_c − h − h(cos θ + sin θ)); both boxes under one general rotation, faces 1.5 mm apart | 3 | g | 9.1e-18 m | 2.3e-17 m |
| G4 perpendicular crossing edges (1 mm = c − 2h√2); the same with the whole scene rotated and moved (0.7 mm); a 45° edge pushed 0.5 / 2 mm into a face | 4 | ±g | 5.2e-18 m | 2.8e-17 m |

**19/19 pass for both paths** at the test tolerance 1e-9 m (justified by float64 rounding of ~1e-17 m per operation on
0.1 m coordinates; six orders below and unrelated to the S6 1 mm tolerance). Further tests, all passing:
sign convention (gap > 0, overlap < 0, touching 0), GJK witness points, quaternion order (w, x, y, z) including a check that
the (x, y, z, w) reading would be detected, local pose × body pose composition and shape index naming, convex-mesh scaling,
and 60 random poses of the asset hand hull against a cube (existing vs reference within 1e-9 m; separated and overlapping).
Log: `logs/pytest_new_tests.log` (49 passed: 44 geometry, 5 loader/mapping).

## 4. The logged C1 shapes (`GEOMETRY_LOGGED_MESH_CHECK.json`)

The collision geometry is identical in all 400 scene records. Boxes (cubes, 8 finger boxes, camera, table) are exact.
**Every logged PhysX convex mesh** (panda_link0–7, panda_hand) is logged as vertices plus a triangulation that is **not the
convex hull of its own vertices**: all vertices are hull vertices and the triangle count equals the hull's, but 36–86
triangles per mesh have a vertex outside their plane (up to 57 mm for a near-degenerate hand triangle) and the mesh volume
is 0.05–0.29% below the hull volume (hand: 93 vertices, 182 triangles, ratio 0.99869).

Effect on the existing calculator, logged hand mesh vs a task cube at 300 random poses: separated — up to 0.064 mm
difference from the convex hull of the same vertices; overlapping — up to 0.21 mm; no sign disagreement. On real C1 rows
(S6 subset): hand–cubeA up to 0.24 mm, hand–cubeB up to 0.22 mm; box pairs ≤ 3e-13 mm. PhysX collides with convex shapes,
so the hull of the logged vertices is the closer model; the existing calculator's hand distances carry this error
(≤ 0.24 mm). Arm-link meshes are not in any S6 pair; they enter other audits (S4 clearances), which were not re-analysed here.

## 5. Scope: verified and not verified

Verified: box–box distance and penetration (axis-aligned and rotated, face/edge/vertex features), sign, posing of logged
shapes (local × body pose, quaternion order, scale, index), the asset hand hull, and agreement of the two paths on 8,394 real
C1 pair-rows for every box kind.

Not verified: penetration depth of non-box convex meshes against closed-form truth (none derived); robustness for
near-degenerate triangles beyond the logged meshes; the PhysX cooked shapes themselves (only what was logged);
continuous-time (between logged rows) collision freedom — the calculator only evaluates the logged poses, and nothing here
establishes collision freedom between them. A correct geometric distance does **not** imply that it should equal a PhysX
point separation (`S6_DIAGNOSTIC_REPORT.md`).

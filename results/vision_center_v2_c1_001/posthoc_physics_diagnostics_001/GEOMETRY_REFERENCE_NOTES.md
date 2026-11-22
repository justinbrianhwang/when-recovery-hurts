# Notes on the reference geometry (`research_diagnostics/c1_physics_001/geometry_reference.py`)

- Shapes are vertex arrays; a shape is the convex hull of its vertices (this is what makes it independent of the logged
  triangulation, and why it differs from the existing calculator on the PhysX convex meshes).
- `gjk_distance(VA, VB)`: support mapping argmax over vertices; simplex of ≤ 4 Minkowski-difference points; the closest
  point of the simplex to the origin by enumerating every non-empty subset, solving the affine projection by least squares
  and keeping subsets with non-negative barycentric weights; stop when ‖v‖² − v·w ≤ 1e-12 ‖v‖² or a support point repeats.
  Returns distance, witness points and an overlap flag (origin reached).
- `penetration_depth(VA, VB)`: Qhull hull of {a_i − b_j}; if the origin is inside (all facet offsets ≤ 0), the minimum
  translation distance is the smallest facet distance from the origin. Exact for polytopes; cost O(|A||B| log).
- `signed_distance_ref`: GJK distance if > 1e-12 m, otherwise minus the Qhull depth.
- `R_wxyz`: scipy Rotation from (w, x, y, z) reordered to scipy's (x, y, z, w).
- Not implemented: EPA, curved shapes (spheres, capsules, cylinders), triangle meshes as non-convex solids, planes.
- Limits: GJK tolerance 1e-12 relative; the Qhull path needs a full-dimensional Minkowski difference (true for every C1 shape).

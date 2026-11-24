# Cover letter (DRAFT — author to complete placeholders)

Dear Editor,

Please consider the enclosed manuscript, "When Recovery Hurts: Separating Rescue from Harm in an Observation-Only
Recovery Intervention for a Frozen Visuomotor Policy", for publication in *Robotics and Autonomous Systems*.

Recovery interventions are increasingly added to frozen learned robot policies, and they are usually evaluated by the
success rate they add. The manuscript shows that this hides a second effect: the same intervention can turn episodes
the policy would have completed into failures. We separate the two effects with paired branching from a common
simulator snapshot, certify the conditional harm rate with an exact finite-sample test on a calibration split, and
confirm the success gain once on a held-out test split under a frozen, hashed protocol. For an observation-only
release-time centring controller on a cube-stacking task, harm was 1 of 348 and 0 of 337 baseline successes on
calibration scenes (certified below 5% for both policies), and success on held-out test scenes rose from 87.5% to
94.75% and from 86.75% to 93.25%.

We think the paper fits the journal's interest in reliable autonomous manipulation and in evaluation methodology. It
also reports the parts that did not work, including a failed contact-measurement check and a deviation from one
pre-declared criterion, and states clearly that the guarantee is statistical and task-level, not a physical safety
guarantee. The study is simulation-only and uses one task.

The manuscript is not under consideration elsewhere. [Preprint status: to confirm.] All data, code and SHA-256
manifests are available [link to add].

Sincerely,
[Author name, affiliation, e-mail]

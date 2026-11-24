# Novelty positioning (what is and is not claimed)

**Claimed (and supported by the data):**
1. The paired rescue/harm decomposition is applied to a runtime recovery intervention on a frozen visuomotor policy,
   with a finite-sample certificate on the harm side (calibration split) and a separate, run-once confirmatory efficacy
   test (held-out split), both pre-registered and frozen in hashed manifests.
2. An observation-only release-time centring controller with confirmed visual memory and explicit abstention rules,
   whose confirmed harm on 1,600 new paired scenes is 1 (calibration 1/685 baseline successes, test 1/697).
3. A mechanism trace from observation to outcome for the rescue and the harm cases, including a commanded-vs-actual
   motion audit.
4. An honest physical-record audit: a failed pre-declared check (S6) reported with a partial, measured explanation.

**Not claimed:** novelty of paired branching, of exact binomial bounds, of Holm, or of risk control (Learn then
Test); physical safety; transfer to other tasks, robots or hosts; that memory alone causes the gain.

**Closest work:** RESCUE (counterfactual value of acting on monitor alarms from the same state). Difference: we add a
calibrated harm certificate and a held-out confirmatory test, and we study one bounded intervention end to end rather
than a monitor-evaluation framework. Learn then Test provides the statistical template; our use is its simplest case.

**Reviewer risk:** "one task, simulation only" is the main weakness and is stated in the abstract, limitations and
conclusion.

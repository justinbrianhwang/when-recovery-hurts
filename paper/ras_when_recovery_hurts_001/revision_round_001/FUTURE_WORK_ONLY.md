# Future work only (not executed; would need new experiments)

1. External baselines (learned recovery, retry/rewind) on the same paired states.
2. Matched-delay (hold) arm on confirmatory-size scenes to bound the re-planning noise floor.
3. A rule-level ablation with enough acting scenes to estimate harm per rule.
4. A feasibility/value gate (remaining offset vs executed rate and remaining horizon), then a new calibration.
5. Controller-internal logging (IK/drive targets, sub-steps) on the development host to explain Check 4 first steps
   and the S6 residual.
6. Other tasks and a real robot.

None of these was run in revision round 001; the manuscript states them as limitations or untested ideas.

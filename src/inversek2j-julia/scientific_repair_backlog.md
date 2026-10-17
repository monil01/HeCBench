# Inverse kinematics scientific repair backlog

The measured study reproduces the documented frozen-pose native defect. It does not solve inverse kinematics and is excluded from scientific-fidelity/performance recommendations. Original CUDA and genuine historical CCD Julia sources remain intact.

A corrected secondary study must implement actual forward position updates, specify radian relative-joint semantics, bounded step policy and termination/solution error targets, and verify forward-kinematics endpoint error independently. Preserve initial/final error and iteration history; distinguish reachable, unreachable and zero-distance targets. Historical CCD contracts error on nonsingular witnesses but fixed25steps is not a general solution guarantee, and target(2,0) stalls. Do not call wholesale native algorithm replacement a simple repair or untouched-reference verification. No corrected fullCCD GPU experiment has been run.

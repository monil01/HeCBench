IRIS DMem port preserving all18 original tensor contraction cases in float
and double precision. Native tensor and adjacency inputs remain all ones.

`make run` retains dimension64 and5 repeats. Validation uses64 with one repeat
on native HIP and IRIS, comparing original checksum/min/max output. IRIS also
checks every output element against its analytic case result: N^3 for cases1-5,
N^2 for cases6-15, and N for cases16-18, rejecting nonfinite values and errors
above1e-6 relative tolerance. No kernel arithmetic or native sibling is changed.

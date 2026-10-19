# HBC observations

RQ1: Full exact/default Brandes on the recovered original gene-coexpression graph (7,819 vertices,195,928 edges), plus path/grid/star controls, passes all twenty checked trials. Independent LongDouble replay and positive FP32 componentwise error budgets were fixed before GPU data. Exact path/star closed forms, complete final BC, actual input readbacks, last-source reachable workspaces and actual guards are checked. No native mathematical correction. Biology and optional approximation are unverified.

RQ2: One mathematical body/model; CUDA.jl and JACC each need two program/compiler-control checks because default90/98 registers reject original1,024-thread blocks. Explicit64-register caps resolve this without geometry change. KA first check succeeds. Shared adapter typedef B2 and disconnected-workspace checker T1 are distinct from native mathematical failures. Token windows include inherited/repeated context, verification waits and coordination, with nested verification nonadditive.

RQ3: Means2–5 for sum of four original kernel ROIs: native0.4988771165s; CUDA.jl41.0195099785s; KA38.25169678s; JACC33.261700257s. Retain this large slowdown. These are aggregate four-case timings, not isolated original-graph or allocation-inclusive CLI times.

RQ4: Irregular BFS/CSR, UInt64 path counts, FP32 dependency/BC atomics, workgroup barriers, dynamic source scheduling and adaptive traversal. CUDA.jl/KA authored kernels use their own framework/Atomix APIs; JACC uses explicit CUDA static scratch and pinned indexing-wrapper compiler control, classified hybrid. No register/spill/roofline counter evidence establishes why performance differs.

RQ5: CUDA-specific full operator expression and verified original-data agreement are feasible. Compiler launch-resource control and adapter boundaries matter; no general backend portability, biological discovery or causal framework-ranking claim.

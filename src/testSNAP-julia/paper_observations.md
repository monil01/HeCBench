# TestSNAP observations (RQ1–5)

**RQ1.** Complete original J14/2000-local/26-neighbor/100-step force workload, plus J2/J4 controls, passes all four implementations. Each model makes 85,416,090 pointwise pair-gradient/local+ghost-force comparisons over five trials, retains exact actual input readbacks and 4,341,760 actual guard bytes. An independent SU(2) polynomial/dual-energy-gradient oracle and small-case energy differences support the source operator. CG coefficients remain declared shared inputs, not independently rederived tables. The supplied J4 all-zero force field failed an auxiliary check; the explicit revised J4 criterion and original rejection are preserved. No native kernel correction, full MD trajectory or material accuracy claim.

**RQ2.** Each target's first complete seven-kernel source bundle and first GPU program passed. Kernel definitions, complete-program checks, frontend evaluations and unknown backend invocations are separate units. Actual context-inclusive translation activity tokens: CUDA.jl 2,554,578, KA 2,656,906, JACC 3,275,036. Native preparation 8,571,110 is separate. Nested verification overlaps; do not sum. Shared checker-development feedback includes a caught floating-metric overload and J4 auxiliary-groundtruth revision. Known missing original metric-candidate bytes are explicitly disclosed.

**RQ3.** Last-four means (s): CUDA 3.798497667; CUDA.jl 6.479696718; KA 6.444109561; JACC 6.733073813. The ROI includes 742 kernel calls plus repeated host preparation, transfer, force assembly and source error accumulation. One block cannot establish a statistical ranking.

**RQ4.** All three use genuine framework device entries and public FP64 atomics, source 256/16×16 geometry and zero shared allocation. Native C++ versus Julia host execution, stream lowering, extra H2D context completion and observer gaps confound abstraction-only interpretation. No profiler or causal roofline claim.

**RQ5.** All targets express this complete scoped scientific pipeline on the recorded V100/CUDA backend. The study supports no cross-backend capability, trajectory stability or trained-potential quality claim.

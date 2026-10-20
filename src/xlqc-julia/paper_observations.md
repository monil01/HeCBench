# XLQC observations

RQ1: full original H2O/STO-3G SP/DP, nine converged SCF iterations;40 complete7029-word artifacts and11360 raw global state files pass. Alternative tight-Boys B-array ERIs check every primitive J/K. Every iteration checks metric/eigen/projector equations; fresh unscreened physical F(Dnew) checks final HF energy/residual. Source GPU and CPU controller unchanged. Raw CPU MO coefficients retained with checks, phase identity unclaimed. This is finite-basis model validation, not experimental chemistry or an independent full SCF implementation.

RQ2: CUDAjl4 program submissions/3 actual translated JIT processes/1 math bundle; two host/evidence failures. KA3 submissions/3 JIT/2 index-mapping candidates/1 formula bundle; physical-index mismatch caused one GPU fault, repaired with logical Cartesian IDs. JACC2 submissions/2 JIT/1 candidate. Counts separate C++ adapter builds, math helper definitions and internal compiler calls unknown. Actual tokens in effort_summary preserve inherited/cached context and ordered reuse, not intrinsic abstraction cost.

RQ3: SP means native/CUDAjl/KA/JACC ms: 6.111990, 6.929433, 7.336615, 7.734970. DP ms: 8.495948, 12.589826, 12.801667, 13.357761. Five checked trials, mean2–5. These are sums of original J/K dispatch+context stage ROIs, not full SCF or pure GPU events. Extra validation pauses between stages can affect clocks/caches and are common, declared instrumentation.

RQ4: SP retains Float64 global arrays/shared accumulation but Float32 Rys private math; DP uses Float64 throughout. KA flattens physical groups and requires explicit logical IDs. Reversing only index substitutions recovers the frozen CUDAjl math byte-identically. Source actual Q is raw ERI despite sqrt comment; preserve source screening and independently test unscreened physical state. No general Schwarz-bound, hardware-resource, bottleneck or roofline claim.

RQ5: feasible genuine Julia GPU math through all target dispatches on the studied CUDA backend; native CPU/GSL host controller and CUDA shared/barrier features are explicit hybrid requirements. Higher-root/basis variants remain unverified.

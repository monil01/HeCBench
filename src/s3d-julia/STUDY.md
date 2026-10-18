# s3d: complete pointwise chemical-rate pipeline

## Scientific scope

The source evaluates a **27-stage chemical source operator** independently at every grid point: forward Arrhenius rates, thermodynamic factors, concentrations, reverse rates, pressure falloff/mass action, ten-unknown quasisteady algebraic elimination, and22 production rates. It is **not a spatial PDE solve, ODE integrator, or independent validation of a chemical mechanism**. Species meanings are not invented from approximate molecular weights. Source stage split/thread geometry and original scalar coefficients, casts, expressions and ordering are preserved. Original comments describe some splits as register-pressure management; actual registers/spills are unprofiled.

Both source class1 cases are measured: **Float32 N=24³=13824 and Float64 N=16³=4096**, each100 complete pipeline calls per outer trial. Their unequal sizes prohibit interpreting crossprecision timings as an equal-N precision comparison. All four implementations use the same inputs within each case: uniformT1000/P1.0132e6, mass-fraction entries3=.218/14=.064/21=.718 (zero-based source indices), molecular-weight output multipliers all1, conversion factors1. This synthetic operator fixture is not a calibrated application dataset. The strictTEMP>1000 high-temperature branch is translated but **not numerically exercised**, nor are heterogeneous spatial fields, invalid inputs, other size classes or a reacting-flow trajectory certified. Homogeneous output evidence alone cannot detect every possible grid-point permutation; source index review supplements it.

Historical Julia performed unrelated sin/cos work and printed canned chemistry rows. It is archived. Genuine CUDA.jl restoration authors shared pure Julia lane helpers, through a limited auditable C++ literal/index/flow translator without algebra simplification. CUDA.jl uses ownblock/thread indices, KA own global indices, and JACC its own flat-index argument. Every version keeps all27 stages onGPU; CPU uniformity is only an oracle optimization, never a GPU shortcut. CUDA host allocation/completion glue is tested onNVIDIA only; no foreign CUDA device intrinsic or vendor operator library is called from KA/JACC.

## Numerical evidence and limitations

Every one of40 final captures contains complete **WDOT22/C22/RF206/RB206/RKLOW21/EG31 arrays**,508values per grid point, plus exact input/parameter prefix. UnwrittenEG32 and unused sparseA storage are excluded. Float64 serialization preserves actual Float32 or Float64 numerical values. Counts: prefix24N+29, payload508N, completeartifact532N+29. All five dual-reference comparisons pass for each portable version/precision.

The C++ CPU evaluator retains the original scalar formulas and uses host math instead of GPU lowering. This is a **source-formula computational cross-check, not independently derived chemistry**. Separate algebraic checks establish nonnegative concentrations and sumC=P/(T*8.314510e7), source spatial homogeneity, and source-defined zeroWDOT22. No elemental/mass conservation, equilibrium, stiffness, or integration-convergence claim is made. `saved_output_audit.json` reports actual errors separately by field. A global absolute maximum across differently scaled concentrations/rates/thermodynamic factors is not one physical accuracy measure.

Tolerances were declared before actual numerical results: Float32 relative2e-4 plus eight smallest subnormal ULPs (1.12103877146e-44), Float64 relative1e-10 plus1e-55. All fields must be finite. Original **FLT_MIN is a Float32 floor even in Double**, and DIV=x*(1.0f/y) is preserved. No fastmath/FTZ option is requested. Four Float32 production-rate species retain actual nonzero subnormal values in every model; instruction-level FTZ or math-lowering causes are not inferred. Source fmax/fmin versus Julia NaN/invalid-input policy is outside this finite positive fixture.

Against the source-formula CPU evaluator, Float32 production-rate maximum absolute error is1.91e-17 native and3.47e-18 CUDA.jl; maximum relative error is5.48e-6 and3.16e-7. Float64 production-rate errors are about8.08e-27 native and1.62e-27 CUDA.jl. These operator errors do not establish physical mechanism accuracy. Very large global absolute differences belong to large intermediate thermodynamic values and must be interpreted with their field scale/relative error.

## Timing and resource contract

One complete untimed pipeline warms compilation, then five in-process outer trials each execute100 complete pipelines. Whole ordered GPU dispatch work plus **device-wide completion** is timed and divided by100; the first outer time is excluded, last four averaged. Allocation, inputs, output transfer/serialization and checks are outside ROI. Recomputed intermediates require no input reset. Only final output of each100-call outer trial is captured; intermediate99 outputs are not individually audited. Four retained sample means are not400 independent observations. All samples and variability are retained; no fastest selection.

Stage group sizes remain64 forratx/ratxb/qssab and128 otherwise. There is no shared scratch, workgroup barrier, atomic update or crosspoint coupling; global stage arrays are preallocated outside ROI. JACC explicitly uses separate64/128 LaunchSpecs with matching block counts, **0shared bytes** andsync=false; device completion is imposed by the common timing harness. No counter-based bottleneck/occupancy/roofline claim is made. Transcendental-heavy algebra and global staging/launch boundaries are meaningful ontology traits, but ordinary FLOP roofline counts alone cannot attribute the timings.

| Source case | Native CUDA (us) | CUDA.jl (us) | KA (us) | JACC (us) |
|---|---:|---:|---:|---:|
| f32 N=13824 | 734.79 | 788.65 | 785.71 | 778.72 |
| f64 N=4096 | 761.35 | 779.01 | 784.78 | 786.90 |

## Attempt and usage accounting

Reference preparation has **two program candidates and three process checks**: initial frontend parse rejection (invalid function-call whitespace, deviceJITnotreached), then successful Float32 and Float64 GPU checks. One shared semantic restoration replaces the historical wrong computation; a native harness initializer-list compiler error and a Python source-generator syntax error are preserved separately. Native kernels remain unchanged. CPU evaluator built once; native adapter builds total3 (onefailed/twopassed). Historical attempts/tokens remain unknown.

KA and JACC each succeed with **one candidate, two actual GPU process checks, zero repairs,27 authored generic kernel definitions**. Helpers, definitions, precision cases, GPU stage launches,100inner repeats and compiler repair rounds are distinct. Internal compiler passes, cache hits and specialization counts are unknown. Shared formula/fixture/reference repair cost is separate from marginal wrappers; JACC sees successful prior reference/KA work. Consumption windows include context/cache/reasoning/validation/coordination and overlap broad/child phases; they are not additive exclusive billed costs. Source chars/4 proxies remain separately partial. Queue handoffs during reference/KA phases are disclosed.

Restart using `study_config_f32.json` and `study_config_f64.json` with the central verifier, or `make run-all` in a child. Saved-artifact audits, source/dependency hashes, fixture generators, failed-source archives, actual attempts, usage, per-model ontology and paper observation are checkpointed. Root coordinates separate variant commits.

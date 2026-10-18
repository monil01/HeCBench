# eikonal: scientific and implementation observation

This restoration recovers an iterative scientific algorithm rather than a superficially matching output shape. The independent checker solves the discrete Godunov system with a global monotone Float64 iteration, contrasting with the native blocked mixedprecision FIM and CPU active-list scheduler. Native source Float32 square roots produce measurable submicro field differences from the Float64 witness despite double storage. Fullstate four-way checks and seed/axis/residual observations separate reproducibility from continuous-PDE accuracy; neither synthetic convergence messages nor a checksum can do this.

## Reference audit

Historical Julia only fills indices, emits fabricated iteration/convergence timings and writes one checksum rather than full NRRD field; it does not solve the native Eikonal equation. Restored three source-equivalent authored kernels plus actual host active-list FIM algorithm. Native original kernels/precision policy unchanged; independent global Godunov oracle is a checker, not substituted GPU algorithm.

## Numerical and performance evidence

Configuration `eikonal_full_block4_FIM_constant_variable_speed_3fixtures_v1`, result `/home/imo/HeCBench_julia/.porting-state/portable_julia/runs/20261018T053826730753Z_eikonal/result.json`. Five in-process trials per model; full outputs checked in every trial against both references; mean trials2–5. Warmup precedes trial1. Native kernel unchanged. This named configuration does not certify all CLI modes or physical model validity.

## Cost and failures

Reference two wholeprogram checks: own kernels compile/launch, then shared BitVector D2H scalarfallback R1; host VectorBool repair then pass. Authored device-source candidate1, two wholeprogram JIT processes, no device compiler-feedback repair. KA firstcandidatepass; JACC phase follows. Referencecost includes full hostalgorithm/nativeadapter/independentoracle; original historical effort unknown. Historical CUDA.jl attempts unknown. Reference correction or documented P1 mapping phase is separate from target translation. Prospective harnessactivitycounters include cached/repeatedcontext and coordination, nonexclusive; historicalusageunknown, sourceproxiespartial.

Reproduce: `python3 tools/verify_portable_julia.py --config src/eikonal-julia/study_config_ka.json`, then JACC config reusing saved references. Do not run a new series merely to refresh metadata.

## Full scientific algorithm and evidence

All20captures16383Float64words:4617exactinputprefix+11766fullstates/guards/counts. Full dsol independentlycheckedagainst globalFloat64GodunovJacobi field, allfinite/nonnegative/originseed0, bothconstant-speedfixtures coordinateaxesexact. All3Julia capturesbitidentical, native differs about1.04315e-7 withinunchangedpredeclared1e-5+1e-5relative. Native sqrtf despite Float64storage/math retained; independent field error native1.432214e-7 vsJulia1.172883e-7. LocalGodunovresidual maxima recorded descriptively, not changedacceptance. Full tsol/con/listVol/list/counts directlycomparedall5, all168guards andreadonlyspeed/mask exact. All3nativekernels andCPUactive-listneighbor/terminationalgorithm retained. Staticshared1728Bsolver/neighbor and64Breduction; JACCpublicshared templates addpreloadwork. FullsolverROI includesCPUlistinitialization/scheduling/H2D/D2H, notGPUkernel-onlythroughput; sourceoriginalhosttempallocation startsbeforetimer whereas namedtemporarylist initializationinsidehelper all4. Original512cubeCLI/continuousPDEconvergence/obstaclecasesunmeasured. Meansseconds {"cuda": 0.0019309952499999999, "cuda_jl": 0.0028942110000000003, "ka": 0.00295411675, "jacc": 0.0046104965}. Oneauthoreddevicecandidate eachmodel; CUDAjl firstprocessfailsafterkernellaunch atsharedpackedBoolD2H, repairedhostVectorBool thenpass. Nativefirstvalid5 reused, no favorabletimingselection.

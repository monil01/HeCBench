# stencil3d: scientific and implementation observation

Source-levelscientificauditmatters: a7pointscalarLaplacianisnotthenative9componentanisotropicfacefluxoperator. Genuineportsretainrolling4planeFP64sharedtilesandcrossderivativefacegradients. Independentdirectsixfacefluxoraclesavoidreusingrollingstorage, andconstantpotentialtestsnullresponsewithvariableSPDconductivity. Full YZ tiles define the measured scope; partial YZ tiles remain untested, while the partial X slab is covered. ThespatialoperatorstudydoesnotestablishaPDEsolverconvergence/stabilityorparticularcardiacapplication.

## Reference audit

HistoricalCUDA.jl replacesnative9componentanisotropicfacefluxdivergencewith7pointscalarLaplace*firstsigma, discarding8tensorcomponents andallrolling/sharedmapping. GenuineS1operatorrepair, notmerelymappingdifference. Nativeoriginalhasnoaccuracychecker.

## Numerical and performance evidence

Configuration `stencil3d_n30_spd9tensor_nonuniform_constant_rolling_v1`, result `/home/imo/HeCBench_julia/.porting-state/portable_julia/runs/20261017T192115623702Z_stencil3d/result.json`. Five in-process trials per model; full outputs checked in every trial against both references; mean trials2–5. Warmup precedes trial1. Native kernel unchanged. This named configuration does not certify all CLI modes or physical model validity.

## Cost and failures

One saved candidate and one compiler/JIT process per target model; no observed target repair. Historical CUDA.jl attempts unknown. Reference correction or documented P1 mapping phase is separate from target translation. Observed prospective usage intervals are recorded separately; exclusive billed translation cost is unknown. Final-source proxies remain partial.

Reproduce: `python3 tools/verify_portable_julia.py --config src/stencil3d-julia/study_config_ka.json`, then JACC config reusing saved references. Do not run a new series merely to refresh metadata.


Independent tracked scientific_operator_witness.py uses generic face-normal/tangential gradients and conductivityrows, notnative rollingbuffers. Every54000value inall20artifacts matchesexactly. Pointwisetensors are symmetricstrictdiagonal-dominantwithGershgorinlowerbounds[1.625,2.5,3.375] beforepositivespatialscaling, establishingpointwiseSPD, notPDEsolverstability. Constantpotentialgiveszeroresponse. Historical7pointscalarformula disagreesat 21952interiorcells onthisfixture, maxerror 9.74462890625; thisisaCPUsource-formulawitness, notanoldGPUrun. KAusesdocumentedGPU-onlycpu=falsecodegenforreturn; nosupportfailureclaimed. JACCadds1024FP64shared-templatecopy/initialbarrier. Nativeoriginalhasnonumericalverifier;1e-10wasdeclared, notinherited.


Prospective rawusage intervals include cached/repeatedcontext and coordination; the parentworkflow contains its childphases and must not be summed with them. Exclusive billed translation cost remains unknown; source-size proxies remain separately labeled.

## Source scope correction

Partial YZ tiles were not tested. Early returns alone do not establish a barrier defect: on Volta, barriers wait for all non-exited threads. Source-specific halo and data dependencies remain unmeasured. The N30 full-YZ results are unchanged. See the [CUDA 12.6 programming guide, Volta migration](https://docs.nvidia.com/cuda/archive/12.6.0/cuda-c-programming-guide/index.html#independent-thread-scheduling-7-x). Archived configs retain the original wording; the source_scope_adjudication supplement supersedes that interpretation.

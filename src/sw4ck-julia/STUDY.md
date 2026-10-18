# sw4ck: scientific and implementation observation

All five original FP64 elastic kernels: upper six-layer one-sided boundary contraction, three distinct interior displacement-component kernels with direct/mixed derivatives, lower six-layer boundary contraction; original no_gp coefficient pointer offsets and operation initial-L preserved. Source3D blocks(16,4,3),(64,2,2)×3,(16,4,4), same paddedglobalranges; one outputcell/component per thread, directglobal coefficient/field loads, no sharedmemory/atomics/barriers. C component-major address macros expanded exactly to1basedarrays and lexical C locals renamed. Two gridshapes and3fieldtypes (quadraticCartesian,constantCartesian,varieddyadiccurvilinear); complete88686 outputvalues and389724inputprefixvalues. Analyticinterior quadratic L=(64,73,82),constantL=0; boundaryonlysourcefidelity,nophysicalclosureclaim. Parent preservesallfailedcandidatebytes andsourceGPLlicense.

## Reference audit

HistoricalJulia gpu_timed_work performs unrelatedFloat64adds then reports EXACT_NORM+0*sample; no nativeelasticoperator. Originalbytespreserved. Newsource-faithful5kernelreference required; candidate1 sharedCPU+GPU generatedcode lostC multiline += continuation, independentlymanufacturedNavier checkcaughtthirdcomponent initial−42 versuscorrectinitial−82. WholeRHSparenthesization restores allsource arithmetic.

## Numerical and performance evidence

Configuration `sw4ck_fullfivekernels_bothboundaries_two_shapes_three_fields_v1`, result `/home/imo/HeCBench_julia/.porting-state/portable_julia/runs/20261018T020934934473Z_sw4ck/result.json`. Five in-process trials per model; full outputs checked in every trial against both references; mean trials2–5. Warmup precedes trial1. Native kernel unchanged. This named configuration does not certify all CLI modes or physical model validity.

## Cost and failures

OriginalS1norminjectionreference rejectedbyinspection. Sharednew5kernelcandidate1 CPUandGPU agreeexactly butindependentPDEcheckfailedN1; source-continuationdefectS1 repairedonce withallcandidate1CPU/GPU/generatedscripts preserved. SharedPythontranslatordeclarationparserB2 initiallyfailedbefore GPUdraftcompletion. Two newCUDAjlprogramchecks, five authoredkerneldefs each; KA/JACC onecandidate/checkeach. Individualcompilerinvocationcounts unknown ratherthan assumed5perprocess. Historical CUDA.jl attempts unknown. Reference correction or documented P1 mapping phase is separate from target translation. Observed prospective usage intervals are recorded separately; exclusive billed translation cost is unknown. Final-source proxies remain partial.

Reproduce: `python3 tools/verify_portable_julia.py --config src/sw4ck-julia/study_config_ka.json`, then JACC config reusing saved references. Do not run a new series merely to refresh metadata.

## Independent scientific witness and shared-helper failure

For homogeneous Cartesian metrics with mu=2 and lambda=3, Navier elasticity gives L(u)=mu*Laplacian(u)+(lambda+mu)*grad(div(u)). The manufactured quadratic has Laplacian(u)=(2,4,6) and div(u)=12x+13y+14z, hence L=(64,73,82). The constant field has L=0. All20 saved captures independently check8748 interior values (maximum error1.1368683772161603e-13) and54180 untouched halo values with exact bytes. All88686 outputs are compared against both references on every trial;389724 input-prefix values match exactly. Boundary values additionally have full scalar source-expression agreement, which does not independently establish physical SBP boundary accuracy for the synthetic q/1000 coefficients.

The first new reference had a common-mode shared-helper error: C multiline += expressions with a later leading+ were not parenthesized as a whole in Julia. Its generated CPU and GPU agreed (reported error0), but independent elasticity rejected component3 at(0,0,7): expectedinitial−82, actualinitial−42. Native CUDA passed. The immutable failed result/capture and archived candidate sources remain linked in candidate1_failure_evidence.json. This one semantic repair is recorded separately from target framework translations; neither KA nor JACC required a repair. The historical norm-injection stub is a separate original-reference S1 finding.

All five kernels retain the original source arithmetic and geometric launches. Derived sources retain the original GPL-2.0-only license/attribution; source_translation_manifest.json records exact original hashes and transformation rules. Original source initialization interleaves components while kernel addressing is component-major; named fixtures declare their layout explicitly and do not claim the supplied large-dataset expected norms.


Prospective rawusage intervals include cached/repeatedcontext and coordination; the parentworkflow contains its childphases and must not be summed with them. Exclusive billed translation cost remains unknown; source-size proxies remain separately labeled.

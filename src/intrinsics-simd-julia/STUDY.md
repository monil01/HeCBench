# intrinsics-simd: scientific and implementation observation

The historicalchecksum could agree while the algorithm was unrelated. Fullarray comparison removes the fabricatedacceptance shortcut, and individualnative/portable/arbitraryprecision results for all118 operations prevent XORcancellation from hiding coincident errors. Direct API absence does not imply semanticunsupported: ordinary integerbit-lane code expresses the required finite contract. Its lowering differs from nativeintrinsics, so performance reflects this explicit mappingchoice and frameworklaunch behavior, not an isolated intrinsic abstractionpenalty. SourceReLU andDPX names do not change this syntheticprimitive into an ML or sequencealignment application.

## Reference audit

HistoricalJulia explicitly substitutes unrelated integer stress operations for118 native SIMD/DPX operations and hard-codes expected checksum forN1024/repeat1. This is operator and validation substitution, not a faithful native port. Preservedoriginal; restored all118 finite functional semantics and source XOR order using portable lane arithmetic, with separate mappingadaptation vs nativeintrinsics. Native source unchanged.

## Numerical and performance evidence

Configuration `simd_all118_source33_1024_4097_individual256_v1`, result `/home/imo/HeCBench_julia/.porting-state/portable_julia/runs/20261018T071033293050Z_intrinsics-simd/result.json`. Five in-process trials per model; full outputs checked in every trial against both references; mean trials2–5. Warmup precedes trial1. Native kernel unchanged. This named configuration does not certify all CLI modes or physical model validity.

## Cost and failures

Oneauthoredbenchmarkkernel plusoneauxiliaryvalidationkernel permodel;118devicefunctions are operationdefinitions, not118kernelcandidates. FirstCUDAjl candidate/process passed all118 individualnative/CPUchecks; one sharedPythonoperationgenerator syntaxB2 repairedbeforeanyGPUcodegeneration/JIT, separatelyattributed. Shared auditedoperationbodies reusedbyKA thenJACC, marginaltargettranslationnotindependent118operatorinvention. Activitywindows include cached/repeatedcontext, source/docsreview, waiting/coordination andsamebenchmarkmetadata; noexclusivebilledcostclaim, workflowoverlapsphases donotsum. Historical CUDA.jl attempts unknown. Reference correction or documented P1 mapping phase is separate from target translation. Prospective activity counters retain cached/repeated context, docsreview/coordination/waiting/metadata scope, not exclusive billedcost; historical consumption unknown and sharedcode omitted by main-sourceproxy.

Reproduce: `python3 tools/verify_portable_julia.py --config src/intrinsics-simd-julia/study_config_ka.json`, then JACC config reusing saved references. Do not run a new series merely to refresh metadata.

## Full contract versus native instruction mapping

Every20capture226990UInt32words exactly matches independentlyconstructed full expected bytes:95786prefix, actual sourceinputs/triples,5154nativebenchmarkXORoutputs,30208individualoperationvalues and64canaries. All4models bitidentical, no XORchecksum shortcut. Source118operations fullyrestored; boolpredicate pointeroutputs dead/unusednotclaimed. All3 own1benchmarkkernel and1auxiliaryvalidationkernel,118portable devicefunctions are not118kernelcandidates. Nativeunchanged intrinsiccalls vs targetordinaryprimitive integerlanes: a disclosed source-to-modelmappingadaptation, no directISAequivalence/universalunsupported/frameworkpenaltyclaim.

SourceN33/1024/4097 signed32additionrange defined; individualoperationprobe domain avoids signed32overflow ambiguity byexplicitb0fixturepolicyonlyforthoseoperations. Packed8/16overflow/saturationboundaries remain exercised. This finitecontract is not all32bitoperandproof. Auxiliaryprobesareuntimed, source100launches/finalcontextcompletion retained. Mean2-5 sum3componentseconds {"cuda": 1.0563455e-05, "cuda_jl": 1.416085e-05, "ka": 1.49903125e-05, "jacc": 2.132767e-05}. Public math semantics reviewed against [CUDA12.0 SIMD API](https://docs.nvidia.com/cuda/archive/12.0.1/cuda-math-api/group__CUDA__MATH__INTRINSIC__SIMD.html), signedaveraging policy against [PTX ISA](https://docs.nvidia.com/cuda/archive/12.9.0/parallel-thread-execution/index.html); pinnedlocalheaders hashes retained. No physicalapplication accuracy or MLtraining claim is implied by ReLU/DPX operation names.

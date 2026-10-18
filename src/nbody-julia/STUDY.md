# nbody: scientific and implementation observation

Fullactual15timedpoststepstates+particleenergyarrays/KEscalar and4initialforce-replays, 21741Float64encodingwords/prefix2951, full18290oraclevalues withpredeclaredconditionalFP32forwardcomponentbudgets. Ballistic/symmetric2bodycontrols andstrictTinyGforce/energy preventglobalKE-onlyfalseacceptance. No nativealgorithmrepair. All3framework_device_kernel/noCUDAdeviceextensions; standardJuliaFastMathisolatedinvsqrtproducesactualnativesameSFU butcompilersdifferentunrolling/resourcepolicies. NamedfiniteODEsteps notastrophysical/continuum/totalenergyconservationproof.

## Reference audit

HistoricalJulia genuinelyimplementssoftenedforce/Eulerupdate, butSoA/nativeAoS, inv(sqrt)insteadsourceapproxrsqrtf andCPUenergyreductionoutsidemeasured2kernelROIvsnative3rdserialGPUreduction. Align40BAoS/source256ceilgrid/standard@fastmathinvsqrt-only(actualrsqrt.approx.f32PTX)/one-threadserialFloat32sum, sourceGPUmathunchanged. DefaultJuliaRNG/drawlayout differsCPP, exactcommonfixturesincludeCPPnativeMT42for33/257 plusanalytic1/2. Thesearemapping/measurement/inputalignment, notwrongscientificalgorithmS1.

## Numerical and performance evidence

Configuration `nbody_AoS40_threeKernel_allsteps_forceReplay_forwardFP32_N1_2_33_257_v1`, result `/home/imo/HeCBench_julia/.porting-state/portable_julia/runs/20261018T095739116223Z_nbody/result.json`. Five in-process trials per model; full outputs checked in every trial against both references; mean trials2–5. Warmup precedes trial1. Native kernel unchanged. This named configuration does not certify all CLI modes or physical model validity.

## Cost and failures

CUDAjl/JACCfirstcompletecandidate/processpass. KA2savedwrappercandidates/2wholeprogram+partial/fullJITprocesses; firstHOSTconstructornameaccumulate!resolvedBaseaccumulate!namespaceafter2kernelwarmups, noGPUcompiler/numericalfailure ortimedtrial. Renamedspecificenergy_serial_ka!resolvesAPIbinding; mathematicalalgorithmcandidate1/helper+contractunchanged. CUDAjlcurrentreferenceworkincludesCPPinitializerbuild/run andnative+CUDAjlcompile-onlySFUinspections separately (noGPUkernels/retiming), sourceforwardmodel/officialaccuracyresearch. Prospectivewindowscachedcontext/math/source/metadata/waiting/coordination nonexclusiveactivitynotbilledcoding. Historical CUDA.jl attempts unknown. Reference correction or documented P1 mapping phase is separate from target translation. Prospectiveactivitywindowsavailable/cachedcontext/source/physics/forwardbudget/SFUresearch/codegen/metadata/waiting/coordinationincluded, notexclusivebilledcoding. Originalsessionunknown/source-sizeproxyonlypartial.

Reproduce: `python3 tools/verify_portable_julia.py --config src/nbody-julia/study_config_ka.json`, then JACC config reusing saved references. Do not run a new series merely to refresh metadata.

## Field-level accuracy and tiny-signal controls

All20full21741Float64encodedcaptures/prefix2951/full18290oraclevalues/500guards passconditionalcomponentwiseFP32forwardmodel. Float64hostencoding actualFloat32(notGPUFP64); everyactualtimedpoststepfullparticle/energy+KE, thenexplicituntimedinitialforcereplayinSeparatebuffer. Nativealgorithmunchanged. Maxabsoluteoracleerror1.2775232172e-7/maxbudgetratio.4883406367; Julia3bitidentical, native-Julia maxabs3.7252902985e-9. StrictN2force 5.306203169923682e-10 vsbudget 2.4669403388731037e-15 andkineticenergy 9.009853736178057e-20 vsbudget 1.2692962793264796e-24 rejectzero-force/zeroenergy evenwhereoriginalKEabs1e-3criterionpasses. This analyticalnegativewitnessisnot a failedGPUcandidateornativedefect.

The [NVIDIA CUDAaccuracytable](https://docs.nvidia.com/cuda/archive/12.9.0/pdf/CUDA_C_Programming_Guide.pdf) lists2ULPforrsqrtf; a conservativenormal-rangeSFUmarginentersoursource-inputFP32forwardmodel. It isconditionalengineering, notformalcompiler/library/continuumaccuracyproof. InitialCppMT42fixturebitsource-equivalentfor33/257; analyticballistic/pairedcontrols1/2. No totalenergyconservation/astrophysicalpredictionclaim.

Sourceperstepcore3kernels+contexttimerpreservedall4, sum15physicalsteps/45calls; fullpoststepD2Hcapturesoutsideintervals canaffectunmeasuredintersteptraffic/cacheversusoriginalscalarreadcaller. NotwholeCLI TotalTime. Same40BAoS/256ceilgrid/1-threadserialsum andactualrsqrt.approx.f32 native+CUDAjlPTX; onlyinvsqrtstandardJuliaFastMath, no backenddeviceextensionsinKA/JACC. NativePTXloop160B/4particles vsinspectedCUDAjl40B/1; KA/JACCunrollingunmeasured, commonmathbodynotISAproof. No physicalregister/rooflinebottleneckclaim. Meantrials2–5 {"cuda": 0.00043594624999999997, "cuda_jl": 0.0006822830000000001, "ka": 0.0007906907500000001, "jacc": 0.00089725925} seconds; allvalidoutliersretained/notfastestseriesselection.

CUDAjl/JACCfirstcompletecandidate/processPASS. KA2wrappercandidates/2wholeprogram/partialthenfullJITprocesses,1mathematicalalgorithm: hostBaseaccumulate!constructorbindingcollision B3 afterforce/updatewarmup/beforethirdJIT, renamedserialentryresolves, no math/tolerance/timingchange. SeparateCPPfixturebuild/run andnative/CUDAjlcodegen-onlySFUinspections includedreferencecost, not extraGPUtimingseries.

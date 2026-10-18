# ace: scientific and implementation observation

A coupled phase/temperature update checks final andpreviousfields plusall3forcefields, independentlyrecomposeddiscreteequations andconstantnegativephaseequilibrium. Nativebodytextverbatim. Matchingdiscreteoperators andshorttimehorizons doesnotvalidatefreeenergyphysics,materialgrowth,400cubedscaleorlongtimestepping.

## Reference audit

HistoricalCUDAjl genuine sourceAllenCahn/thermal equations and3Dmapping, notstub. Preserveoriginalbytes. Study aligns its1Dswap to unchangednative3D8x8x4geometry, adds preallocatedfixture/full7fieldoracle/exactguards/warmup/fivetimingtrials; original400cubedCLI/offloadregion unmeasured. Originalnative6devicekerneldefinitions retainedverbatim under named16cubed/24cubedcompiletime dimensionnamespaceincludes, no mathematicalnativecorrection.

## Numerical and performance evidence

Configuration `ace_fourfixtures_sixkernels_sevenlaunches_16_24_steps1_3_v1`, result `/home/imo/HeCBench_julia/.porting-state/portable_julia/runs/20261018T032544970239Z_ace/result.json`. Five in-process trials per model; full outputs checked in every trial against both references; mean trials2–5. Warmup precedes trial1. Native kernel unchanged. This named configuration does not certify all CLI modes or physical model validity.

## Cost and failures

Referencewholeprogram2candidates/2checks, firstsharedhostsyntaxB2beforedeviceJIT thenpass; oneauthoreddevicebodycandidate. KAfirstcandidate/checkpass, thenJACCfirstcandidate/checkpass. Sharedreference/math/fixtures reused intentionally, nonewindependentcoldstartcomparison. Historical CUDA.jl attempts unknown. Reference correction or documented P1 mapping phase is separate from target translation. Prospective perphase activity windows retain actual harness counters, repeated/cached context and marker lag; not exclusive billed or cold-startcost. Historical consumed tokens unknown; saved-source proxies are partial.

Reproduce: `python3 tools/verify_portable_julia.py --config src/ace-julia/study_config_ka.json`, then JACC config reusing saved references. Do not run a new series merely to refresh metadata.

All4fivecaptures:52224exactprefixvalues +182784rawoutputvalues. Allocationguardscheckedexactly56/trial (notserialized inartifact), finiteness+algebraicoracle1e-10+1e-10rel. Analyticuniformequilibrium28672valuesexact. Independentwitness.jsonandfixturegeneratorhashesretained. Native6kernelbodytextunchanged; dimensions16/24specializedviaadapter, no400cubedcertification. CUDAjl1Dswapmappingaligned3Dsource(P1). Mean4fixturepipelines: {"cuda": 0.00028252025, "cuda_jl": 0.0008078945, "ka": 0.0009103265000000001, "jacc": 0.001429378} seconds.

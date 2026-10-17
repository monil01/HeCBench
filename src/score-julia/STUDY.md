# score: scientific and implementation observation

NativeconfidenceTopK is an approximatehistogramselectionalgorithm whoseverifiercomparessortedindexsets andcounts. Genuineversionsretainhistogram/warp-suffix/atomicrankmapping. ExplicitCUDAshuffleandatomicbackendextensionsmakethesehybridports. Theuniquebinfixture permitsanindependentexactTopKsetwitnesswithoutpretendingarbitrarywithin-bintiesaredeterministic. Thisdoesnotvalidatefulldetectorquality,rawselectionorder,orallbinboundaryrounding.

## Reference audit

HistoricalJuliaonlyincludesunrelatedmcpr program, nohistogramTopKselection. Repairedreferenceimplementsnative2048binhistogram,firstwarp64groupsuffixscan,atomicrankallocationandcount,256threadsperclass/batch.

## Numerical and performance evidence

Configuration `score_b2_c5_priors1024_bins2048_k10_unique_eligible_bins_v1`, result `/home/imo/HeCBench_julia/.porting-state/portable_julia/runs/20261017T190227329363Z_score/result.json`. Five in-process trials per model; full outputs checked in every trial against both references; mean trials2–5. Warmup precedes trial1. Native kernel unchanged. This named configuration does not certify all CLI modes or physical model validity.

## Cost and failures

One saved candidate and one compiler/JIT process per target model; no observed target repair. Historical CUDA.jl attempts unknown. Reference correction or documented P1 mapping phase is separate from target translation. Observed prospective usage intervals are recorded separately; exclusive billed translation cost is unknown. Final-source proxies remain partial.

Reproduce: `python3 tools/verify_portable_julia.py --config src/score-julia/study_config_ka.json`, then JACC config reusing saved references. Do not run a new series merely to refresh metadata.


Independent analytic witness inverts73*i+19*g modulo1024 to recover the10highestscoreindices without histogramorCPU-sortreuse. All100canonicalindices+10counts matchall20savedartifacts. CPUFloat32binconstruction has614distincteligiblebins/group; nativeandJuliafixtureselectedsetagreementdoesnotestablisharbitrarytied-binbehavior orintermediatebinprecision. Sortedindexnormalization isoutside timing andmatchesthenativeverifiercontract. Flatparententrypointnowinvokesthegenuine namedstudyreference; historicalunrelatedmcpralias isarchived, notclaimedCLIcompatible.


Prospective rawusage intervals include cached/repeatedcontext and coordination; the parentworkflow contains its childphases and must not be summed with them. Exclusive billed translation cost remains unknown; source-size proxies remain separately labeled.

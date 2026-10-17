# libor: scientific and implementation observation

ThisLIBORbenchmarkhasdeterministicidenticalpaths(fixed.3shocks),soarrayagreementdoesnotestablishMonteCarlostatisticalquality. FullpricesandlastinitialrateGreeksarecheckedagainstindependentFloat64forwardvaluation+finite-differences;preservedglobalvsprivatehistorymappingisakeyperformanceconfound,separatefromfinancialoperatoraccuracy.

## Reference audit

HistoricalJuliaforward/reversealgorithmsareactualLIBOR,notstub. Globalperpathscratchreplacesnativeprivatearrays;explicitP1resource/mappingadaptationpreservedforcomparability,notincorrectfinancialoperator. Constants/loopsshiftedonlyNPATH257/GRID4innamedstudy;originalbytesarchived.

## Numerical and performance evidence

Configuration `libor_80tenor40step257paths_three_market_cases_value_greek_v1`, result `/home/imo/HeCBench_julia/.porting-state/portable_julia/runs/20261017T210350720259Z_libor/result.json`. Five in-process trials per model; full outputs checked in every trial against both references; mean trials2–5. Warmup precedes trial1. Native kernel unchanged. This named configuration does not certify all CLI modes or physical model validity.

## Cost and failures

HistoricalCUDA.jl genuineglobal-scratchdevicefunctions reused,not replaced. One saveddevicecandidate andone compiler/JITprocesscheck perJulia model; noobservedtargetrepair in current check. SourceP1globalvsnativeprivatescratch is mappingdisclosure,notobservedwrongoutput orcompilerfailure. Same-benchmarkverificationwaiting andscientificbookkeeping fall inside model activitywindows; notexclusivecodingcost. Historical CUDA.jl attempts unknown. Reference correction or documented P1 mapping phase is separate from target translation. Observed prospective usage intervals are recorded separately; exclusive billed translation cost is unknown. Final-source proxies remain partial.

Reproduce: `python3 tools/verify_portable_julia.py --config src/libor-julia/study_config_ka.json`, then JACC config reusing saved references. Do not run a new series merely to refresh metadata.


## Financial sensitivity and mapping witness

All20artifacts passindependentFloat64forwardvaluation andcentralfinite-differenceGreek,h1e-6/5e-7stable(maxdifference3.56e-8). Everyartifactchecks333serializedinputslots exact(1332bytes) and2313perpathoutputs:3cases×257paths×(value-onlyprice,adjointprice,lastinitialrateGreek). {"max_price_error": 0.00036163155616009135, "max_last_initial_rate_greek_error": 2.729621064645471e-05, "forward_vs_adjoint_price_gap": 0.0001220703125}. Firstcase reproducesnativepublished224.323price/21.348Greek;othercases exercisechangedmarketparameters,includingazero-sensitivitycase. Native__expf/__fdividefvsJuliaexp/divide explainpossiblearithmeticdifferencesbutarenotcounter-profiled. Nativeprivate3280-elementhistoryvsJuliafamilyglobalperpathscratch is anexplicitP1resourcemappingconfound,notafinancialoperatorfailure. NoMonteCarloconfidenceintervalorfullJacobiancertification;pathshocksremainnativeconstant. OnlyGRID_SIZE andNPATHmacroschangedinarchivednativecopy;allkernel/functionsverbatim. Onecandidate/checkeach,nofailedtargetattempts.


Prospective rawusage intervals include cached/repeatedcontext and coordination; the parentworkflow contains its childphases and must not be summed with them. Exclusive billed translation cost remains unknown; source-size proxies remain separately labeled.

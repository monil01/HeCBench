# gd: scientific and implementation observation

Thisbenchmark trainsregularizedsparse-logisticclassification,notrelatedmcpralias. NamedunitnormCSRrowscontainknownlabelcorrelatedsignal. Fullfinalweights/gradient/pre-updateobjective/L2/countarecheckedagainstindependentFloat64optimization,withfinite-differenceanddescentwitness;aggregateobjectivealonewouldnotcertifyallweights. CUBvsmanualnormreductionandnativecudaMemsetvsCUDA.fillareexplicitresourcemapping/hostAPIconfounds.

## Reference audit

HistoricalJuliafilealiasesmcpranddoesnotimplementanysparse-logistictraining. PreserveoriginalbytesandimplementactualnativeCSRcompute/atomicgradient/regularizedupdate. NativeCUBblocknormreplacedbydeclaredbinarysharedtree;full512featuresavoidunmeasuredpartialCUBparticipationcase.

## Numerical and performance evidence

Configuration `gd_logistic_csr257x512_unitrows_32training_fullfinalstate_v1`, result `/home/imo/HeCBench_julia/.porting-state/portable_julia/runs/20261017T211612814945Z_gd/result.json`. Five in-process trials per model; full outputs checked in every trial against both references; mean trials2–5. Warmup precedes trial1. Native kernel unchanged. This named configuration does not certify all CLI modes or physical model validity.

## Cost and failures

One saveddevicecandidate/onecompilerJITprocesscheckeachforrepairedCUDA.jl,KA,JACC;noobservedcheckfailure. HistoricalmcpraliasS1sourcefinding separatefromnewrunattempts. Resource/collectivemappingP1CUBnormvsbinarytree,notwrongoperator. Historical CUDA.jl attempts unknown. Reference correction or documented P1 mapping phase is separate from target translation. Observed prospective usage intervals are recorded separately; exclusive billed translation cost is unknown. Final-source proxies remain partial.

Reproduce: `python3 tools/verify_portable_julia.py --config src/gd-julia/study_config_ka.json`, then JACC config reusing saved references. Do not run a new series merely to refresh metadata.


## Independent optimization witness and region scope

All20artifacts passfull1027finalstates+3086exactserializedinputslots,includingall512weights/all512gradiententries/old-pointobjective/L2/exactcorrectcount. IndependentFloat64mathematicalsigmoidgradient agreeswithcentralfinite-differences ofregularizedlossatiterations0/31coordinates0/13/511(maxerror 1.871001409359252e-10). CPUlossdecreases 0.6931471805599466 to 0.32013249674126987;knownlabelcorrelatedfeaturelearnedweight 2.1342571835891597,all257classified. UnitnormrowsboundaverageHessiannorm<=.25+lambda,step.4satisfiesdescentbound. FullGPUfinalstateerrors {"weights_max_abs_error": 3.7141877484714314e-07, "gradient_max_abs_error": 5.2387414719134995e-05, "objective_abs_error": 8.579867638047745e-05, "l2_abs_error": 6.583825022943302e-07}. CPUtrajectory/descent is independentwitness;intermediateGPUtrajectoriesarenotretained. NativeCUBnormreductionandJuliabinary256wordsharedtreeareexplicitmappingadaptation;CUDA.fillvsnativecudaMemsetandJACCtemplatecopy/barrieraredisclosedperformanceconfounds,notframework-onlycausalclaims. Native512featureblocksfullypopulated;partialfeaturescopeunmeasured,notdeclaredunsupported. Nativekernels/sourceunchanged;syntheticunitnormCSRfixturedoesnotcertifygisettedatasetaccuracy.


Prospective rawusage intervals include cached/repeatedcontext and coordination; the parentworkflow contains its childphases and must not be summed with them. Exclusive billed translation cost remains unknown; source-size proxies remain separately labeled.

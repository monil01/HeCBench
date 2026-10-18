# lavaMD: scientific and implementation observation

Scientificintent andimplementedoperator needseparateevidence. OriginalLavaMD v is independentlyrandom, so r2=vA+vB-dot is not automatically squared distance; the namedGaussiancompatiblefixture enforcesv=half norm². Exactcollocatedcharge-sum/zerovector andindependentfullpairoracle makeoperatoraccuracyvisible. FP32summation createsmaxabs~.00713potentialerror andnonzerochargeweightedforceimbalance evenwhenallmodelsagreewithinpredeclaredcontracts; noMDtrajectoryaccuracy follows. MatchingAoS/sharedgeometryalone also doesnotisolateabstractionperformance: currentJulia baseline~6xnative includes metadataextraction/structstore/boundscheck/codegenconfounds. Preserveallvalidfirstseries ratherthanretimeforpreferredspeed; a futureperformancecontrol must be separateconfiguration.

## Reference audit

HistoricalJulia genuine mathematicalpairaccumulation usesSoA/global loads, omitting nativeAoS/shared staging and changing q*fs*dx grouping. Restored source16BFOUR_VECTORABI/656Bboxmetadata/3600Bshared/128threads/100particles/neighborder andsourceq*(fs*dx) grouping; historicalmathnotS1. Nativekernelunchanged. Targetreads metadata asexactUInt64words toavoidlarge656Btuplecopies; low-level loads/storelowering/boundschecks notinstructionequivalent.

## Numerical and performance evidence

Configuration `lavaMD_AoS_shared_B1_2_3_forwardmodel_v1`, result `/home/imo/HeCBench_julia/.porting-state/portable_julia/runs/20261018T073741285423Z_lavaMD/result.json`. Five in-process trials per model; full outputs checked in every trial against both references; mean trials2–5. Warmup precedes trial1. Native kernel unchanged. This named configuration does not certify all CLI modes or physical model validity.

## Cost and failures

All3firstauthoredcandidate/process, noobservedGPUrepair. Sharedreferencephase aligns genuinehistoricalmathwithsourceAoS/shared/ABI/operationgrouping andbuildsindependentfullpairoracle+predeclaredforwarderrorbudgets. KAinlines barrierbodywithownlocalmem, JACCuses samehelper/publicsingle3600Bsharedtemplate. Latermodelsreusevalidatedsource/science, notcoldstart independenttranslations. Activitywindows include cached/repeatedcontext, scientific/ABIanalysis, documentation, waiting/coordination/metadata; workflowoverlapsphases donotsum asbilledexclusivecost. Historical CUDA.jl attempts unknown. Reference correction or documented P1 mapping phase is separate from target translation. Prospective activitywindows include cached/repeatedcontext, source/science/ABIanalysis, docs/waiting/coordination/sharedmetadata; not exclusive billedcost. Historical consumption unknown andmain-sourceproxy omitssharedcode.

Reproduce: `python3 tools/verify_portable_julia.py --config src/lavaMD-julia/study_config_ka.json`, then JACC config reusing saved references. Do not run a new series merely to refresh metadata.

## Scientific and performance limits of the named operator

All20captures62343Float64encodedwords:23919exactprefix, completeactualmetadata/particle/chargebytes,14400actualFloat32outputs and120encodedguardwords. Julia3bitidentical; native/Julia maxabs0.000244140625/maxrel.02405348736 nearzero underunchangeddualreferenceatol2e-4/rtol2e-5. Independentfullpairoracle maxabs0.00712796315679, maxconditionalforward-modelbudgetratio0.08467186107; actualweightedforceimbalancesretained, notassertedexactconservation. Collocatedexactcharge-sumpotential51.625/zerovectors pass. Error-modelSHA ed11c4982d18db7d49bcc91f21e2db8d56839ed821bc4ef2dbf4443a29fa4768 frozenbeforeallGPUchecks, documentednormalFP32/standardexpf assumptions; notformalglobal/applicationtoleranceproof. [CUDAstandardexpfaccuracy](https://docs.nvidia.com/cuda/archive/12.2.0/pdf/CUDA_C_Programming_Guide.pdf) informs theconservativeSFUmodel; nofastmathkernel.

Sourceoriginalrandomv is not automatically half squarednorm. B3source-compatiblev=.5norm² supports a scaledGaussiannegativegradient operator (vectorfactor2/a2), notgeneralphysicalforce-law/MDtrajectory/thermodynamicaccuracy. B2remainsformal sourcev/coordinateoperator. Chargeweightedantisymmetry mathematicaloraclechecks make FP32netdriftvisible; completepointwiseoutputs aretheprimaryevidence, notscalarnetforce alone.

All3firstauthoredcandidate/processpass, ownframeworkkernels. Native/KA/CUDAjlshared3600B, JACCpublicshared adds900Float32templatecopies/initialbarrier, resourcechoiceexplicit. NativeC++uncheckedfields vs partialJulia@inbounds coverage/UInt64metadataextract/wholeStructvaluestores arecodegenconfounds despitephysicalAoS/sharedgeometrymatch. Preservefirstvalidtimingseries; do not rerununtilpreferredspeed. Meanouter2-5seconds {"cuda": 0.00030552525, "cuda_jl": 0.0018611822500000002, "ka": 0.00178012125, "jacc": 0.0018528467500000003} sums3sourceonekernelROIs, original30boxCLI/offloadtimeunmeasured. Noisolatedabstraction/rooflinebottleneckclaimwithoutcounters/controllednewconfig.

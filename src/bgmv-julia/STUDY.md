# bgmv: scientific and implementation observation

LoRAgatherprojectionsrequirepertokenadapterselectionandmixedprecisionstateupdates. Unlikehistoricalserialperoutputmapping, allversionsnowretainnativeoneblockperoutputandall6blocksizes/4modes. PortabletreeadaptationreplacesCUB; JACCaddsfullsharedtemplatecopy. Independentinteger-dotandFP16recurrencechecksvalidateactualpostrepeatstates. Mapping/reductioncostsandshapechangesconfoundanyabstraction-onlyperformanceinterpretation; notrainedmodelaccuracyclaim.

## Reference audit

HistoricalJulia computescorrectLoRAoperator butserialdotoneoutputperthread atonly256threads,ignoringnative6blockreduction/vector8mapping. P1mappingdifference, notestablishedwrongscientificoperator. Newreference preservesoneblockperoutput,6blocksizes,scalar/vector8shrink,expand overwrite/add; CUBreplacedbysharedbinarytree.

## Numerical and performance evidence

Configuration `bgmv_t4_h72_r16_l3_all6blocks_4modes_dyadic_v1`, result `/home/imo/HeCBench_julia/.porting-state/portable_julia/runs/20261017T184834173440Z_bgmv/result.json`. Five in-process trials per model; full outputs checked in every trial against both references; mean trials2–5. Warmup precedes trial1. Native kernel unchanged. This named configuration does not certify all CLI modes or physical model validity.

## Cost and failures

KA candidate1compiledbutall24caseoutputsremainedzero; anuntimeddiagnosticisolatedthispattern. Candidate2preservesonebasedindexvariablesacrossbarriers andcomputesderivedzerooffsetswithoutmutatingtheindexbinding. Numericallycorrectivecheckkeepsreferencearrays/timingsunchanged. Othermodelattemptsrecordedindependently. JACCcandidate1hit1024threadregisterlimit74regs/thread; candidate2narrowsknownsmallglobalindices toInt32withoutchangingblockgeometry, fixtureoroperator; finaloutcomerecordedseparately. Historical CUDA.jl attempts unknown. Reference correction or documented P1 mapping phase is separate from target translation. Observed prospective usage intervals are recorded separately; exclusive billed translation cost is unknown. Final-source proxies remain partial.

Reproduce: `python3 tools/verify_portable_julia.py --config src/bgmv-julia/study_config_ka.json`, then JACC config reusing saved references. Do not run a new series merely to refresh metadata.


Independent Python integer dots and binary16 recurrence match every4224value in all20savedartifacts exactly. KA required two candidates after an initial zero-output numerical failure plus oneuntimeddiagnostic. JACC required three candidates: 2D74register/thread andInt32-2D66register/thread candidates couldnotlaunch1024threads onV100, then1Dflattenedgrid passedallblocks. This is an observed candidate-specificresource limitation repaired through mapping, not blanketJACCunsupported. Exactcompilermechanism for theKAindexbindingfailure wasnotisolated.


Prospective rawusage intervals include cached/repeatedcontext and coordination; the parentworkflow contains its childphases and must not be summed with them. Exclusive billed translation cost remains unknown; source-size proxies remain separately labeled.

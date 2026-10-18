# kiss: scientific and implementation observation

RNGtranslationmustpreserve perthreadseedhash, wordwidthandcarry, not onlyprint samplelikeoutputs. The study checks thecompletegridstrideoutput includingsecond/third draws, uint32seedwrap andzero-local-seedfallback. Independentcarry useswidened addition whiletargetfollows sourcez<t test, providing a distinctarithmetic witness. Exactstream matching is a functional guarantee, not proof ofrandomquality. Sourceinitializationinsideeverylaunch materiallydefineswhatthismeasurement includes; shortseededstreams must not be reportedas original2GiBsteady-state throughput. Sourcezero-seedfallback aliases local_seed0 andUInt32MAX; proving nonzeroinitialstates is not proving injective seeds or statisticallyindependentstreams. This preserves sourcefallback policy, not an established violation of a promised qualitycontract.

## Reference audit

HistoricalJulia smoke kernel writes a marker and prints literalzero RNGoutputs/timing; no KISS algorithm. Preservedoriginal, restored actualUInt64 sourcegenerator/Murmur32thenMurmur64seedconstruction/LCG/xorshift/carry/gridstride, fullstream exactoracle. Nativekernel/header unchanged. Portable invalidstate diagnostic retains a branch but skipswrites rather than nativeassertabort outside a globallyprovednonzero-state precondition.

## Numerical and performance evidence

Configuration `kiss64_fixedG_33_G33_2G17_seed0_42_MAX_3repeat_v1`, result `/home/imo/HeCBench_julia/.porting-state/portable_julia/runs/20261018T072125062139Z_kiss/result.json`. Five in-process trials per model; full outputs checked in every trial against both references; mean trials2–5. Warmup precedes trial1. Native kernel unchanged. This named configuration does not certify all CLI modes or physical model validity.

## Cost and failures

Oneauthoredkernel candidate andonewholeprogram/deviceJITprocess permodel, noobservedrepair. Sharedreferencephase restoresallRNG/seedmath fromsmoke andbuildsindependentwidenedfulloracle/Pythoncontrols; KA thenJACC reuse auditedcommonbody andvalidatedpreviousmodel. Prospectiveactivitywindows include cached/repeatedcontext, sourceaudit/coordination/waiting/ontology/metadata, notexclusivebilledcoding; workflowoverlapsphases donotsum. One separate sharedscientific-annotation Python B2 (missing or keyword whitespace) repaired outsideGPU/devicecode; not a targetcompilerfeedbackrepair. Historical CUDA.jl attempts unknown. Reference correction or documented P1 mapping phase is separate from target translation. Prospective activity windows retain cached/repeated context, coordination/waiting/sharedmetadata scope, not exclusive billedcost; historical consumption unknown, source-sizeproxy partial and omitssharedcode.

Reproduce: `python3 tools/verify_portable_julia.py --config src/kiss-julia/study_config_ka.json`, then JACC config reusing saved references. Do not run a new series merely to refresh metadata.

## Complete seed/state/stream evidence

All20captures3145844UInt64words exactly match full independent expectedbytes:9scalarprefixwords,3145811actualgeneratoroutputs and24canaries; all4models bitidentical. Thereisno deviceinputarray in source, so scalarprefix plusdirectlaunchargument binding is theinputevidence, not inventedbufferreadback. FullUInt128modularCPUoracle doesnotinclude nativeheaders, independentlyuseswidenedsumcarry vs sourcez<t;21Pythonstreamcontrols/37draws agreeexactly. Namedcases exercise1,572,897carryupdates and2zero-local-seedfallbackstreams, includingseedUInt32wrap.

Murmur64oddmultipliers/xor-shifts prove allconstructornonzerostates; this is not a statisticalindependence proof. Sourcefallback mapslocal0/MAXtosameinitialw, a preservedpolicy not an establishedqualitycontractviolation (seed_alias_policy_review.json). No distribution/period orMonteCarloapplicationcertificate. SourceUInt64only, unusedheaderfloat/double/32specializationsnotclaimed. All3 models ownordinaryframeworkkernel withoutdevicebackendextensions; portableinvalidstatebranchreturnswithoutwrites ratherthan CUDAassertabortoutsideprovedvalidprecondition, explicitlyunmeasuredABI difference.

Perinvocation32Murmur64scrambles/stateconstruction staysinsideeachROI, sourcefixed4096x256grid preserved; named1..3draws/thread doesnotcertifysource2GiB/256drawstream throughput. Meanouter2-5 sum3caseseconds {"cuda": 0.00012501541666666666, "cuda_jl": 0.00015023025, "ka": 0.00015714650000000002, "jacc": 0.00017576408333333334}. All3 firstauthoredcandidate/processpass. One sharedPythonannotationkeywordsyntaxB2 recordedseparately, nottargetGPUcompilerfailure. NativeH2DvsJuliadevicefill canaryreset outsideROI, unisolatedcacheeffects; no rooflinebottleneck orcausalframeworkpenaltyclaim.

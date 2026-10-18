# mtf: scientific and implementation observation

Corrected SECONDARY onlineMTF: initialalphabet0..255, eachsymbolcopies256bytealphabetH2D, devicefindreturnsrank, emitsrankbeforemove, copiesprefixD2Hright-shift andupdateshostfront. Fullrankstream+finalalphabet; no erroneousglobal-inplacerelabelpass. Nativecorrectedbaseline retains host-drivenThrustfind+cross-devicecopy (0authoreddevicekernels). EachJulia model has1authored256threadsingleblocksearch: thread1setsrank256, fullblockbarrier, unique matchinglanewritesrank; samehostalphabet/rankread/prefixshiftcontrol andcompletion. Vendorprimitivevscustomsearch is an explicit mappingadaptation. Independentlast-occurrence-rankformula uses256recencytimestamps, mathematicallydistinctfrommutableencoder; fullrank/finalalphabet exact anddecoderroundtrip everyposition. All20capturepass4903outputs+2626exactprefix; explicitCPU/GPUallocationguards checked in180model-case-trials. Logical665344alphabetH2Dbytes+318214prefixD2Hbytes perfull9cases, notmeasuredDRAM/bandwidth.

## Reference audit

HistoricalJuliaonlywritesmarkerandprintsPASS/zeroelapsed; originalnative+CPUzeroalphabet causesactualCPUASanWRITE256past256bytes, andinit-onlypatchstilldoesglobal-inplacerelabel ratherthanonlineMTF (ABAexpected65,66,1versus66,65,66). Preservedalloldsource/diagnostics. FullycorrectedSECONDARYnative/CPUsemanticstudy changesalgorithm, withpredeclaredrank-before-moveall-bytecontract; notone-lineoruntouchedport.

## Numerical and performance evidence

Configuration `mtf_corrected_secondary_online_hostdriven_thrust_vs_framework_search_preallocated_v1`, result `/home/imo/HeCBench_julia/.porting-state/portable_julia/runs/20261018T023729446538Z_mtf/result.json`. Five in-process trials per model; full outputs checked in every trial against both references; mean trials2–5. Warmup precedes trial1. The faulty original native source is unchanged; the executed baseline is the explicitly corrected secondary algorithm, and the original program is not certified. This named configuration does not certify all CLI modes or physical model validity.

## Cost and failures

EarlierstencilCPUASanandsemanticwitness retainedaspriornativeoriginfailures; historicalJuliaS1separate. Newcorrectednative/CUDAjl/KA/JACCfirstactualcandidate/checkallpass; no observednewdevicecompiler/runtime/numericfeedbackrepair. OneworkerPythonSCOPEgenerationsyntaxB2(beforeexecution,notGPU) repaired; notclassifiedasmodelcompilerfailure. Referencealgorithmcorrection/measurementfixturecost separatelymetered beforeKA thenJACC. Historical CUDA.jl attempts unknown. Reference correction or documented P1 mapping phase is separate from target translation. Exact consumed tokens unavailable; final-source proxies are partial, never total costs.

Reproduce: `python3 tools/verify_portable_julia.py --config src/mtf-julia/study_config_ka.json`, then JACC config reusing saved references. Do not run a new series merely to refresh metadata.

## Conditional reference and independent mathematical evidence

This is an explicitly corrected SECONDARY online-MTF study. The original native and CPU code are unchanged and remain invalid: actual CPU ASan overflow and the safe ABA global-relabel counterexample are preserved. Initializing the alphabet alone would not correct the operator. The new owned native function initializes all256 unsigned symbols and emits each rank before moving the symbol; this algorithm repair is excluded from untouched-native aggregates. The historical Julia smoke is preserved separately.

The independent recency formula counts more-recent symbols for a repeated byte; for a new byte w it uses w plus the number of previously seen symbols greater than w. Final order is decreasing recency followed by unseen ascending symbols. All20 captures exactly match2599 emitted ranks plus2304 final-alphabet bytes, with decoder round trips at every position and all180 case-guard groups passing. Known answers include ABA=[65,66,1], ascendingallbytes=0..255 and descendingallbytes=255repeated. All2626 input-prefix values match exact bytes.

The measured region is a preallocated named pipeline, including identityreset and host-controlled alphabetH2D/devicefind/rankread/conditionalprefixD2Hshift/hostmove and devicewidecompletion. Original allocation-inclusive timing and large CLI sizes are unmeasured. Native uses Thrust vendorfind (zero authored kernels); targets use one own framework search kernel plus the same host state/copy flow. Whole pipeline class is mixed_host_device_pipeline; device component is framework_device_kernel. Vendor algorithm/temporary storage, compiled host language and interface differences prevent an intrinsic framework-penalty interpretation. No float roofline applies;665344H2D/318214D2H are logical specified alphabet payloads, not measured memory traffic or bandwidth.

# Triad kernel-only configuration

RQ1–RQ5 / E1–E5. Configuration triad_kernel_only_n4194304_r100_v1.

kernel only; native original streamed host/device transfer pipeline and tile sweep are excluded for ALL4 models; this does not port or measure fullnative transferbenchmark. The benchmark is designed for transferperformance, so thiskernel-only result must not be substitutedfor its originalend-to-endtime. Allfour models insteadtime the same synchronized compute launchloop, normalizedperrepeat, block128, n4,194,304,100internalrepeats, transfers/reset/JITexcluded. Sharednontrivial deterministictwoinput fixture replaces RNG. Native triaddevicefunction unchangedviaoriginalsourceinclude; historicalJulia sourcepreservedbyteforbyte.

Full4,194,304Float32 outputs allfive trials passindependentCPUoracle andbothGPUreferences (atol1e-3 rtol0), maxnativecrossmodelabs1.90735e-6, exactCUDA.jl. FMAroundingdifferencesacceptedunderoriginalcriterion.

| Model | Fiveus | Meanlastfourus |
|---|---|---|
|cuda|66.125300, 66.237500, 66.156810, 66.155700, 66.245100|66.198778|
|cuda_jl|554.499950, 66.342500, 66.184900, 66.164110, 65.991100|66.170653|
|ka|1196.094890, 66.395400, 66.858610, 66.110300, 66.050600|66.353728|
|jacc|335.096650, 167.511720, 164.667600, 168.856240, 164.254000|166.322390|

Eachnewvariant onecandidate, onecompiler/deviceJITcheck, zerofeedbackrepairs; firstcandidateverified. JACCafterKAverified, unchangedrefs/KAfive-trialbundle reused withhashvalidation. Partial saved-summary/source proxies do not establish totalconsumedtokens; parentusageis sharedunallocated. No unsupportedfeatureobserved forconfiguredkernel. Ontology streamingmap, logical2FP32FLOP/12bytesperoutput, profiling underwayseparatefromthese5trials; no rooflinepositionclaimeduntilcountersprocessed. Raw STATE/portable_julia/runs/20261016T230400879312Z_triad.

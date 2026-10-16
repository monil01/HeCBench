# axpby multi-tensor study

RQ1–RQ5 / E1–E5. Named configuration: eight tensors, chunk4096, ten internal repeats. Full 96,668Float32 outputs passCPUoracle and BOTHreferences exactly on ALLfive trials (atol1e-3 rtol0). Native original AxpbyFunctor and multi_tensor_apply includedunchanged; existingCUDA.jl block/chunk selection and4wayILP scalarloops preserved. Deterministic sharedfixture replaces originalnative RNG; nativevectorizedloads+hostmetadata versusJulia scalarloads+device lengthscan is an explicitperformanceconfound. Otherchunks oforiginalCLI sweep are notmeasured.

All models prewarm exactlaunch path beforefive synchronized hostclocktrials. Timingincludeslaunch/dispatch and excludesreset/transfers/JIT; nativehostmetadata construction remains inside originaldispatch. Trial1discarded, meanlastfour.

| Model | Fiveus | Meanus |
|---|---|---|
|cuda|4.757000, 4.869100, 4.848100, 4.786000, 4.783100|4.821575|
|cuda_jl|10.521100, 10.389000, 8.586100, 8.552100, 8.516000|9.010800|
|ka|21.375200, 12.779100, 9.425100, 9.189000, 8.748000|10.035300|
|jacc|27.934300, 29.986300, 22.032300, 21.290200, 21.617200|23.731500|

Onecandidate andonecompiler/device-JIT invocation pervariant, nofeedbackrepair. KAverifiedbeforeJACCauthored; unchangedrefsandKAfive-trial bundle reused withdependencyhashchecks. No unsupportedfeatureobserved. Partial saved-summary/source proxy cannot establish totalconsumedtokens; exactparent sessioncounters are shared/unallocated. Raw evidence `.porting-state/portable_julia/runs/20261016T225528116664Z_axpby`. Ontology multi-tensor-map/chunk-scheduling reviewed; no measuredrooflineclaim. Futureprofile/scaling needed beforeattribution.

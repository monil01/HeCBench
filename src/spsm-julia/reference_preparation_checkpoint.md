## Completed result

Allfourversions pass exact full-artifact and independent residual checks in `.porting-state/portable_julia/runs/20261018T110338267963Z_spsm/result.json`. See `output_audit.json`, `source_checker_review.json`, `study_notes.json` and per-version `verification.json`. The following is the earlier preparation plan, not pending work.

# Historical preparation plan: spsm

Source-only audit complete. NO implementation edits/archives drafted yet. CUDA actually sparse lower-triangular multiple-RHS solve via cusparseSpSM. HistoricalJulia one-line ../mcpr-julia alias (S1). Preserve full NVIDIA notice from main.cu in derivative code/docs. Root owns canonical+workerqueues after untouched stencil confirmation.

Source policies: Float32/Int32CSR, ROW-order denseB/C ld=nrhs, LOWER fill, NON_UNIT diagonal, NONTRANSPOSE both, alpha1, SpSM_DEFAULT. Handle/descriptors/query/workspace/SpSM_analysis outsideROI, repeatedsolve+devicecompletion insideROI (Makefile m20082,rhs128,nnz150616,R100). System actualcuSPARSE12001DSO withdladdr/functionaddress+SHA asspmm/spgeam. All3libraryinterop/zeroauthoredGPUkernels/noGPUlowering, explicithostorchestrationprogramcheck counters.

Critical checker weakness: source derives B=A*Ctrue and uploads Ctrue into GPUOUTPUT before anysolve. A skippedsolve would pass its residualchecker byconstruction; static source-proof only, do notclaimobservedskippedGPUkernel. Namedverification must poison GPUC outsideROI AFTERwarm/analysis/beforeeachtrial, then checkeveryoutput againstindependentknownsolution andresidual. This strengthensoutputinstrumentation, doesnotchangeSpSMAPI/mathematicalreference. Record initialGPUCpoisonactualreadback inprefix. CPUoracleX MUSTNOTbeuploadedtoGPU! BderivedfromX, GPUonlyA/B/poisonC.

Plan strong exactfixtures: namedoriginalm20082/rhs128/nnz150616 dims/count retained via controlledlowerCSR +compact1,17,33,257 controls/oddRHS/zeroRHS control ifallAandBdefined. Diagonal2or4, offdiag±1/64, atmost7strictlower entries/row, columns aroundr/2+t13 modulo r (deduplicate/fillunique), distribute exactly130534offdiag forfullcase. Strictdiagdominance rowoffsum<=7/64, noillconditionedsourceRNGclaim. TruthX signedinteger/8 varyingbyrow+RHS, exactknown; B computedintegerunits1/512 fromdiag*X andoffdiag*X, boundedabsoluteunits<2^24. Everyforwardsubstitutionpartialsum exactFP32, power2diagonaldivisionexact; validateFraction oncompactcases, separateFloat64CSRresidual/fullknownsolution allcases. ActualfulloriginalRNGinitializer/non-dominantconditioning/generalfloat/solverconvergence unmeasured. Named R10vsoriginal100explicit. Keeporiginalshape/fulloutput10.3MB, overall20artifacts~1-2GB within~115GBfree. Ifruntime/ISA contradictsexactness preservefullcapturesbeforerevisingacceptance.

Fivecheckedtrials (mean2–5) eachmodel, identicalsameDSO/APIpolicies/allinputs/ROI. FullprefixactualGPU Avalues/rp/ci/B/outputCpoison/guards (UInt32rawF32bits+Int32). Post allA/Breadonly+allC/guards. IndependentCPUoracleXrawbits separatelycommitted; usefinite/dtype/shortreadfailfast andruntime_assetsfornative tempcwd.

Open rootmeters spsm/shared_benchmark_workflow andcuda_jl_reference_preparation began2026-10-18T10:49:34.347Z. NoKA/JACCmeters yet; open BEFORE drafting wrappers wherepossible; no historicaltokenrecovery/exclusiveattributionfiction. GPU6only globalrunnerlock, noexternalflock. RootaloneGit.

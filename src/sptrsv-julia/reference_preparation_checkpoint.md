# Completed SPTRSV preparation

HistoricalJulia unrelatedMCPRinclude S1. Restored customsyncfree solver, NOT vendorSpTRSV. Actual original warp-mode failures preserved (27/60/249 mismatches firstrepeat on32/65/257 controls; LP1threadcasepasses). Only corrected native change __syncwarp after sharedstore and each tree stage; all20 corrected results exact. Conditional corrected-native stratum/exclude untouched-native aggregate.

.porting-state/portable_julia/runs/20261018T113145966047Z_sptrsv/result.json

CUDAjl three host program candidates/process checks: missing JSON3 predevice, CUDA macro map argument shadows Base.map predevice, then first actual GPU solver candidate passes. KA first wrapper/GPUprocess/check. JACC two wrapper/GPUprocess/checks: first runtime device exception with 0 dynamic shared bytes, fixed explicit2048bytes. Shared CPU fixture generator TypeError once before GPU (array.extend int array), fixed float generator; one original-header CPU extractor build/process. Native four build invocations: bad relative include flag first, original failed five captures second, corrected third and same corrected source paired fourth. Native correctness repair is shared reference work, not three target numerical repairs. One separate compile-only CUDAjl PTX inspection, zero launches.

All20savedcaptures exact/fullCSRresidual0; currentonlyconditionalcorrectednativeprimary, no untouchednativeaggregateclaim. SeeSCOPE/REPRODUCE/output_audit/sourcewitnesses.

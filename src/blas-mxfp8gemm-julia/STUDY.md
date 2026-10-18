# MXFP8 block-scaled GEMM: reviewed hardware prerequisite

The unchanged native source builds successfully with available CUDA 12.9.41 and its matching cuBLASLt headers/library. On physical GPU6, the native guard reports: `Skipped: MXFP8 block-scaled matmul requires compute capability >= 10.0, found sm_70 (Tesla V100-PCIE-32GB)`. The earlier CUDA 12.0 toolkit guard is therefore resolved; the actual V100 hardware prerequisite remains unmet. Exit zero is a skip, not a numerical pass.

No matrix multiplication, five-trial timing series, or numerical comparison ran. Native FP8 E4M3 storage, UE8M0 scales, FP32 accumulation and BF16 output describe the requested native contract, not implemented Julia precision. The historical CUDA.jl MCPR alias remains byte-preserved and rejected. KA/JACC have no genuine target implementation; framework expressibility is untested and neither is classified unsupported.

The CUDA12.9 build used validated linker arguments and compiled unchanged source on its first new review attempt. Historical CUDA12.0 evidence and this CUDA12.9 review remain in `.porting-state/portable_julia/diagnostics/blas-mxfp8gemm/`. The hash-bound adjudication and exact commands are in `blocked_300/blas-mxfp8gemm.json`. No lower-precision substitute or software emulation is counted as this hardware path. Review is complete for the selected hardware; resume requires an explicitly allocated compatible sm100-or-newer platform.

## Active150 final recheck

The final review rehashed all eight earlier source, raw build/guard-log, toolkit-header and review records and found them unchanged. The recorded CUDA12.9.41 build resolves the toolkit prerequisite, while the actual sm70-versus-sm100 source guard remains conclusive for this selected platform. No repeated GPU process, target draft, numerical/timing pass, or framework-unsupported claim was introduced. See `final_hardware_limitation_review_150.json`.

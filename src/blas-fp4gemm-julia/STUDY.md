# FP4 block-scaled GEMM: reviewed hardware prerequisite

The unchanged native source builds successfully with available CUDA 12.9.41 and its matching cuBLASLt headers/library. On physical GPU6, the native guard reports: `Skipped: FP4 block-scaled matmul requires compute capability >= 10.0, found sm_70 (Tesla V100-PCIE-32GB)`. The earlier CUDA 12.0 toolkit guard is therefore resolved; the actual V100 hardware prerequisite remains unmet. Exit zero is a skip, not a numerical pass.

No matrix multiplication, five-trial timing series, or numerical comparison ran. Native FP4 E2M1 storage, UE4M3 scales, FP32 accumulation and BF16 output describe the requested native contract, not implemented Julia precision. The historical CUDA.jl MCPR alias remains byte-preserved and rejected. KA/JACC have no genuine target implementation; framework expressibility is untested and neither is classified unsupported.

The first CUDA12.9 build command passed a versioned shared-library pathname in an unsupported nvcc argument form (B2, shared adapter). The corrected linker arguments compile the unchanged source. Both attempts remain in `.porting-state/portable_julia/diagnostics/blas-fp4gemm/`. The hash-bound adjudication and exact commands are in `blocked_300/blas-fp4gemm.json`. No lower-precision substitute or software emulation is counted as this hardware path. Review is complete for the selected hardware; resume requires an explicitly allocated compatible sm100-or-newer platform.

## Expansion150 reassessment

The unchanged CUDA12.9 build/source/hardware-guard logs and toolchain/header hashes were independently revalidated in expansion150_evidence_review.json. No duplicate GPU run, new GPU compiler attempt, or Julia translation was performed in this review. All3 target implementations remain unverified; the historical CUDA.jl alias is archived and rejected. Requested API compute32F is a contract, not evidence of actual executed accumulation. Resume requires compatible allocated hardware.

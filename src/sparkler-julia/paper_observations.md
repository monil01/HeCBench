# Sparkler observations (RQ1–RQ5)

RQ1: Full source Makefile FLOAT32 pipeline plus two rank/padding controls passes exact full four-model comparisons and independent integer witnesses; four MPI ranks share GPU6, as in the original default-device source.

RQ2: CUDA.jl/KA/JACC used 5/1/2 whole MPI evaluations, but each has one authored mathematical body and one successful authored-JIT program (four rank processes). Host API errors are separate from device-kernel errors. Actual context-inclusive target windows are {'cuda_jl': 4344543, 'ka': 1551653, 'jacc': 1978645}. CUDA.jl and KA windows have disclosed shared/native/annotation interleaving and are excluded from clean calibration.

RQ3: Last-four mean seconds: {'cuda': 6.369190231999999, 'cuda_jl': 6.526667549500001, 'ka': 6.378580274000001, 'jacc': 6.3763163115}. The primary timer covers source GEMM/completion/barriers, not the authored preparation kernel or whole CLI; event preparation times are separate.

RQ4: The same pinned cuBLAS operation executes all primary numerical work. One public framework initializer per target uses matched source 256-thread blocks, 64-bit indexing and column-major layout; no counters or causal lowering result is established.

RQ5: Public KA/JACC kernels suffice for preparation, while this end-to-end source operator still depends on CUDA/cuBLAS/MPI interoperation. These measurements do not establish four-GPU scaling, alternate-backend portability or unused FP16/INT8 capability.

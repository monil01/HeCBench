# Target tensor modes

## Conditional mode approvals

| Target | Kernel families | Source/API mode | Calibrated conversion |
|---|---|---|---|
| WMMA native CUDA | gemm_impl0, gemm_impl1 | FP16 A/B, FP32 accumulator/C/D, 16×16×16 WMMA | 512 FLOPs per sm__inst_executed_pipe_tensor.sum instruction, **sm70 only** |
| WMMA CUDA.jl | gemm_impl0, gemm_impl1 | same, explicit CUDA.WMMA | same conditional conversion |
| WMMA KA | gpu_gemm_impl0, gpu_gemm_impl1 | same, KA dispatch/indexing plus CUDA.WMMA device interop | same conditional conversion |
| WMMA JACC | _parallel_for_cuda_MN callback gemm_impl0/1 | same, JACC dispatch/indexing plus CUDA.WMMA device interop | same conditional conversion |

`approval_table.json` supplies eight per-model/family rows with exact demangled selectors and required architecture/source/dependency/counter conditions. The default collector remains unchanged. These approvals establish the source/API tensor mode; **selected successful target SASS has not been observed**, and no target kernel was run or compiled by this reviewer.

`review.json` binds selected result/config, all selected source/input/oracle hashes, current compiler headers/binaries, current Julia project/manifests and package source hashes. Current dependencies are prospective bindings, not retroactive certification of missing historical dependency captures.

Native fragment overloads use half matrix inputs and float accumulators; the CUDA12.9 header implementation selects `__hmma_m16n16k16_mma_f32f32`. Actual Julia input construction creates Float16 A/B; `fill_c(0f0)` creates a Float32 accumulator and Config{16,16,16,Float32} fixes the tensor result. Installed CUDACore6.2.1 generated dispatch selects `llvm.nvvm.wmma.m16n16k16.mma.row.col.f32.f32`. This is stronger evidence than storage/oracle precision alone. KA/JACC use a CUDA-specific device API; this is not a generic backend tensor interface.

The scalar FP32 epilogue `.5*acc+2*C` remains a separate scalar projection. Impl0 combines within the K loop, impl1 after it. Do not add scalar and tensor peak ceilings, or infer useful-work efficiency from physical instruction work.

## Retained disassembly: limited provenance

CPU-only `cuobjdump --dump-sass /tmp/wmma-direct-probe/study_native` succeeded. The retained probe's two gemm_impl functions contain sm70 HMMA.884.F32.F32.STEP0–3, with 20 static occurrences of each STEP per function; scalar FFMA also appears. These are static listings, not dynamic operation counts.

That executable is a retained **failed CUDA12.0 diagnostic probe**. The successful selected native measurement was built using CUDA12.9 through the retained result reuse chain. The probe is therefore illustrative retained target code, not the measured binary or proof that a particular counter row executed it. No retained Julia JIT cubin/SASS was found in the bounded search. Exact receipts and disassembly are preserved here.

## Other targets

- blas-groupgemm native/CUDA.jl: FP16 inputs and FP32 vendor API compute are established, but vendor instruction choice is unknown. No tensor conversion approved.
- blas-groupgemm KA/JACC: authored scalar FP32 multiply/add loop with FP16 input/output. No WMMA conversion.
- blas-gemmEx / blas-gemmEx2: vendor paths include distinct FP16/FP32 accumulation and other modes; API labels do not prove HMMA variant or per-kernel stage. No blanket tensor conversion.
- lut-gemm: half scalar/LUT and packed half2 arithmetic/atomics are distinct from HMMA. HFMA2 does not justify a tensor conversion.
- tensorAccessor / tensorT: scalar matvec/accessor or data-layout work; the name does not establish tensor instructions.

All these records retain source/config/result bindings. Unknowns remain unknown.

## ROI limits

WMMA has eleven selected cases, thirty warmup launches per case and one hundred launches per timed case/trial. Native also runs an excluded impl1 96×80 geometry diagnostic before the scientific warmups; it shares the kernel family and precision mode. A tensor-mode approval never assigns ROI membership or makes a warmup/diagnostic representative of retained trials. No new GPU measurements, completeROI trace, wholebenchmark bottleneck, or timing replacement follows.

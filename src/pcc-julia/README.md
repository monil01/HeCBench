# Same-vendor fMRI correlation component

The source normalizes voxel time series on the CPU, delegates a Gram matrix to cuBLAS SGEMM, then extracts packed pairwise correlations. Historical Julia is a genuine vendor-GEMM/single-extraction implementation and is preserved. All current versions retain the vendor multiplication: KA and JACC each author two extraction entrypoints, with zero authored GEMM kernels. This is a CUDA vendor-library interoperability pipeline, not a pure portable GEMM translation.

## Scientific scope and checks

Five synthetic datasets cover9×17single pass and forced memory-budget multipass,17×33multipass,5×1allconstant and7×64single pass. Source remaining_B partitioning with54/160 available float slots yields row chunks3/4/2 and4/6/7. These are named small budgets, not the original4GB CLI threshold. Multipass computes fewer Gram cells with more launches than single pass, so their timings cannot be pooled as identical computed work.

Actual original C++ Float64 CPU preprocessing generates one common Float32 normalized-input fixture. An independent Decimal80 oracle checks normalization and computes centered raw-series Pearson coefficients. The maximum normalization error is1.47997e-8. Constant rows become zero by source policy; Pearson correlation for zero variance is mathematically undefined. Zero correlations for those rows are a convention, not a validated defined coefficient or brain-connectivity inference. Duplicate, opposite and offset-shifted rows supply analytic controls.

Fixed acceptance declared before GPU work is1e-6+3e-6relative for every computed matrix and packed value. Constant-row zeros, actual input bytes and guards are exact. Nonconstant diagonals must be1 and correlations bounded by1 within3e-6. Output payloads start as NaN outside timing, requiring every accepted SGEMM beta0/extraction value to be written finite. Full raw outputs remain without canonicalization. Each of20 artifacts has6276 Float64 words:2656exact prefix words,405complete computed Gram entries,239packed upper entries,1320actual normalized GPU input echoes and1656guard values. Leading64 floats preserve256B payload alignment. Single/multi same-input packed outputs and every field are independently audited.

These short synthetic numerical checks do not establish statistical significance, functional connectivity, physiological validity or causal brain relations. CPU preprocessing, full original host CLI and large fMRI throughput are outside the measured component.

## Vendor identity and execution

Every process uses the same realcuBLAS12.0.2 file `/usr/lib/x86_64-linux-gnu/libcublas.so.12.0.2.224`, SHA2566571bf8344cf13f3b116e6d613dd234cefca081c2303c3667e71b25d0eba5ec2, typedcublasSgemm_v2 and owned default-math0 handles. Native dladdr/version/math-mode logs and bound binary hashes match Julia dllist/SHA/version/mode logs. Julia checks HOST scalar pointer mode and calls an ABI-exact host-scalar FFI; it does not pass host Refs through the generated device-CuRef wrapper. Native's actual CPU scalar calls pass numerically; no unlogged native pointer-mode query is invented. Vendor accumulation, internal algorithm and kernel count remain opaque.

Each trial times nine SGEMM+completion intervals and nine extraction+completion intervals across five cases. Per-chunk multiplication/extraction times and per-case sums are retained. These18 stages are not a claim of18GPUkernels: each SGEMM API call may launch multiple vendor kernels. Extraction uses the original1024-thread block and exact native ker/ker2 packed indexing. Handles, CPU normalization, allocation, warmup, NaN resets, readback and CPU checks are outside. Five checked in-process trials retain all samples and average only2–5.

## Compiler and tooling history

Native first build rejected an owned RPATH option before compiling kernels; the corrected link adapter passes the first actual native GPU candidate. Julia had five program starts. Four fail in common host instrumentation/FFI setup before authored extraction compilation: math-mode Ref ABI, host Ref versus generated device-CuRef scalar ABI, missing PointerMode_v2 suffix, and pointer-mode Ref ABI. Their sources/logs remain preserved. One separate verifier preflight rejected a descriptive input-contract change; the description was restored and unchanged native captures reused. None is a framework device-kernel compile failure or unsupported feature.

CUDA.jl, KA and JACC each pass their first actual authored extraction candidate/check. Shared normalization/oracle/vendor adapter work belongs to the reference phase and is reused KA then JACC. Prospective token windows include repeated/cached context, instrumentation and coordination, overlap the outer workflow, and cannot be summed or treated as intrinsic framework/exclusive billed costs. Historical creation cost remains unknown. Root alone commits shared work and each portable version separately.

Fixture regeneration uses `make fixture` from this directory: it builds the preserved C++ normalization helper into ignored local logs, then runs the independent fixture generator. Existing measured binary fixtures and hashes remain the primary evidence; regeneration is not another measured series.

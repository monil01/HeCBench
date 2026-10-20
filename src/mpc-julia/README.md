# MPC: Massively Parallel Compression of scientific data

MPC here means **Massively Parallel Compression**, not model predictive control. It stores scientific double words losslessly through integer operations on their raw64bits. The source trace's physical variables and units are unknown. No end-to-end simulation accuracy or compression-optimality claim is made.

## Declared studies and original behavior

| Study | Scope | Native reference |
|---|---|---|
|Primary|First1,048,576 real trace words, dim1; plus the same1024-word xorshift stream at every dim1..32 (33cases)|Original capacity and unchanged GPU bodies; full chunks|
|Secondary|n1/63/64/65/1023/1025 × dim1/3/32 (18cases)|Explicit corrected host output capacity; GPU arithmetic unchanged|

Both keep1024threads/block and160blocks on the80SM V100. Source `__launch_bounds__(1024,2)` and matching Julia NVVM controls preserve the selected resource policy for its cross-block busy-wait offset ring. Resource queries permit two blocks/SM; these are maximum-residency calculations, not measured achieved occupancy or a formal preemption-safe progress proof. Every actual series completed.

The exact original DVC archive was recovered:275,049,163B, MD5`bb136e68786cde4485a255d6b9dd6a3c`, SHA256`835548af863ffa790febf034bc28fc6894e2e235286de9188d4cd41ed6127536`. Its raw trace contains36,263,232words/290,105,856B. The unchanged native full CLI actually compressed and decompressed that entire trace exactly (raw SHA256`bcf0b51289caf381dd13b15d9675a498c45e5b4d31fe554c23f2bb06b8dd5df3`). Those single executions are supplemental correctness controls, not four-way full-trace timings.

The original host capacity `n+1+ceil(n/64)` is insufficient for partial chunks. For the valid finite normal Float64 word `0xd58cf0f3f77ecbe4` (approximately-1.2964e104), n1 allocates three words but the source kernel emits30. Existing CUDA12.9 compute-sanitizer on the unchanged CUDA12.0 binary observed invalid global8B stores, exit99. Smaller logical-capacity guards also expose n63/64/65/1023 overruns. The system sanitizer first failed to find its injection library before application execution; that tool setup failure is separate.

The owned corrected CLI changes only the host capacity expression to `ceil(n/1024)*1024+1+ceil(n/64)`, a conservative complete-chunk payload bound, **not** minimum required memory. CRLF→LF normalization is disclosed; all GPU definitions before main match after line-ending normalization. Corrected n1 compression and decompression both passed memcheck with zero errors and exact original8B output. Original failures and all primary trials remain unchanged. The secondary has `native_reference_repaired:true` and a canonical hash-bound correction/witness provenance supplement. The full-chunk primary has `native_reference_repaired:false`.

Both studies use owned resident drivers with captured buffers and same-process header rewriting instead of separate source file CLIs, so both are `native_integration_adapted:true` and excluded from untouched-native aggregates. This is a named GPU codec study, not a rewritten original file-processing CLI.

## Numerical and mathematical evidence

- Primary:20captures ×34,941,456B; secondary:20 ×490,608B. All four versions match exactly, including actual initial GPU inputs, actual decoder-input buffers, complete encoded capacity, decoded words, both final offset arrays and64B guards. Source unused capacity is explicitly zero-initialized outside the ROI; it is not a claim about undefined unused bytes in the original CLI.
- Every actual decoded payload equals its actual initial device input bit-for-bit. A separately authored scalar source-format encoder and inverse mask-stream decoder check the complete format and roundtrip, without GPU warp/ring mechanics. The encoder implements the source method, not a different compressor; inverse decoding and direct actual-input identity supply independent mathematical controls. No check is replaced by output size or a checksum.
- A separate saved-file audit checked all708,641,280B and rehashed324unique execution bindings after all supplemental work. All40capture comparisons and exact guards passed. No overwritten timed intermediate snapshots are claimed.
- Source partial streams can contain an unneeded padding-transition payload word. Encoded-used and decoder-consumed counts are checked separately; exact source encoding is retained rather than silently minimizing the stream.
- These are integer/bit-pattern contracts, atol0/rtol0. Actual fixtures contain no NaN, infinity or signed-zero patterns; one xorshift subnormal pattern is present. See `input_bitpattern_inventory.json`. Do not claim exhaustive IEEE-edge conformance or reinterpret compressed words as floating numerical outputs.

## Implementation and performance

All three implementations contain genuine hand-translated Julia device computation: chunk-local dimension deltas,16bitmatrix transposes, adjacent-word differences, bitmap compaction, global offset ring, sparse inverse prefix sums and dimension-strided reconstruction. They do not delegate the codec to native kernels or load native PTX. Two **scalar** volatile-global PTX leaves preserve source polling/store semantics; original compiler PTX confirms those operations and launch bounds. This is backend-extension evidence, not measured memory traffic.

CUDA.jl uses its own kernel/index/warp/shared/barrier APIs. KA uses its own @kernel/@localmem/allocation/context and foreign CUDA device helpers. JACC uses its generated parallel-for wrapper, shared-memory and array APIs. Explicit CUDA backend launch adapters retain the source min-two-CTA compiler control; the pinned JACC public LaunchSpec has no such compiler keyword. These selected implementations are `hybrid_device_intrinsics`, not backend-independent ports or demonstrations of fundamental framework inability.

Native shared arrays are Int64 with an Int32 alias for count scans. Targets represent them with UInt32 word pairs in4098shared words (16,392B total), preserving modular bit arithmetic but changing access representation/placement. JACC requests dynamic shared memory and template preload/barrier; every actual template word is checked zero, and all algorithmically read positions are overwritten before use. CUDA.jl/KA use static shared memory. Common timed offset resets use the CUDA driver memset API versus the source CUDA runtime memset; no dummy observer kernels are inserted.

Five checked trials follow an in-process untimed warmup. Discard only timing1 and average2–5. The ROI is the sum of source compression offset-reset/kernel/device-completion and decompression offset-reset/kernel/device-completion regions. Header readback/rewrite, transfers, allocation, IO, oracle checks and captures are outside. Per-case individual times were not stored; these are mixed-case aggregate times.

| Mean milliseconds | Native | CUDA.jl | KA | JACC |
|---|---:|---:|---:|---:|
|Primary33cases|2.054655|2.311839|2.411620|5.623437|
|Corrected secondary18cases|0.596894|0.863954|0.934493|2.593867|

Both Julia environments have CUDA.jl6.2.1, runtime12.0.0 local, compiler12.9.41 artifact, Julia1.12.6/LLVM18.1.7; native NVCC is12.0.140. Fresh loaded-library resolution is retained separately from the uncollected per-primary-process DSO traces. Compiler, host-call, cache and shared-layout differences are covariates.

All target kernels report32registers and permit two blocks/SM. Compiled local allocations are32/40B (compress/decompress) in CUDA.jl,32/48B in KA and240/224B in JACC. These values are **per-thread local allocations**, not isolated register-spill counts or actual spill traffic. JACC is slower in these aggregate scopes, but no control isolates which implementation difference causes that gap. No achieved-occupancy, causal abstraction-overhead, bandwidth or floating roofline claim is made.

## Effort and observed tokens

Each target used one submitted source bundle and two declared configuration/JIT process evaluations; there were no target compiler, GPU or numerical repairs. Two kernel/callback definitions and nine shared helper functions were created in CUDA.jl; KA/JACC reuse those mathematical helpers with their own integration. Two configurations,51shape/dimension cases and five trials are not51candidates. Backend compiler subprocess/cache-hit counts remain unknown. Shared native diagnostics, fixture generation, metadata-helper errors and verification effort are recorded separately.

Observed target windows: CUDA.jl2,484,657tokens, KA1,756,468, JACC2,289,544. They include repeated/cached inherited context; cached and reasoning counters are subsets. CUDA.jl creates the shared harness. The KA window includes forward JACC design reasoning, and JACC's planning is therefore left-censored in its own window: both are excluded from per-target workflow calibration, with raw counters preserved. No exclusive billing or recovered historical consumption is claimed. Exact event/model identities and separate shared phase windows are in `observed_token_windows.json`.

## Reproduction and evidence limits

Essential assets regenerate; no binary is irreplaceable. Requirements are Python3 standard library, g++ C++17, exact original archive/raw trace, native CUDA12.0, the pinned Julia projects and the declared V100/GPU6 environment. Preserve the original source tree and MPC BSD notice. Do not add raw GPU captures, executables, large trace/fixture binaries or package caches to Git.

The successful independent regeneration compiled the unchanged scalar C++ generator into a new/tmp directory and ran it with that directory as cwd, reading the already verified raw trace. All308files/35,394,117B reproduced exactly; originals were rehashed unchanged. The separate full-network Python convenience recipe exceeded a180s whole-process limit at an unobserved stage, and its raw child output was not retained. That failure remains explicit. A proposed combined recipe edit/run was rejected by automatic review before execution; no recipe or frozen fixture changed. The source-inspected direct C++ temporary-directory alternative passed.

In a **clean checkout**, using the recovered raw trace whose size/hash are above:

```sh
g++ -O3 -std=c++17 src/mpc-julia/prepare_assets.cpp -o /tmp/mpc_assets
cd src/mpc-julia
/tmp/mpc_assets /path/to/verified/msg_sp.trace.out
```

This writes relative `fixtures/` and case files into the current directory. Never run it in a tree containing an active or frozen measured series; use a fresh checkout or temporary directory. Original retrieval URL/archive/member hashes are in `native_diagnostics/asset_retrieval.json` and `original_roundtrip.json`. The unchanged `prepare_assets.py` is an optional network convenience attempt, not the validated cold-reproduction path.

From the clean source root, create new configs without historical reuse:

```sh
python3 src/mpc-julia/prepare_cold_config.py --configuration primary --output /tmp/mpc_primary_cold.json
python3 src/mpc-julia/prepare_cold_config.py --configuration partial_corrected --output /tmp/mpc_partial_cold.json
PYTHONPATH=/tmp/pasc-julia-plot-tools python3 tools/verify_portable_julia.py --config /tmp/mpc_primary_cold.json
PYTHONPATH=/tmp/pasc-julia-plot-tools python3 tools/verify_portable_julia.py --config /tmp/mpc_partial_cold.json
```

The central runner owns the GPU6 lock; do not outerflock it. Pin/instantiate exact project dependencies before use. The private JACC CUDA wrapper dependency is relocated only after exact file-hash matching. Fresh results retain their new environment/source identity and do not replace the original series.

Read-only local audits require retained raw research files, which hashes alone cannot recover:

```sh
python3 src/mpc-julia/audit_saved_outputs.py .porting-state/portable_julia/runs/20261020T004745451491Z_mpc/result.json .porting-state/portable_julia/runs/20261020T004823417418Z_mpc/result.json
python3 src/mpc-julia/collect_compiled_resources.py
```

See source-bound traits/domain/scientific witness, canonical model manifests/status/calibration, failure events, correction adjudications and RQ1–5 observations. Root alone coordinates source commits.

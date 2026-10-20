# Final five-case hardware assessment review

Read-only review of saved source, diagnostic logs, per-model terminal assessments, and source-bound scientific intent. No GPU executions, edits to source/state/registry, package installations, or Git mutations. The companion `review.py` reproducibly verifies explicit path/SHA-256 bindings reachable from the listed JSON records; `review.json` retains each check and the reviewed document digests. This does not turn missing textual references or unknown historical attempts into zero.

## Result

- Five native prerequisites remain unavailable under the recorded V100 sm70 environment; all **15 CUDA.jl/KA/JACC terminal statuses** explicitly record `hardware_blocked`, `terminal_adjudication:true`, `repair_or_limitation_review_completed:true`, and `framework_unsupported:false`.
- **132/132 explicit evidence hash bindings match**, with no missing or mismatched files. This includes diagnostic source/log/review bindings and the declared cuBLASLt library binding where present.
- No case supplies a completed native mathematical operation suitable for the requested four-way certification. Do not count these as numerically verified translations, performance samples, or proven KA/JACC feature failures. Historical Julia markers/aliases remain unverified.
- No blocking contradiction found. Required wording distinctions and one resolved stale-export observation are below.

## Case-specific evidence and limits

| Case | Actual saved observation | Supported conclusion / resumption condition |
|---|---|---|
| blas-fp4gemm | Unchanged source builds using isolated CUDA 12.9.41, then exits through its explicit compute-capability >=10 guard on V100 sm70. Guard skip exits 0; no matmul executes. | Source-prescribed native execution path is unavailable here. Resume with authorized hardware/library supporting the declared FP4 block-scaled/BF16-output contract. This is **source-guard evidence**, not an observed cuBLASLt rejection after bypassing that guard. |
| blas-mxfp8gemm | Same isolated 12.9 toolchain resolves the older-header prerequisite; unchanged native builds, then its explicit >=10 architecture guard skips V100. | Same guard-limited strength. UE8M0 block scales/E4M3/BF16/FP32/TN contract remains unchanged. No scalar, FP16, or other arithmetic replacement was accepted. |
| blas-fp8gemm | Original E4M3/BF16/FP32-fast-accum/scalar-scale/TN descriptor and 4 MiB workspace: actual status 15 NOT_SUPPORTED for all six original shapes plus aligned 64-cube with isolated cuBLASLt 12.9; earlier 12.0 evidence retained. Saved 12.9 log reports returned_algorithms=0, with probe source initialized to -1. | No successful native heuristic result for the tested descriptors/environment; no actual GEMM. Do not generalize this to every possible library/GPU or every BF16 operation on V100. Resume with a GPU/library accepting and executing this exact contract. |
| blas-mxfp6gemm | CUDA 12.0 unchanged native skips toolkit prerequisite; 12.9 unchanged native skips architecture guard. Exact FP6E2M3/VEC32_UE8M0/BF16/FP32/TN/32 MiB descriptor-only queries bypass that guard and return status15 for all five original shapes plus 128-cube. **Count stays initialized -1.** Loaded cuBLASLt version120900/path/hash recorded. | No successful heuristic result; **-1 is an unchanged sentinel, not zero available algorithms or a negative algorithm count**. No GEMM executed. The existing limitation review already states this correctly. |
| twell | Original source explicitly requires sm90a WGMMA/TMA/cluster tensor-copy instructions. CUDA12.0 build fails source feature guard; 12.9 sm90a build succeeds. Actual legal M256,H2048,I8192 probe passes shape checks then fails its first attempted kernel with “no kernel image … device” on V100. 12.9 sm70 compilation fails explicit original minimum-architecture guard. | Strong actual image/ISA mismatch evidence, no completed source kernels or mathematical output. Resume on authorized sm90a hardware/compiler and establish complete native checks before target translation. A generic dense/scalar replacement would change this source-defined packed asynchronous path. |

The newer compiler was already tried in every case; no OS upgrade or privileged system-package mutation is implicated. The review does not claim to have exhausted every future toolchain or every alternative implementation. An arithmetic-emulation alternative would be a separate, explicitly scoped experiment rather than unchanged-native certification.

## TWELL domain: stale observation resolved

All five current source-bound views agree on `review_status:source_reviewed`, `primary_family:machine_learning`, sparse gated FFN/MLP inference, and the operation

`gate=relu(x*GATE^T); up=x*UP^T; out=(gate*up)*DOWN`.

Reviewed records:

- `src/twell-julia/domain_snippet.json`
- `src/twell-julia/implementation_traits.json`
- `.porting-state/portable_julia/traits/twell.json`
- `.porting-state/portable_julia/scientific_intent_300/twell.json`
- `ontology/studies/julia_abstractions_537/domains/twell.json`

Original main and matmul device source establish the operation; the limitation review establishes hardware mechanism separately. No mathematical or application inference-quality certification is claimed. The current exporter prefers reviewed local intent over provisional global metadata; its refreshed generated domain is already correct. **No annotation patch is needed.** If a paper snapshot still displays a provisional category, refresh that snapshot from the current registry rather than invent a new scientific domain.

## Proposal-only reporting guidance

1. Keep the five terminal cases separate from completed numerical/property/execution-only contracts, with 15 model assessments and zero accepted performance observations.
2. Report FP4/MXFP8 source architecture guards, FP8/MXFP6 unsuccessful exact-descriptor library queries, and TWELL actual incompatible image/ISA as distinct evidence strengths.
3. Prefer “no successful supported result for the tested native contract on the recorded environment” to universal “the framework/precision cannot work.” Keep FP6 sentinel wording explicit.
4. Preserve raw historical smoke programs and failed/skip diagnostics without counting them as successful implementations or compiler-feedback attempts at a new target.

## Usage accounting

This is one combined shared `study537` / `terminal_hardware_assessment_review` window; BEGIN `7b67eeca378e423ca0db29a5`. It includes context reuse, coordination, report construction, and evidence hashing, not five independently attributable benchmark translation costs. A separately captured END event is attached in `usage_receipt.json` after report completion. No retrospective per-case counter allocation is made.

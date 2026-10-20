# Logic rewrite: exact Boolean circuit resynthesis

## Current verified scope

Native CUDA, CUDA.jl, KernelAbstractions and JACC pass the complete original 15-pass `resyn2` computational pipeline on four structured AIGs with 4/6/9/12 primary inputs and 32 outputs. The original source-supported refactor cut parameter is K3. Every output is checked over all 16/64/512/4096 assignments after every stage. Every model has five checked trials, each with 169,762 exact Int32-encoded fields and 5,659 exact initial input fields. The saved native/Julia captures all have the same SHA256; a separate Python replay validates every graph edge, output truth function, fanout count and stage-to-stage state transfer.

The implementation is a qualified hybrid. All selected explicit source kernel launches execute authored Julia bodies: 50 distinct called Julia entry functions, 51 typed callback labels and 2,050 explicit kernel calls per four-case pipeline. Non-null Julia callbacks cannot fall back to native CUDA kernels. The original C++ CPU controller and native Thrust host operations, including their predicates, remain common library interoperation. This does not establish pure-framework or backend-independent portability. KA and JACC each have one real generic framework entry, specialized on the 50 shared mathematical bodies and 51 typed callback labels; these instantiations are not 50 separately authored framework entry definitions. JACC uses a scoped compiler extension for its own callable type to impose the source-compatible 128-register limit, retaining JACC compilation and launch machinery.

The named controls are not the original 256-input `hyp.aig` default CLI workload. That original CLI completed as an execution diagnostic; its full Boolean equivalence is not certified here. K12 refactoring on earlier bounded controls hit a preserved source row-pool assertion; K3 is an explicit source parameter, not a hidden native kernel repair. The original CUDA sources and their hashes remain in `reference_history` and `source_archive.json`.

## Reproduce the complete experiment

From the repository root, using the pinned Julia project, existing CUDA toolkit and unprivileged Python3/C++ tools:

```sh
python3 src/logic-rewrite-julia/aiger_controls.py --generate src/logic-rewrite-julia/controls --output /tmp/logic_controls_review.json
python3 src/logic-rewrite-julia/generate_gold.py
python3 src/logic-rewrite-julia/prepare_full_dispatch.py
python3 src/logic-rewrite-julia/make_config.py --output /tmp/logic_rewrite_study.json
python3 tools/verify_portable_julia.py --config /tmp/logic_rewrite_study.json
```

The central verifier builds the declared original and callback adapter sources in isolated directories and serializes GPU6 work with the shared lock. Do not wrap it in another lock. All includes, runtime inputs and actual Julia helpers are explicit bound dependencies. Code generation with libclang is optional: the generated Julia mathematical bodies are already present. No operating-system package installation is part of reproduction.

Expected regenerated fixture SHA256 values:

| File | SHA256 |
|---|---|
| `control_pi4.aig` | `9d7ce324cba3e96699b06c06222a2b60206c1648170582353637aa28be9b5b5a` |
| `control_pi6.aig` | `f00e2e45c7e9994846c4547ff1e11d368e21b1025aaf3b60d33ca79817574820` |
| `control_pi9.aig` | `c3982e676a54fade96b3731aacd07713a9d6321288ebdc8476808057fdd9e784` |
| `control_pi12.aig` | `de378b770cc28fabe7507f38ae72626f59234aeac7f039a5d7714896a5894c2f` |

For retained local captures, repeat the independent CPU gate with:

```sh
python3 src/logic-rewrite-julia/audit_captures.py --result RESULT_JSON --output /tmp/logic_full_accuracy.json
python3 src/logic-rewrite-julia/audit_fanouts.py --result RESULT_JSON --output /tmp/logic_full_fanouts.json
```

## Timing and mapping

Each model executes a full four-case warmup before five checked trials; report the mean of trials 2–5. The ROI sums 60 complete source optimization-stage calls, including original CPU control work, native Thrust operations and context completion. Explicit device staging, complete input/output observations, CPU Boolean gates and serialization are outside the ROI. These are not original CLI or isolated-kernel times.

Accepted means (seconds) are native 4.1628664985, CUDA.jl 4.15935645225, KA 4.162142544 and JACC 4.1585666155. All 20 complete captures are byte-identical. No profiler counters, roofline position or bottleneck attribution were measured.

Julia uses C-compatible packed records, raw device pointers and actual device malloc/free for automatic arrays and addressable cells. The source C++ automatic/local storage placement is different. The cooperative truth allocator uses CUDA warp shuffles and static shared pointers instead of CUB scan scratch. Device shuffle follows the installed 24-round Feistel/minstd_rand algorithm. Stable insertion sorting preserves source ordering but changes sorting complexity. Source geometry is retained. CUDA.jl and the scoped JACC compiler extension impose a 128-register cap; KA passes its static workgroup size as a compiler launch bound. Actual compiled register/spill counts were not profiled. These are implementation covariates, not evidence of identical instructions or resources. Source allocations are not claimed to have synthetic guards.

## Feedback, accuracy and effort evidence

Preserved source/unit revisions document the XOR-to-exponentiation semantic mistake, device type/address errors, register resource rejection, host LLVM callback compilation crash, and cross-translation-unit managed-variable misbinding. These are separate subsystem checks, not repeated complete benchmark translation candidates. The accepted complete CUDA.jl pipeline passed its first executed source candidate; one descriptor-only verifier preflight rejection occurred before execution.

The factor-unit raw gate rejects five bytes encoding scratch overflow-row labels. A separate CPU diagnostic follows every retained link: all 512 ordered synthesized graphs match, and all 4,096 Boolean values pass. That diagnostic does not relabel the rejected raw unit as a pass. The complete pipeline independently meets its original exact full-output contract.

The CUDA.jl prospective translation window recorded 44,602,447 input tokens (43,685,632 cached) and 212,167 output tokens (112,700 reasoning). Cached input and reasoning are subsets. This includes source-generator/ABI/unit-harness creation, compiler/runtime feedback, overlapping verification and limited read-only cross-benchmark coordination. It is not exclusive billing or intrinsic cold-start framework cost, and is excluded from simple-kernel calibration. Historical missing consumption remains unknown. `effort_summary.json`, per-model ontology, source-bound failure events and `paper_observations` retain the scopes and RQ evidence.

## Final evidence and ordered target costs

The all-model result is `.porting-state/portable_julia/runs/20261020T015900290225Z_logic-rewrite/result.json`. Each complete artifact has SHA256 `887f759efa970012f25bf068e50f40e8df4216249a85348873729e8cacabe8ad`. Independent replay covers 124 complete AIG states per capture, every output assignment, stage continuity and all 124,040 incoming-edge/output fanout counts across 20 captures. `all4_dispatch_audit.json` records actual runtime labels.

The KA translation window recorded 1,731,439 input tokens (1,673,984 cached), 9,893 output tokens (6,656 reasoning), total 1,741,332. JACC recorded 1,412,153 input tokens (1,379,328 cached), 8,370 output tokens (1,365 reasoning), total 1,420,523. Later versions reuse translated mathematical bodies and the full native/Julia harness. Their lower marginal counts do not estimate independent cold-start framework costs. Native preparation, annotation and overlapping verification windows are separate and must not be blindly summed.

# minimap2: bounded anchor-chain scoring

## Scientific contract

The native benchmark is one cooperative anchor-chain dynamic-programming kernel, not a complete minimap2 mapper. A read has packed UInt16 x/y/width/tag anchors. Each anchor propagates its score to the next64 anchors satisfying tag, distance and diagonal-band constraints. The gap penalty uses Float32 multiplication then Int32 truncation, with the source capped integer-log rule. Equal predecessor candidates choose the latest predecessor; an own-width tie resets the parent to -1. All barriers and this rule are preserved.

The original DVC archive was recovered from its exact MD5 object, 22,646,314B, MD5`1a9d6bde2cf5246bc7cc955d6c861169`. Its1001reads contain2,300,485anchors; all coordinates18..48183 fitUInt16. The unchanged native CUDA12.0 CLI actually ran and its complete20,281,306B output was byte-identical to the recovered expected output. This is a supplementary source-data control, not four-way default-dataset coverage.

The declared four-way configuration selects the first128 source-order reads with64..3072anchors, retaining all192624anchors. It keeps source1792PEs,64threads/block,1024iterations/tile and1089packedinputanchors/tile; three sequential batches exercise carried trackers. Many PE slots are inactive, exactly as the source scheduler representation allows. Larger/default read population and throughput scaling are unmeasured. The source EOF reader leaves max-distance/band fields unspecified while its scheduler copies them into global parameters. The actual original CLI passed; this static host issue is not an observed numerical failure. The owned resident study explicitly uses the dataset's5000/5000/500 parameters, without certifying arbitrary original EOF behavior. Original GPU bodies are unchanged, `native_reference_repaired:false`, `native_integration_adapted:true`, `exclude_untouched_native_aggregates:true`.

## Numerical and scientific evidence

Final result: `.porting-state/portable_julia/runs/20261020T002231392173Z_minimap2/result.json`.

- All20 independent timed-trial captures are byte-identical,91,837,056UInt8 values each. The first46,878,976bytes are actual initial device anchor/control readbacks with guards. Complete padded score/predecessor returns, both final continuation trackers and all64B guards follow. Source inputs are also checked unchanged after execution. No checksum replaces full outputs.
- An independently authored CPU **forward recurrence** computes all padded returns and carried trackers without reproducing cooperative shared-ring mutations/barriers. Every primary process checks the complete expected bytes after every trial.
- All385248selected logical scores/predecessors also match the recovered historical output. An independent actual-capture graph check validates192624real states,152930eligible earlier predecessor edges,39694seed nodes, exact path-score recurrence, and acyclicity. Longest observed parent path is2268nodes.
- `saved_output_audit.json` records a separate full1,836,741,120B audit and51rehashed source/asset bindings. No overwritten timed intermediate trace is claimed. This evidence establishes the selected bounded-anchor recurrence, not end-to-end genomic mapping quality, unrestricted chain optimality or biological accuracy.

## Execution and representation

Five checked executions/model follow an in-process untimed warmup. Mean2–5, seconds:

| Native CUDA | CUDA.jl | KA | JACC |
|---:|---:|---:|---:|
|0.002722223|0.005873624|0.006480016|0.00818056875|

The ROI contains three source-shaped batched launches plus whole-device completion; preparation, source CPU scheduling, allocation, captures, file IO and checks are outside. Source scheduler/descheduler is not rewritten as a Julia CLI. All frameworks author the actual device DP; none delegates it to native PTX/C ABI. Common host allocation/readback uses CUDA.CuArray outside the ROI. CUDA.jl uses its own index/shared/barrier APIs, KA uses @kernel/@localmem/@synchronize, JACC uses parallel_for/shared/sync_workgroup. No foreign CUDA device intrinsics are needed by KA/JACC.

Native shared arrays use separate Int32 score/parent arrays. The Julia variants pack each pair into UInt64 shared slots alongside64raw8Banchors, preserving1024logical shared bytes but changing layout/access instructions. Native short4 anchor loads versus UInt64 target loads may also lower differently. JACC's explicit1024B dynamic shared template adds a preload/barrier; the actual GPU template is checked zero and all algorithmic slots are overwritten before use. These source-backed differences, host APIs and compiler policies are covariates. No instruction/occupancy/traffic counters, causal abstraction-overhead or roofline claim is made.

CUDA.jl's first actual device-JIT check rejected inferred Int32 kernel return; exact source/log are in its candidate history. A trailing `nothing` repairs the void ABI. Two CUDA.jl programs/JIT processes, one mathematical body and one compiler-feedback repair were observed. KA and JACC each passed their first program/JIT. One native asset-staging failure occurred before GPU execution; it is separate from target compiler effort. Internal backend compiler invocation totals and historical effort remain unknown.

Observed target usage is1,151,304tokens CUDA.jl,670,784KA,823,318JACC, including repeated/cached inherited context. CUDA.jl creates the shared harness/helpers; KA/JACC reuse that verified mathematical body in order. Cached-input/reasoning counters are subsets, not additions. See exact identities/marker lag in `observed_token_windows.json` and canonical per-model calibration files. These are current activity windows, not exclusive billed costs or historical recovery. Shared readiness/verification/annotation windows remain separate.

## Reproduce

Dependencies: Python3 standard library, g++ supporting C++17, native CUDA12.0 NVCC, pinned Julia projects, network access to the exact declared DVC object. No Python package, OS update or privileged installation is required. Preserve the original native source tree and bundled source license.

```sh
python3 src/minimap2-julia/prepare_assets.py
python3 src/minimap2-julia/prepare_cold_config.py --output /tmp/minimap2_cold.json
PYTHONPATH=/tmp/pasc-julia-plot-tools python3 tools/verify_portable_julia.py --config /tmp/minimap2_cold.json
```

Do not wrap the central verifier in flock: it owns the GPU6 lock. Run fresh reproduction in a clean checkout/tree; never overwrite fixtures during an active measured series. `prepare_assets.py` is CPU-only: exact remote archive, deterministic selection, source CPU packing, independent forward DP. The seven essential local assets regenerate exactly (hashes/byte counts in `fixture_provenance.json`); no irreplaceable binary input exists. A fresh isolated/tmp cold tree actually reproduced every asset, while frozen originals were rehashed unchanged before/after (`cold_regeneration_audit.json`). Original source EOF parameter hazard remains explicitly outside general CLI certification.

An earlier proposed in-place regeneration/qualification command was rejected by automatic approval review before execution. No source or fixture mutation occurred. The safe isolated reproduction used unchanged recipes, not a workaround that overwrote measured evidence. Raw GPU artifacts remain local-only; do not add them or executables to Git. Preserve generator/recipes/proofs/hashes, not large generated binary/text data.

Saved scientific checks, no GPU:

```sh
python3 src/minimap2-julia/audit_saved_outputs.py .porting-state/portable_julia/runs/20261020T002231392173Z_minimap2/result.json
python3 src/minimap2-julia/audit_chaining_science.py .porting-state/portable_julia/runs/20261020T002231392173Z_minimap2/result.json
```

Raw logs/captures need the local research bundle; hashes alone cannot recover missing artifacts. Environment/project/source bindings, original-data retrieval, actual full-CLI geometry and RQ1–5 observations accompany this packet.

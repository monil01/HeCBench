# RSMT: warp-cooperative rectilinear Steiner-tree heuristic

## Scientific scope and inputs

The original benchmark computes a **heuristic** rectilinear Steiner tree for VLSI routing nets. This study preserves both original CUDA stages and the iterative Prim/median-Steiner algorithms, including deterministic distance/edge tie encoding, warp ballots, dynamic work assignment, and the source degree≤256 truncation. It does not certify global Steiner optimality, routed congestion, physical timing or signal integrity.

The exact original DVC object was recovered anonymously with its declared MD5 `8b70aae8ccc5aa050941b5afb6854b40`, 194,316,711 bytes and SHA256 `bd182c40615b4a9c4bc0b5bae19a86df3edac278b841e7c622a3264e0269dd08`. It contains 2,635,625 nets and 10,073,726 retained/truncated pins. Recovery does not establish historical execution: the older no-input deferral remains unchanged historical evidence.

The measured named fixtures contain **8,281 actual nets**: the first8,192 plus the first missing degree representatives, covering source degrees through256; and **19 independent controls** for 2/3-terminal identities, duplicates, collinearity, warp tails, and degrees32/33/255/256. These contain43,290 total pins and exercise both stages. The full2.6-million-net CLI is unmeasured; the complete raw capture footprint motivated the bounded sample.

## Numerical and memory evidence

An independent serial C++ integer Prim/median-Steiner implementation includes no CUDA, original implementation or Julia code. It derives every output coordinate, edge and offset. A separate mathematical witness verifies terminal preservation, edge bounds, connected acyclic trees, wirelength≤terminal-only Manhattan MST, and exact 2-/3-terminal optimum identities. There is no larger-net global optimum claim. Total resulting wirelength is5,441,748 on actual nets versus terminal MST5,614,598; controls307,941 versus337,749.

Each model executes five checked trials after a complete untimed warmup. Every trial retains451,734 Int32 primary fields, including94,886 exact input-prefix fields, complete output arrays, the complete canonical live worklist and actual guard bytes. Every physical parent buffer is retained separately. Worklist order is intentionally nondeterministic: the primary contract sorts its complete live IDs, proves the exact bijection and retains the original ordering and unchanged unused suffix in raw storage. This is complete numerical semantic agreement; **raw_output_identity_claim:false** does not claim literal worklist-slot identity.

Native has16 guarded allocations across the two cases (8,192 actual guard bytes/trial). Julia adds two separately guarded3-word device-state buffers in place of native globals, giving18 allocations and9,216 bytes/trial. `audit_complete_capture.py` rehashes all5 primary files and joins each physical buffer to the actual input/output/canonical contract. Actual source kernels are preserved byte-for-byte in `original_native/`; only `main` is renamed when included by the harness.

The first native series passed all numerical checks and retained primary guard bytes, but wrote full raw worklist storage only in the temporary working directory. Its missing raw storage remains explicitly missing. A persistent-path evidence repair generated a new five-trial native series; there was no GPU-math, input, oracle or acceptance change. Both source variants, old receipts and the retention-gap review remain preserved.

## Mapping and timing qualifications

CUDA.jl authors two device entry kernels using its own CUDA.jl shared memory, atomics, warp synchronization, shuffles, ballots and vote APIs. KA and JACC author their own two entry kernels and invoke the same source-faithful CUDA-backend helper body; they are **hybrid_device_intrinsics**, with no hidden `@cuda` launch. This establishes the recorded CUDA-backend execution, not cross-vendor support or a general framework limitation.

All models preserve the source interleaved4-byte `(Int16 src,Int16 dst)` edge layout, warp-cooperative work, 2×SM block count and768/96 threads/block. The source `__launch_bounds__(threads,2)` compiler hint is not encoded in the Julia entries; no register-count or occupancy identity is claimed. Statically declared helper shared storage is43,104 bytes for the768-thread kernel and43,020 bytes for the96-thread kernel; these are source-declared totals, not profiler observations. JACC explicitly reserves0 dynamic shared bytes because the helper already uses static storage.

The ROI sums both cases' three source output resets, two original algorithm stages and context completion. Allocation, input transfers, device-global reset, full captures and the independent oracle are outside. Native uses runtime `cudaMemset`, Julia uses driver byte-memset on the task stream. This host API/compiler-control difference is a covariate; no causal abstraction, launch-bound or roofline claim follows from the measured time difference. Five timings are retained; only the mean of trials2–5 is reported. The bounded scope is excluded from untouched full-CLI aggregates.

## Reproduction

Use the existing pinned Julia KA/JACC environments, CUDA12 native compiler, a compatible GPU and the shared study lock through `tools/verify_portable_julia.py`. No system package installation is required.

```
python3 src/rsmt-julia/prepare_fixtures.py --source PATH_TO_EXACT_DVC_OBJECT --output-dir src/rsmt-julia/data
g++ -O2 -std=c++17 src/rsmt-julia/independent_oracle.cpp -o /tmp/rsmt_oracle
/tmp/rsmt_oracle src/rsmt-julia/data/original_first8192_plus_degree_coverage.bin src/rsmt-julia/data/original_first8192_plus_degree_coverage_oracle.bin
/tmp/rsmt_oracle src/rsmt-julia/data/degree_tail_duplicate_collinear_controls.bin src/rsmt-julia/data/degree_tail_duplicate_collinear_controls_oracle.bin
python3 tools/verify_portable_julia.py --config src/rsmt-julia/study_config_native.json
```

The subsequent CUDA.jl/KA/JACC configs reuse exact hash-bound references. Update reuse paths to the freshly obtained receipts when replaying elsewhere; do not invent old receipts. The four runtime fixture/oracle binaries total1,798,744 bytes and are listed in `essential_assets_for_commit.json`. The194MB original object, raw run files, executables and `.so` libraries are not required Git assets. Prospective translation/native/verification/annotation windows record actual context-inclusive activity counters; verification overlaps translation and must not be added again. They are not billed/exclusive coding costs or reconstructed historical consumption.

# Lanczos: full sparse recurrence and eigenvalue filtering

## Scope and scientific checks

This study retains the original CSR Lanczos recurrence, both GPU reductions, both vector updates, the host sums/scalars, and the original CPU TQLRAT/spurious-eigenvalue filter. All four scientific GPU kernel definitions are translated in each Julia version. The original CUDA sources and Makefile are preserved in `reference_history/`; `source_adapter_provenance.json` records the byte-identical native kernel prefix and the launch/observation hooks.

The named inputs are five connected257-node unit undirected graphs: three complete cliques followed by a connecting path covering the remaining vertices. Clique sizes are(6,5,4), (17,13,9), (25,19,13), (37,29,23), and(53,43,31). They exercise the source's complete subgroup-size selection2/4/8/16/32, irregular degrees and boundary rows. Each run performs71 source iterations, requests three largest eigenvalues and starts from the original e0 vector. Iteration71 is one of the original CLI's sweep points. Precision is Float64 with Int32 CSR; Float32 is not certified by this configuration.

All20 complete trial artifacts are byte-identical:3,103,860 Float64 fields per trial, with39,099 exact initial-input fields. Each trial executes2,125 scientific kernel calls and365,090 independent output/spectral checks. Every actual GPU input and complete stage output is captured, including the CPU scalar/alpha/beta chain and actual allocation guards. Native/current variants retain the original C++ CPU controller and guarded native CUDA allocations through FFI; the scientific GPU kernels are genuinely Julia code, with no native GPU fallback on the selected callback path.

Independent checks include full extended-arithmetic SpMV, dot, scale and axpy outputs (budget1e-10 absolute plus1e-10 relative), normalized recurrence inputs, and all three returned eigenvalues against a separately solved complete dense symmetric graph problem (budget1e-4 absolute plus1e-6 relative). Direct cross-version budgets are1e-8 absolute plus1e-8 relative; the observed differences are exactly zero. The maximum independent stage error is8.15e-14 and maximum top-three eigenvalue error is1.76e-5.

These are engineering accuracy bounds for this fixed-step naive method, not a general convergence or eigenvector-orthogonality guarantee. The independent audit measures recurrence residuals below3e-15, while basis off-diagonal inner products reach0.71–0.97: the original algorithm does not reorthogonalize. Its eigenvalue filter and finite-precision behavior are preserved. This observation is not hidden by source agreement and is not labeled a framework failure.

The original DVC graph archive was recovered with exact size/MD5 and the unchanged full800000-node CLI completed all12 GPU iteration counts7 through183 and CPU comparisons7/23. Those common-step outputs agree approximately1e-11. That diagnostic is not a five-trial four-way accuracy/performance certification of the original full-size graph.

## Mapping and timing

Native groups and256-thread blocks are retained. CUDA.jl uses CUDA shared-memory/barrier primitives; KA uses framework local memory/barriers; JACC uses framework dispatch plus explicit CUDA static shared memory and block barriers for the two reductions. JACC launch specs bind complete grids and request zero dynamic shared memory. This is a disclosed CUDA backend extension, not proof that a portable JACC alternative is unsupported.

The native SpMV uses volatile shared-memory warp-group reduction with implicit warp progress. Its original body is unchanged and passed all named checks. Julia versions give that logical reduction explicit whole-block synchronization; this is a mapping/synchronization adaptation, not an observed native defect. Actual compiled resources, occupancy and bottlenecks were not profiled.

A complete five-case warmup precedes five checked trials; only timing1 is discarded and timings2–5 are averaged. The region includes the complete `gpu_lanczos_eigen` call: native allocation/transfers, recurrence, host reductions, TQLRAT and filtering. Each mathematical kernel has common whole-context completion **included** before observation timing begins. Actual readbacks, independent checks, guard checks and serialization are clocked and excluded. Guard construction and framework launch/FFI overhead remain included. Initial graph construction is outside the region. This instrumented synchronous pipeline is not the original uninterrupted asynchronous CLI timer or a kernel-only measurement; no roofline or causal overhead conclusion is claimed. Per-case times were not retained separately; reported time is the complete five-case sum.

## Reproduction

The recorded environment is Julia1.12.6, CUDA.jl6.2.1, KA0.9.45, JACC1.4.0, native CUDA12.0.140 and physical GPU6 V100PCIe32GB. Exact pinned project/Manifest/preferences hashes and analysis precision are in `environment_provenance.json`. Named graph inputs are generated at runtime by `graph_controls.h`; no binary fixture or graph download is required.

From the repository root on a compatible configured machine:

```sh
python3 src/lanczos-julia/make_config.py --output /tmp/lanczos_config.json
python3 tools/verify_portable_julia.py --config /tmp/lanczos_config.json
```

The config producer supports `--source-root` and `--julia`, drops all old-capture reuse and binds fresh all-four measurements. The central verifier owns GPU6 locking; do not wrap it in an outer lock. `regeneration_contract.json` gives the expected initial-input prefix SHA256. Source math, original input ordering and complete outputs remain bound to recorded files.

The CPU-only independent saved-evidence audit requires NumPy2.2.6:

```sh
OPENBLAS_NUM_THREADS=1 python3 src/lanczos-julia/audit_captures.py \
  --result .porting-state/portable_julia/runs/20261019T235806629045Z_lanczos/result.json \
  --output /tmp/lanczos_audit.json
```

`--run-root` relocates archived run folders. Every file is fully hashed. Mathematical replay may be reused only after identical complete SHA256; in this case one full replay covers20 byte-identical captures. The in-process independent checks were executed separately in every trial. The replay independently reconstructs the graphs/CSR/input bits, every GPU operation, all recurrence chains, both the graph and tridiagonal spectra,17,920 actual GPU guard bytes, and the five-trial/last-four means. Negative CPU controls reject a changed result, CSR column and guard byte. Extended CPU arithmetic here has a64-bit mantissa; no arbitrary-precision oracle is claimed.

For optional original800k CLI diagnostic reproduction only:

```sh
python3 src/lanczos-julia/regenerate.py --output-dir /tmp/lanczos_original_asset
```

This fetches only the exact public DVC object, checks23,191,260 bytes and MD5 `2b9896a988b45223138a88317fa4a7e3`, and safely extracts bounded regular files. The62MB graph and trial outputs remain outside source Git. Their exact hashes/locations are recorded; hashes are not substitutes for absent raw data. No OS/package mutation or privilege escalation is required or performed.

## Feedback and effort

CUDA.jl, KA and JACC each passed their first complete scientific program and device-JIT process; each defines four GPU kernels. Logical kernel definitions, complete-program candidates, device-JIT processes and five checked trials are counted separately. There were no compiler, runtime or numerical translation repairs. One native pre-execution observer revision retained warmup diagnostics; it did not change kernel math and is not a GPU translation iteration. Unused-variable compiler warnings are not failed attempts.

Prospective native, three target, independent verification and annotation usage windows record actual input/cached-input/output/reasoning counters and observed model context. Cached input and reasoning are subsets of input and output. Native preparation is excluded from target windows. CUDA.jl creates the shared Julia FFI harness; KA/JACC reuse it and accepted source context. Costs describe the ordered marginal workflow, with inherited context and coordination, not intrinsic cold-start framework costs or historical reconstruction.

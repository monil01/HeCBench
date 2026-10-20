# miniDGS: full communicated Maxwell step, degree 6

## Scope and source qualification

The named configuration runs the original default 1,022-tetrahedron mesh on two MPI ranks (511 elements each), with six electromagnetic fields, 84 valid nodal points per element, 28 points on each of four faces, and five low-storage RK stages. The original run overwrites its computed CFL timestep with .001, then clamps it to `FinalTime=.0005`, giving one complete source time step. These are simulation time units. There are 515,088 valid field DOFs and 588,672 padded state floats. Other orders, meshes, rank counts, long trajectories and physical accuracy are unmeasured.

Both source ranks select logical GPU0. With `CUDA_VISIBLE_DEVICES=6`, both share physical V100 GPU6, exactly the source device-selection policy. This is a two-context MPI-sharing stratum, excluded from untouched-native and ordinary single-context GPU aggregate rankings. It is not a two-GPU result.

The immutable original CUDA source fails CUDA12.0 frontend compilation at removed legacy `texture<T>`/`cudaBindTexture` APIs. The owned compatibility recipe replaces those references with element-mode linear texture objects, preserving channel types, extents, indices and device arithmetic. The qualified full source CLI actually ran on the complete mesh. Its 60 actual stage/halo fields are bit-identical to the owned native harness warmup. This is a compile/API/integration qualification, not a scientific-arithmetic correction or an execution of the unchanged legacy CLI. See the exact patch, extraction hashes and `native_readiness.json`.

Targets use ordinary global arrays in place of native texture fetches. All four source device stages are genuine: outgoing-halo gather, volume curls, surface numerical flux/lifting, and RK residual/state update. Each stage exchanges actual prior GPU-computed fields through nonblocking MPI. CPU mesh loading/partitioning/packing is regenerated from source; its CLI frontend is not rewritten in Julia.

## Implementation and precision

The whole region is `mixed_kernel_communication_pipeline`: four framework device kernels plus real host MPI runtime interoperation. CUDA.jl, KA and JACC use their own indexing, local memory, barriers, allocation and launches. Their device component is `framework_device_kernel`, with no foreign CUDA device intrinsics. A shared CPU-only C ABI performs source two-rank Isend/Irecv/Waitall, including original per-call status allocation/free. Host Float32 buffers are used; CUDA-aware MPI is unnecessary.

Storage is Float32 with Int32 gather indices. Derivatives, lifting and RK use Float32; source Float32 jumps feed Float64 surface-flux intermediates that are cast back to Float32. The CPU oracle and artifact Float64 encoding are separate roles. Source GPU `--use_fast_math`, explicit Julia muladd, texture/global loads, compiler defaults and emitted instruction mix differ or remain unmeasured. No measured roofline position or pure frontend causal effect is claimed.

Element layout preserves six field-major arrays of 96 nodes, with 12 padding nodes each. Volume/surface shared requests are 2,352/2,688 bytes; targets pack the source volume arrays into one workspace. JACC requests zero-template preload plus a barrier; actual zero-template GPU readbacks are recorded separately, and all subsequently read workspace locations are overwritten. Nonshared JACC launches request zero shared memory. Emitted preload traffic was not measured.

## Numerical and scientific checks

Final receipt: `.porting-state/portable_julia/runs/20261019T235356970423Z_miniDGS/result.json`.

Every one of 20 primary captures contains 6,633,872 Float64-encoded values (53,070,976 bytes). The first 3,316,936 encode exact actual device inputs and guards. Full final Q/rhs/res, gather scratch, incoming halo, operators, maps and guards follow. Float32 values are encoded exactly; Int32 indices are exact. All initial prefixes, read-only inputs and allocation guards match byte-for-byte. All three Julia output arrays are bit-identical; native differences pass the predeclared field contract.

The central `atol=.002, rtol=2e-6` envelope is insufficient alone. The mandatory full saved-array audit checks every model pair and all five trials: Q/res/gather/received use `3e-5 + 2e-6*min(abs(a),abs(b))`; RHS uses `.002 + 2e-6*min(abs(a),abs(b))`. Constants, guards and input prefixes require exact serialized bytes. These budgets were chosen from native-only independent diagnostics before any target draft. All 30 pair-by-trial checks pass.

An independently expressed Float64 matrix-curl/linear-flux/RK evaluator propagates conservative Float32 gamma/absolute-operator error bounds through all five stages. Every model checks full final fields against per-value bounds. Separate actual untimed warmup traces check all volume/surface/RK fields: 160 full fields, 47,093,760 values. They also check 40 actual gather/peer-receive arrays against previous actual GPU states. These traces are not described as overwritten intermediates surviving the timed executions.

Independent geometry checks validate analytic Cartesian linear gradients, normal lengths, positive surface scales, field ordering and physical coordinate association for all 4,256 communicated face-node instances. They do not establish mass/energy conservation, temporal convergence, global solution accuracy or stability of the retained timestep override.

## Performance and measurement repair

All models complete a warmup and five checked trials. Report the last four mean, retaining every sample. The ROI includes halo gather/D2H, source nonblocking MPI, volume computation, MPI waits/ghost H2D, surface/RK and completion for all five stages; report max-rank duration. Setup, input readbacks, independent checks and artifact storage are outside the aligned ROI.

The first source-mirroring timer started before an initial MPI barrier. Peer outside-ROI preparation could therefore enter its timed wait. Both original reference series and exact source snapshots remain retained and hash-bound excluded. The prospective v2 contract adds an untimed MPI alignment barrier after setup/completion, then retains the original timed barrier and pipeline. Both references received fresh five-trial series. The criterion, device mathematics and inputs did not change; no speed selection or GC attribution is made. The actual GC contribution is unknown. Ordinary in-ROI allocation/GC costs remain included.

Final means: native 2.842538 ms; CUDA.jl 2.018706 ms; KA 2.209310 ms; JACC 2.903131 ms. Native texture/global mapping, host orchestration, allocator/view/GC policy, explicit context fences, MPI sharing and JACC shared requests are covariates. These results describe the named communicated pipeline.

## Cold reproduction

Required existing dependencies: CUDA12.0 toolkit/headers, Julia1.12.6 with the pinned project manifests/preferences, OpenMPI4.1.6 compilers/runtime, make, and NumPy2.5.4. CPU mesh generation uses the source CUDA preprocessor branch for padded map indices and needs SDK declarations; it links no CUDA runtime. All 114 generated assets (220,063,554 bytes) reproduce exactly without saved GPU outputs or visible GPUs.

```sh
PYTHONPATH=/tmp/pasc-julia-plot-tools python3 src/miniDGS-julia/cold_regenerate.py --output src/miniDGS-julia/fixtures
python3 src/miniDGS-julia/prepare_cold_config.py --output /tmp/miniDGS-fresh.json
PYTHONPATH=/tmp/pasc-julia-plot-tools python3 tools/verify_portable_julia.py miniDGS --config /tmp/miniDGS-fresh.json
```

Run generation before preparing fresh measurements. The helper rebuilds only the CPU/MPI C ABI and removes historical series reuse. The central runner acquires the GPU6 lock internally; do not outer-flock it. Large CPU fixtures, `.so` files, executables and raw GPU captures stay local. Commit source recipes, hashes, the source license and proofs. There is no irreplaceable local binary input.

Two complete CPU cold checks passed. The first successful build's captured stdout text was not retained beyond hashes; its report explicitly preserves that gap. The second is a new observed reproduction with actual command logs, not recovery of the earlier text. Full actual GPU rank/stage logs and outputs are retained under the receipt's auxiliary paths. Per-process runtime-DSO traces were not collected; project manifests/local-toolkit preferences and exact MPI C ABI/DSO provenance are recorded with that limit.

## Feedback and tokens

CUDA.jl used two whole two-rank program candidates: both first ranks hit a host CUDA.@cuda name-binding error before authored device JIT; `map` shadowed an unqualified macro helper. Renaming it fixed the API integration. A static Base.all shadow was corrected in the same edit; it was not a second observed failure. The successful translation reached authored JIT in two rank processes. The later shared timing repair added two fresh Julia JIT processes and is costed separately. KA and JACC each passed their first whole program with two rank JIT processes. Each version has one four-kernel source bundle, with reviewed mathematical-body and host-harness reuse.

Actual context-inclusive translation counters are CUDA.jl 3,339,276, KA 1,587,992, JACC 1,648,463 total tokens. Cached input and reasoning are subsets. CUDA.jl's window closed at numerical success with measurement repair pending and is calibration-ineligible; its actual counter remains. Native preparation and shared timing repair have separate windows (8,261,085 and 1,279,173). Saved verification and annotation are separate. Counters are neither exclusive billed translation cost nor recovered historical consumption. Backend compiler invocation totals remain unknown.

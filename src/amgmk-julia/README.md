# AMGmk: original CPU/GPU operators, nonunique relaxation states

**This is property equivalence, not identical GPU numerical states.** All three source operators are retained: serial CPU CSR Matvec, GPU in-place relaxation and serial CPU Axpy. Only relaxation is a source GPU kernel. The whole measured implementation is hybrid_host_control because actual CPU numerical work is timed; its GPU component is framework_device_kernel in CUDA.jl, KA and JACC. Normal host launch/completion alone is not the reason for that classification.

Original CUDA/HYPRE code is immutable. Native compiles the original source helper translation units and includes the original relaxation body unchanged. Julia implements the source caller's active default CPU branches (Matvec alpha1/beta0/singlevector/fullrows and Axpy alpha.5); other generic HYPRE API modes are not claimed. No Jacobi, coloring, atomics or alternate relaxation algorithm is substituted. Source attribution/disclaimer is retained in LICENSE.source and the original tree.

## Source semantics and scientific contract

The source builds the seven-point operator A=6I-grid_adjacency on a50×50×50 unit grid, with absent exterior neighbors omitted (zero exterior). It uses N125000,E860000,R500 and256threads/489blocks. This is an AMG operator microbenchmark, not a full multigrid hierarchy, V-cycle or application solver.

Original CPU Matvec/Axpy and GPU relaxation start at simple constant inputs. The original full CLI actually ran, returned0 and printed no errors. Its GPU allones case is an exact fixed point with f=A*1 and is only supplemental accuracy evidence. A separate unchanged-kernel one-sweep diagnostic, with identical perturbed input/geometry16 times, produced16 distinct full states (maximum difference from first0.09505208). Source global non-atomic neighbor reads/writes have no within-sweep ordering. Formal CUDA memory-model/data-race validity is **unproven**; observed convergence is not a formal proof of safety or a unique500-sweep trajectory.

Primary inputs are meaningful nonconstant fields at the original size/count:

- CPU Matvec: an affine dyadic coordinate field; full output equals an independent coordinate seven-point stencil, including truncated boundaries. Interior response is zero; boundary response follows the source zero-exterior policy.
- CPU Axpy: nonconstant dyadic x/y; full500-call output equals y0+250*x exactly.
- GPU relaxation: u0=1+0.25*w with positive first Dirichlet sine-mode weights w. Every actual final point is checked for finite/range/stability, weighted convergence and an independently computed coordinate-operator residual. All raw states remain preserved.

### Predeclared FP64 bound

Source compute/storage is Float64 (`double`), not Float32. With u in[0,2], integer f=6-degree, off-diagonal−1 and positive additions/division, IEEE monotonicity preserves[0,2]. Multiplication by−1 is exact. At most six additions with intermediates≤12 and division by6 give a conservative local absolute rounding error δ≤2^-50+2^-52. This covers fused multiply-add as well.

For the actual stored positive weights, q_bound exceeds max_i(sum_neighbor(w)/(6*w_i)) by1e-12; min(w)=0.0002333001463. Let M=max(abs(u-1)/w), η=δ/min(w). If M≥η/(1-q), each update obeys qM+η≤M, and a completed sweep obeys M_next≤qM+η. Thus after500 sweeps M≤q^500*M0+η/(1-q), with an additional1e-10 evaluation margin and1e-12 initial-ratio margin. The predeclared weighted bound is0.09675914356697, and the analytic weighted roundoff allowance is2.609014405e-9. It rejects a no-op (initial weighted norm≈0.25). The bound assumes coherent finite IEEE neighbor reads; the original formal race caveat remains.

The complete independent residual uses the coordinate grid, not CSR traversal. A conservative weighted residual bound follows from6(1+q)*M. All20 states satisfy the same criteria; their actual errors/residuals and source hashes are retained. This contract permits different valid source-schedule states, not just artificial equal arrays. It does not establish equivalence to a serial Gauss-Seidel ordering, Jacobi, or all possible schedules/inputs.

## Complete evidence

Each trial's score artifact has8,880,606 Float64-encoded values, including8,630,502 exact input-prefix values (69,044,016 bytes). The prefix contains actual before/after host CSR, device CSR and RHS readbacks, actual initial GPU state, actual CPU inputs/initial outputs, weights and scalar echoes. Int32 indices are losslessly encoded as Float64 numeric values. Full CPU outputs, actual GPU guards and eight independently evaluated property flags follow the prefix.

**Score-artifact identity is not raw GPU-state identity.** Full125032-value guarded GPU states are separately retained in `.porting-state/porting_logs/amgmk-julia-shared/raw/<model>/`, with unique process/trial filenames; all20 payload states are distinct. Each model also captures a separate full original-equilibrium control before trials. `saved_output_audit.json` hashes and checks all actual raw states and every score/input value, recomputes full coordinate residuals and weighted bounds independently, verifies exact host outputs and31 source bindings. Reused mathematical input checks require a fresh full prefix hash on every capture. GPU output/property checks are performed for every actual trial.

One supplemental CPU audit completed its checks but failed to serialize a NumPy boolean. The original script/log were preserved, the scalar conversion was corrected, and only the CPU audit was repeated. This is shared verification tooling effort, not a target candidate or additional GPU timing series.

Actual GPU states/captures remain local and are not Git assets. Hashes cannot recover unavailable captures. A fresh reproduction may produce different raw GPU states under the declared property contract; it must still pass complete checks.

## CPU assets and cold measurement

Python3.12.3 standard library only:

```sh
python3 src/amgmk-julia/generate_assets.py /tmp/amgmk-regenerated
```

The generator produces11 CPU binary assets, study_params.h and study_plan.json. Actual isolated cold regeneration reproduced all13 files exactly. Compare generation_reproduction.json/study_plan.json hashes before copying regenerated assets into missing locations. The original native graph builder was also checked byte-for-byte against these independent coordinate CSR assets. Essential binaries are listed explicitly in ready_support_files.json; actual GPU captures must not be added to Git.

Fresh full measurement from the source repository root:

```sh
python3 src/amgmk-julia/prepare_cold_config.py --output /tmp/amgmk-cold-config.json
PYTHONPATH=/tmp/pasc-julia-plot-tools python3 tools/verify_portable_julia.py amgmk --config /tmp/amgmk-cold-config.json
```

The config helper relocates paths and removes saved-series reuse. The verifier builds native code in isolation with explicit assets and serializes physicalGPU6 internally; **do not wrap it in an outer flock**. NumPy2.5.4 enables bounded comparator acceleration for the ~71MB artifacts. CPU generator needs no NumPy. Julia projects `_julia_ka_env` and `_julia_jacc_env` are unchanged from the actual NN environment snapshot: Julia1.12.6/CUDA.jl6.2.1/KA0.9.45/JACC1.4.0/localCUDA12.0, V100sm70; native nvcc12.0.140/O3/stdc++17/sm70, OpenMP disabled.

Each model warms the exact source CPU/GPU batch loops, then executes five checked trials; discard only trial1 timing and average2–5. Time is the **sum of three separate elapsed regions**, CPU Matvec500 calls, GPU in-place relaxation500 launches plus whole-device completion, and CPU Axpy500 calls. Per-operator timing is preserved in every row. Inputs/reset/copies/observers/checks/serialization lie outside ROI. Native generic HYPRE CPU APIs and Julia specialized CPU code differ; no CPU affinity/exclusive CPU reservation is imposed. All three Julia frontends share identical CPU code, so CPU timing variation cannot be attributed to KA/JACC.

The GPU component uses ordinary global array math and each framework's own index API, without foreign CUDA device intrinsics, shared tiles or atomics. Native legacy-default-stream and Julia task-stream launches share whole-device completion boundaries. Register use, occupancy, cache behavior and roofline coordinates are unmeasured. CPU-dominated combined timing is not a GPU abstraction comparison or a causal frontend ranking; retain per-operator times and nonunique convergence properties.

CPU-only saved-evidence recheck while original captures remain available:

```sh
PYTHONPATH=/tmp/pasc-julia-plot-tools python3 src/amgmk-julia/audit_saved_outputs.py
```

See paper_observations.json for RQ1–5, model-specific traits/status and source/attempt manifests. Each target passed candidate1/device-JIT-process-check1: one algorithm GPU definition, two CPU operator definitions and one untimed GPU observer. CUDA.jl creates shared helpers/harness; KA then JACC reuse them. Actual token windows include inherited context/cache and coordination, with native preparation/verification/annotation separate; cached/reasoning counts are subsets, not additive exclusive billed cost or historical reconstruction.

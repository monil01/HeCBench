# miniFE: qualified full finite-element/CG pipeline

This study runs the original serial CPU Hex8/Q1 assembly and GPU conjugate-gradient pipeline for 2³, 16³ and the original 128³-cell mesh. The largest matrix has 2,146,689 nodes and 57,066,625 CSR entries. GPU bodies are authored Julia; the original CPU controller, allocations and assembly remain through a C ABI. KA uses its actual kernel/shared-memory/barrier APIs; JACC uses actual `parallel_for` with CUDA static-shared-memory/barrier extensions. This is a qualified CPU/controller hybrid, not a claim of backend-independent execution.

## Reference qualification and accuracy

`reference_history` preserves original sources, hashes in `source_archive.json`. The original hot and cold boundary sets overlap; eliminating cold columns first drops subsequent hot-edge contributions. `boundary_witness.cpp/json` demonstrates the error on all 27 nodes of the 2³-cell case. The owned configured copy removes hot nodes from the cold set before elimination. All eleven original CUDA kernel bodies remain byte-identical (`native_kernel_body_identity.json`). This is a **corrected-native reference**, excluded from untouched-native performance aggregates.

The source load is unit Poisson forcing, while its analytic center checker describes the homogeneous part and uses a loose center threshold. That checker is retained only as auxiliary, including its false result on the corrected 2³ case. Acceptance uses independent full tensor-Q1 matrix/RHS checks, actual initial CSR/vector bit identity, full final x/r/p/Ap, and long-double true residual. The independent operator retains source Gauss coordinate `0.577350269`. The predeclared budgets are in `verification_plan.md` and the configuration. The largest case reaches the original 200-iteration cap: true normalized residual 2.65209e-8 satisfies the engineering 1e-7 budget; it does not meet the source recursive epsilon stopping goal. No full continuum error bound is claimed.

Each trial stores 15,061,421 Float64 values (full initial b/x prefix and final vectors/residuals); full actual CSR and five initial vector readbacks are retained separately and byte-checked every model/trial. Every actual 256-value post-final-reduction buffer is also saved and compared against both references. Four unused vector-update specializations are not certified by the measured pipeline. The seven exercised source operations are recorded in the dispatch counts.

The timed region is full source CLI work over all three cases, including CPU assembly/controller, CUDA allocations/copies, CG and source I/O; added observations/oracles/serialization are subtracted identically. Five checked trials follow a whole-pipeline warmup; average trials 2–5. Source printed/YAML timers include observations and are not the comparative timings. Original source allocations are retained; no synthetic padding-guard or unrestricted memory-safety claim.

## Clean-checkout preparation

Requires the existing CUDA12.0/nvcc, Julia1.12.6 and the pinned `_julia_ka_env` / `_julia_jacc_env` projects. No system installation is performed. From repository root, generate the large **actual native device readbacks**, which are regenerable and deliberately not committed:

```sh
mkdir -p .porting-state/porting_logs/miniFE-julia-shared/canonical_inputs_v1
flock -x /tmp/hecbench-julia-study-gpu.lock env CUDA_VISIBLE_DEVICES=6 python3 src/miniFE-julia/build_program.py --mode native --output /tmp/miniFE_native_prepare --arch sm_70
flock -x /tmp/hecbench-julia-study-gpu.lock env CUDA_VISIBLE_DEVICES=6 /tmp/miniFE_native_prepare "$PWD/src/miniFE-julia" "$PWD/.porting-state/porting_logs/miniFE-julia-shared/canonical_inputs_v1" "$PWD/.porting-state/porting_logs/miniFE-julia-shared/reproduction_prepare_aux" --prepare
sha256sum .porting-state/porting_logs/miniFE-julia-shared/canonical_inputs_v1/actual_input_*.bin
```

Expected SHA256 and sizes:

| Cells/axis | Bytes | SHA256 |
|---|---:|---|
| 2 | 5340 | af6d9a10fbcf9242d4cb4d19c60c4edc6f5e13a94e40f6f10957844512e8f7ae |
| 16 | 1627996 | 7f17c753a8c33d9cf1e469eb58d353b510e75d7608a440d1d6555289b7134502 |
| 128 | 779253852 | 489b2a9f7f0fba554d6478d61a16bd2979f9b46db43dbf6b63befb6ca7538f56 |

Then generate a fresh relocated config and run all four versions. **Do not put an outer flock around the central verifier**, which acquires the shared GPU lock itself:

```sh
python3 src/miniFE-julia/make_config.py --models all --output .porting-state/portable_julia/configs/miniFE_reproduction.json
python3 tools/verify_portable_julia.py --config .porting-state/portable_julia/configs/miniFE_reproduction.json
```

CPU-only saved-capture checks (replace RESULT with the newly printed result path):

```sh
python3 src/miniFE-julia/audit_captures.py --result RESULT --output /tmp/miniFE_full_audit.json
python3 src/miniFE-julia/audit_reduction_histories.py --result RESULT --output /tmp/miniFE_reduction_audit.json
g++ -std=c++17 -O3 src/miniFE-julia/continuum_center.cpp -o /tmp/miniFE_continuum_center
/tmp/miniFE_continuum_center
```

The continuum sine-sum convergence table is a separate physical interpretation check, not a new GPU series or rigorous truncation bound. The original full128 CLI, coarse-case discrepancy, minimal CPU boundary witness and native build/measurement diagnostics remain in the local ignored logs. Exact measured sources/configs/results and actual token boundaries are linked by `effort_summary.json`, `paper_observations.json` and `ready_support_files.json`; no historical costs were reconstructed from source lengths.

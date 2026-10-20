# miniBUDE: native CUDA, CUDA.jl, KernelAbstractions and JACC

All four versions pass five checked trials for the complete original BM1 deck (65,536 poses, 26 ligand atoms, 938 protein atoms) plus five controls: first/stratified source poses, an analytic pair, a 513-type cooperative-cache case and joint translation. Only the source one-pose-per-thread specialization is covered; full BM2 and other specializations remain unverified. Empirical energy consistency does not establish binding-affinity accuracy.

`study_plan.json` prospectively declares Float32 inputs and energy budgets `0.05 + 0.0002*abs(Float64_reference)`. Independent Float64 rotations and piecewise-potential sums evaluate all pose/atom pairs; the separate NumPy audit covers 836 selected poses and analytic invariants. `raw_state_audit_20trials.json` checks every actual input/initial output, complete energy array, read-only buffer and guard across 20 trials. The original CLI's small/signed-denominator checker is weaker; its original warmup timing also lacks completion. Source GPU arithmetic remains unchanged. Primary measurements complete warmup and sum six seven-launch region means, excluding captures/checks; average trials 2–5 only. No native per-case timing curve was retained.

All targets use genuine framework kernels, packed 16-byte structs and source 256-thread padded groups. KA uses typed local memory. JACC.shared copies the actual unpadded forcefield device view, including all 513 table elements with strided loads, followed by its barrier. Invalid padded lanes prepare the clamped final pose and participate in the barrier, then skip output stores. No native device kernel fallback or foreign device helper is used.

## Reproduction

Use the pinned `src/_julia_ka_env` for CUDA.jl/KA and `src/_julia_jacc_env` for JACC, a V100 and NVCC 12.0.140. With NumPy available in a user-space environment:

```sh
python3 fetch_original_assets.py --output-dir /tmp/minibude-original-new
python3 generate_assets.py --asset-dir /tmp/minibude-original-new/data/bm1 --output-dir /tmp/minibude-generated-new
cp independent_oracle.cpp /tmp/minibude-generated-new/
cd /tmp/minibude-generated-new
g++ -O3 -std=c++17 -fopenmp independent_oracle.cpp -o independent_oracle
OMP_NUM_THREADS=16 ./independent_oracle
```

Compare generated hashes with `asset_contract.json`, `independent_oracle_audit.json` and `cold_input_regeneration_audit.json` before preparing a new study directory. The generator refuses to overwrite frozen measured fixtures. Copy verified generated files into a fresh benchmark checkout, retaining these recipes. Run the central verifier with the native configuration first, then CUDA.jl, KA and JACC configurations with updated absolute paths and reuse paths pointing only to those new unchanged references. Preserve the shared GPU lock; check all five trials, not only timing. `audit_saved_evidence.py RESULT` checks the full retained data, including all guards. Existing absolute configurations are historical measured provenance and must remain unchanged.

`translation_accounting.json` and canonical manifests retain observed context-inclusive token windows, program checks and actual device-JIT counts. CUDA.jl's two disjoint windows include user configuration/coordination; KA and JACC include shared coordination. All three are excluded from clean target-cost forecasts. Cached input/reasoning are subsets; shared phase overlaps must not be added. Historical missing tokens remain unknown.

Raw GPU captures, executables, shared objects and regenerable `.bin` arrays remain outside source and paper Git. Exact source asset retrieval and CPU regeneration are provided instead. Paper RQ facts and limitations are recorded in `paper_observations.json`.

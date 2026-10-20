# TPACF: named scientific and source-fidelity study

Four genuine pair/histogram GPU definitions and own Julia host pipeline in each target. All targets are **hybrid_device_intrinsics**, because faithful constant-space loads use custom LLVM AS4 and CUDA module uploads (reused FHD/cmembench helper provenance recorded). No target calls native computation through an ABI; the C++ observer only checks/captures.

## Scope and corrected reference

Cases: N=97176,nr=2,njk=10; N=16388,nr=1,njk=4; N=8,nr=1,njk=2. Source fixed16384-square workspace and complete tiled schedule preserved. Original CLI100random catalogs and actual sky dataset are unverified; declared upstream asset URL returns404. Inputs are explicitly synthetic56unit directions. Exact inputs/outputs, all source final raw/compiled histograms and every report numeric token are checked in five trials/model; one full warmup is outside timing, mean uses2–5.

`reference_provenance.json` preserves source hashes, exact CPU ASan bounds witnesses and padding-bin witness. Minimal corrected secondary allocates63boundary values, bounds RR accumulation and selects actualzero bin29. All4native GPU bodies unchanged. No unsafe original GPU run or original wrong-GPU-output observation claimed. Unrepaired source one-sided group exclusion is **not a valid both-endpoint jackknife**; report overflow indexing is retained. Excluded untouched-native and faithful-scientific-operator aggregates. Independent integer suffix-multiplicity pair oracle and90digit exact-input dot cutpoint separation certify physical pair counts; source report policies are separately scoped, not cosmological validation.

## Reproduce fixture and run

From repository root, existing g++/nvcc, Python3 and pinned Julia environments required. Export original-source boundary values with `g++ -std=c++17 -O2 -ffunction-sections -fdata-sections src/tpacf-julia/export_boundaries.cpp -Wl,--gc-sections -o /tmp/tpacf-export` then `/tmp/tpacf-export src/tpacf-julia/data/source_boundaries63.bin`. Run `python3 src/tpacf-julia/prepare_catalogs.py`, `python3 src/tpacf-julia/prepare_case_header.py`, and `python3 src/tpacf-julia/prepare_native_instrumentation.py` **only before new verification**; do not regenerate dependencies of frozen runs. Essential data SHA/size inventory: `essential_assets_for_commit.json`; actual runtime assets explicitly bound by every model config. Expected input frames are deterministic generator products.

Execute `python3 src/tpacf-julia/run_study.py --config src/tpacf-julia/study_config_jacc.json` (or corresponding native/CUDAjl/KA config). Existing verifier owns globalGPU6 lock; no outer lock. Wrapper places new captures in owned/tmp STATE and writes canonical aliases/storage provenance. These paths must be preserved before reboot. No frozen evidence was deleted/moved.

Run `PYTHONPATH=/tmp/stencil_ports_numpy_226 python3 src/tpacf-julia/audit_actual_captures.py --model jacc --result RESULT --receipt RECEIPT` and `python3 src/tpacf-julia/audit_constant_copies.py --model jacc --result RESULT --receipt RECEIPT` for retained actual input/guard/constant bytes. Source/controller bindings and all20 primary artifacts are in `final_source_artifact_audit.json`.

## Interpretation

Timing includes source tile compute copies/kernels/result copies/CPU partial histogram accumulation, but excludes global setup/I/O/reporting/independent audits. Observer input captures pause the timer symmetrically. Native/private stream, pinned-host allocation, C++vsJulia CPU work, different measurement periods, compiler unroll controls and JACC cooperative shared preload order remain confounds. No causal framework speedup, roofline or original-sky application claim. Attempts and actual context-inclusive usage boundaries remain separate; nested verification windows overlap translation, not additional token cost.

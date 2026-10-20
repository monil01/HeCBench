# XLQC: complete water Hartree–Fock pipeline

All four versions pass the original H2O/STO-3G calculation in both precision modes: 7 contracted/21 primitive basis functions, 10 electrons and nine converged SCF iterations. The source GPU kernels and CPU scientific controller are preserved. Julia versions execute their own four Coulomb/exchange GPU entries while retaining the original CPU/GSL controller through an explicit C ABI. These are hybrid pipelines, not full host-code translations or cross-backend portability demonstrations.

**SP describes the Rys integral arithmetic.** Global coordinates, density, Q and output arrays, shared reductions and the SCF controller remain Float64. DP also uses Float64 for the Rys arithmetic. The original source geometry/basis files were recovered from the exact 1,816-byte DVC object, MD5 `b63a8a847119dbb8daf69de5bc9ad244`; per-file SHA256 values are in `asset_provenance.json`.

## Evidence and timing

| Mode | CUDA | CUDA.jl | KA | JACC |
|---|---:|---:|---:|---:|
| SP, ms | 6.111990 | 6.929433 | 7.336615 | 7.734970 |
| DP, ms | 8.495948 | 12.589826 | 12.801667 | 13.357761 |

These are means of trials2–5 after a complete SCF warmup; all five checked trials are retained. The ROI sums the original18 J/K dispatch-and-context-completion regions. Basis parsing, CPU/GSL DIIS/eigen/density work, readbacks and scientific checks execute outside those regions. Added validation pauses between stages can affect clocks/caches. This is neither full SCF wall time nor pure device event time, and no profiler/roofline or causal bottleneck claim is made.

Every configuration/model has five complete7,029-word artifacts, including495 exact initial-input words. The final CPU audit replays all40 artifacts, hashes11,360 physical captures and rechecks2,642 source/dependency bindings. Every actual stage density input and every primitive J/K output is checked against an alternative Boys/B-array Coulomb integral implementation, with the original raw-Q screening policy preserved. Coupled feedback densities are compared as numerical states, not falsely declared fixed input bits after the first iteration.

Every iteration checks CᵀSC=I, DSD=2D, Tr(DS)=10 and FC=SCε. A fresh unscreened physical F(D_new) checks the converged commutator and HF energy independently of the source's post-DIIS scalar. Raw MO coefficients are retained with these checks; their arbitrary phase identity is not a primary comparison claim. The independent CPU oracle tightens only validation routines, not native GPU or SCF code. `verification_plan.md` gives all tolerances declared before the checked study.

The actual input uses Rys roots1–3/BoysX0..36.76205. Other basis sets/higher-root branches are unverified; the source's empty Root6 path is not a framework limitation. This validates the named finite-basis numerical model, not experimental chemistry or an independently reimplemented complete SCF solver.

## Mapping and feedback

CUDA.jl and JACC retain physical8×8 blocks and21×21 grids; JACC specifies zero extra dynamic shared memory and preserves512B static source reduction storage. KA flattens its physical launch, so its logical Cartesian indices must be supplied to the source math. Its first candidate incorrectly used physical CUDA indices and faulted; the repaired mapping passes all checks. `KA_mapping_review.json` proves reversing only the index substitutions recovers the frozen CUDA.jl math byte for byte.

The first CUDA.jl invocation failed before device compilation due to host variable shadowing in a launch macro and a redundant private API call. A later complete five-trial calculation passed the scientific checks but had truncated C++ stdout JSON through a nonblocking Julia pipe. Its outputs remain rejected historical evidence. The final adapter captures the original C++ text in a regular file and re-emits it through Julia outside timing. No GPU math or tolerance changed to fix reporting. Counts distinguish program submissions, actual device-JIT processes, mapping candidates and compiler internals (unknown).

## Cold reproduction

Use a fresh fixture/dependency root, existing CUDA12.0 nvcc, a compatible GPU and the repository's pinned Julia1.12.6 projects. No system package installation or privileged command is required. GNU GSL2.8 is built only in the selected private prefix. Generated Julia is included; pycparser2.22 is needed only to regenerate it. CPU replay uses NumPy2.2.6.

Run from the repository root:

```bash
XLQC_FIXTURES=/tmp/xlqc-reproduction-inputs
XLQC_DEPS=/tmp/xlqc-reproduction-deps
python3 src/xlqc-julia/retrieve_inputs.py --output "$XLQC_FIXTURES"
python3 src/xlqc-julia/prepare_gsl.py --root "$XLQC_DEPS" --jobs 4
flock -x /tmp/hecbench-julia-study-gpu.lock env CUDA_VISIBLE_DEVICES=6 python3 src/xlqc-julia/build_program.py --gsl-prefix "$XLQC_DEPS/install" --arch sm_70 --output /tmp/xlqc-reproduction-native
flock -x /tmp/hecbench-julia-study-gpu.lock env CUDA_VISIBLE_DEVICES=6 /tmp/xlqc-reproduction-native 0 "$XLQC_FIXTURES" /tmp/xlqc-reproduction-prep-SP 1
flock -x /tmp/hecbench-julia-study-gpu.lock env CUDA_VISIBLE_DEVICES=6 /tmp/xlqc-reproduction-native 1 "$XLQC_FIXTURES" /tmp/xlqc-reproduction-prep-DP 1
python3 src/xlqc-julia/make_config.py --precision sp --models cuda,cuda_jl,ka,jacc --fixtures "$XLQC_FIXTURES" --gsl-prefix "$XLQC_DEPS/install" --output src/xlqc-julia/study_configs/reproduction/SP_all4.json
python3 src/xlqc-julia/run_study.py --config src/xlqc-julia/study_configs/reproduction/SP_all4.json
python3 src/xlqc-julia/make_config.py --precision dp --models cuda,cuda_jl,ka,jacc --fixtures "$XLQC_FIXTURES" --gsl-prefix "$XLQC_DEPS/install" --output src/xlqc-julia/study_configs/reproduction/DP_all4.json
python3 src/xlqc-julia/run_study.py --config src/xlqc-julia/study_configs/reproduction/DP_all4.json
```

Do not put another flock around `run_study.py`: the unchanged central verifier owns that lock. `--julia`, `--nvcc` and `--gpu` configure a new reproduction; the reported study used GPU6 and `/usr/bin/nvcc`12.0.140. Use the two printed result paths for CPU replay:

```bash
python3 src/xlqc-julia/audit_captures.py --sp-result SP_RESULT_JSON --dp-result DP_RESULT_JSON --output /tmp/xlqc-reproduction-audit.json
```

The original source asset/archive, GPU/CPU sources, cloned validation math, generator and package/dependency fingerprints are retained. Large captures and compiled libraries remain local; `/tmp` files and symlinks require archival before cleanup/reboot. Pinned Project/Manifest/preferences fingerprints match observations before and after the study; they were not explicit per-process dependency bindings, so no independent per-process loaded-package audit is claimed.

`all4_accuracy_timing_summary.json`, `all4_full_capture_audit.json`, `implementation_traits.json`, `effort_summary.json`, `paper_observations.json` and `ready_support_files.json` contain the compact evidence and actual inclusive token counters. Cached/reasoning counts are subsets. Marginal KA/JACC costs reuse the completed math/controller/oracles and are not intrinsic or cold-start framework rankings.

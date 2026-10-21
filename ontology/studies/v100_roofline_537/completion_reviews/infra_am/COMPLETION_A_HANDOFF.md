# Completion A: frozen CPU staging handoff

New source helper `tools/v100_roofline_prepare_completion_a.py`; exact SHA, review hash and all fifteen plan/diff/source proof hashes are in `completion_a_delivery.json`.

Staged root: `/tmp/hecbench-roofline-v100-20261020/batch023-completion-a`.

Cases: CCSnative, CMJACC, miniDGS4, MF-SGDnative, BNnative, clocknative, GC4, memcpynative, columnarSolvernative. All15 have empty remaining isolation gates and staged=true; this does not assert replay authorization, scientific success under instrumentation, ROI completeness or a bottleneck. No target/build/GPU execution occurred.

Regeneration requires a fresh output root, exact `completion_a_review.json` SHA, the currentv7 base plans recorded there, and unchanged bound sources. Example command:

```
python3 tools/v100_roofline_prepare_completion_a.py \
  --source-root /home/imo/HeCBench_julia \
  --review /tmp/roofline-completion-20261021/infra_am/completion_a_review.json \
  --review-sha256 5a881043c6581a194b6c910238b89d6fd3b8cbc1e206d56eb4dca5f377930c98 \
  --output-root /tmp/NEW-FRESH-COMPLETION-A
```

Review scope: output paths, readonly canonical includes/fixtures, output-file lifecycle and compiler executable placement. Original selected result/config/math/kernels/inputs/precision/RNG/geometry/repetitions/tolerances/checks unchanged. Compilation remains inside existing Landlock. Runtime assets are copied exact and readonly; output/env/TMPDIR remain fresh per model. Source-before/after checks cover exact reviewed sources plus full current native/Julia context. Original historical adapter bindings stay intact; extra current context is not retroactive numerical certification.

Notable details:

- CCS replaces libc fixed-temp tmpfile with exclusive mkstemp+unlink+fdopen in own scratch, retaining full print/seek/output bytes. This host-adapter copy is not compiler-validated yet.
- CMJACC retains all source bodies and scratch initial/final checks, moving only trace output.
- MiniDGS copied Python driver keeps original canonicalP input/child lookup and2MPI ranks; native buildoutput is aligned with wrapper's existing `STUDY_OUTPUT_DIR/native_rank`. All rank artifacts/persistence moved only into fresh outputs; no source MPI/GPU algorithm change.
- MF-SGD precreates the exact fresh argv4 output directory; no source copy required.
- BN preserves even failure-only capture writes, redirected into scratch.
- GC copied per-model entry includes copied common; all other includes and fixture reads point canonical readonly source. CUDA.jl selected `study.jl`, not legacy `main.jl`, remains the entry.
- Clock native fixture read unchanged, raw timings root fresh.
- Memcpy native fixture read unchanged; fixed `/tmp` capture output becomes fresh private output. Kernel roofline NA, if observed, must follow actual completed run.
- Columnar native retains originalsource/CPUgold and exclusive mkdir; only output/executable roots fresh and compiler path replaced by existing `/usr/bin/nvcc`→actualCUDA12.0.140. Same-O3/-std=c++17/-arch=sm_70/DATA_PATH flags. No12.9 orWMMA policy changed.

Root should freeze `completion_source_proof.json`, `compiler_manifest.json`, `plan_diff.json` and exact helper/review along with ordinary profile evidence. Helpers and plans are now frozen; do not edit the helper while queued plans bind its SHA.

EGS remains separate: observed runtimecompiler exit255 does not prove a tempfile/environment cause. No compiler execution or speculative child-buildenv change was performed. The proposal to prebuild exact unchanged checker still needs independent root review. Existing ASSERT and rootretry021/bitcracker work were not duplicated.

An incidental CPU source audit of active native Rayleigh–Bénard is recorded separately in `RBC_phase_review.json`; it is included in this shared effort window, with no retrospective per-case cost allocation.

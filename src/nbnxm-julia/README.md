# nbnxm four-way study

Original 3199 superclusters/204808 atoms/10000 repeats for both shift flags, plus four nonzero/self/exclusion/cutoff controls. Original masks all zero: no physical pair arithmetic in default. Source GPU body unchanged; common explicitly resident managed-data study, not full GROMACS/PME/integrated MD or original first-touch latency.

All nine actual device buffers byte-compared against six full native canonical readbacks every trial/model; full force and 45 shift vectors, 2462658 Float64-encoded Float32 values/trial, 1231362 exact-prefix values. Full independent analytic LJ/Ewald pair oracle at .002+1e-4*abs(ref); exact default/tails/untouched fields; normalized Newton<=5e-5; nonfinite rejected.

## Reproduce from clean checkout

Run from repository root with existing nvcc, Python3, Julia and pinned environments. No system package changes. Canonical inputs are regenerated from the archived source initializer and owned deterministic controls; no external dataset or binary is needed. The actual native readbacks occupy about22MB. Original source/COPYRIGHT remain in reference_history.

```bash
mkdir -p .porting-state/porting_logs/nbnxm-julia-shared/canonical_inputs_v1
flock -x /tmp/hecbench-julia-study-gpu.lock env CUDA_VISIBLE_DEVICES=6 python3 src/nbnxm-julia/build_program.py --mode native --output /tmp/nbnxm_prepare --arch sm_70
flock -x /tmp/hecbench-julia-study-gpu.lock env CUDA_VISIBLE_DEVICES=6 /tmp/nbnxm_prepare "$PWD/.porting-state/porting_logs/nbnxm-julia-shared/canonical_inputs_v1" "$PWD/.porting-state/porting_logs/nbnxm-julia-shared/reproduction_prepare_aux" --prepare
python3 src/nbnxm-julia/run_study.py --models all
```

The central verifier owns the GPU6 lock; never add an outer lock to run_study.py. A complete series writes about394MB of full outputs. Five trials follow compile/warmup; average trials2–5. Allocation/reset/prefetch/readbacks/oracle/serialization excluded, six launch loops plus context completion included. Default10000repeats/flag; physical controls1repeat. Not original first-touch timing.

Expected canonical SHA256:

`actual_input_centralBipartite.bin` 028a75c32feaf82b7804a960a268b4f8f5fd62639233d128c2abcdb9a6bace16

`actual_input_centralSelf.bin` a1197d9da144aa9c1a58795c10481cc018cd6f3986d2936c6aa2b927c670ba01

`actual_input_originalNoShift.bin` 1f60b09e82e1a0a522e5de87e88f1b99776b10ea49ba6bd954885036df544aea

`actual_input_originalShift.bin` 8fa455b615b765c2a55389ae9a975cd4a6f2b4caaffbf094d442b6db32e19dc6

`actual_input_shiftedBipartiteNoShift.bin` c9dbd16ed50af742507c2b3c1d0ad99e158438c1c306581aa3812d8da18435eb

`actual_input_shiftedBipartiteShift.bin` 4678b0f66d8cb835fde19a583a4e2ce56d58e4cc86ac7ed7dbbe886bd71aa379


CPU-only complete saved-output replay:

```bash
python3 src/nbnxm-julia/audit_captures.py --result PATH_TO_RESULT_JSON --output /tmp/nbnxm_cpu_audit.json
```

See all4_accuracy_timing_summary.json, all4_full_capture_audit.json, effort_summary.json and paper_observations.json. All targets own the genuine8x8 dispatch and force kernel. KA uses lexical local memory/barriers with CUDA warp/atomic/math extensions; JACC uses explicit static shared/warp/atomic extensions and zero dynamic shared. Native allocation/controller remains C ABI. No hidden native selected-kernel fallback. Compiler/host-dispatch/device-code contributions unprofiled. Float64 artifacts encode Float32 GPU data; analytic CPU oracle is separate. Original72-atom tail, readonly inputs and untouched outputs checked; no synthetic padding guards or universal memory-safety claim.

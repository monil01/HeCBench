# Group B hidden capture-root isolation proposals

18 source/config/result-bound, **unstaged** CPU plans: sheath 4, snicit 4,
testSNAP CUDA, tpacf 4, zmddft 4, sparkler CUDA. No gates were removed.
No source originals, historical captures, configurations or results were modified.
No compiler, Julia, MPI, GPU or staging commands were executed by the planner.

## Reproduce fresh proposals

```sh
python3 tools/v100_roofline_prepare_capture_overlays_b.py \
  --base-plan-root /tmp/roofline-method-review/reviewed_plan_v4_plus_v5_cap512 \
  --review-table /tmp/roofline-method-review/dynamic_output_scan/confirmed_output_root_proposals.json \
  --output-root /tmp/FRESH_B_PROPOSALS
```

`--stage` is reserved for coordinator review; it copies declared inputs and exact
adapter texts only. Builds remain the original explicit compiler commands with
fresh output paths. Output must be a new directory below `/tmp`; asset copy cap
is 512 MiB **per slot**, with 2048 MiB free reserve. These limits are staging
limits, not predictions of subsequently generated capture size.

## Reviewed output and read-only roles

- **sheath:** native physical/grid diagnostics and Julia rawdir move below fresh
  outputs. Native relative `data/` inputs retain staged cwd behavior; Julia ROOT
  remains canonical. `studyphysics.so` stays freshly built in scratch cwd.
- **snicit:** copied Python wrapper preserves child arguments and canonical
  `audit_model.py`, retaining the independent complete trace audit. Copies of
  traces/audit move to `STUDY_OUTPUT_DIR/retained_diagnostics` instead of old STATE.
- **testSNAP CUDA:** canonical `source/data` and reference-header include flags
  preserved. Observer captures/CAS/guards move below fresh private_capture.
- **tpacf:** canonical catalog/boundary/oracle roots retained. Observer CAS,
  guards, constants move to fresh private_capture; report files and randomized
  catalog symlinks remain scratch-cwd outputs, with canonical read-only targets.
  Fresh scratch observer library and existing audit calls remain intact.
- **zmddft:** canonical BASE retains full original inputs/oracles; impulse-stage
  diagnostics move below fresh outputs. Full 256-cubed/R100 kernels unchanged.
- **sparkler CUDA:** native backend capture root moves to scratch. Original
  launcher, 4 MPI ranks co-resident on GPU6, streams, MPI collectives, backend
  libraries, allocations and numerical/CPU control remain unchanged. This does
  not independently establish MPI replay safety or whole-ROI attribution.

Copied C++ adapter quoted includes are rewritten to their canonical absolute
paths (and hash-bound). Copied Julia entries retain canonical device/case
includes while loading copied common adapters. Mathematical/kernel bodies,
iterations, scientific acceptance and timed-region statements are unchanged.

Every slot records base-plan, immutable result/config, original sources, exact
patch texts/diffs and prospective copy hashes. `execution_bindings` includes
original/current canonical dependencies; staged copy bindings are added only
when the coordinator explicitly stages. Original input bindings are rehashed
before and after proposal generation. Runtime write guard is untouched.

These are output-isolation proposals, **not** proof of complete source ROI,
executed kernels, replay safety, successful compilation or profiling outcomes.
The shared profiling effort receipt is separate from benchmark translation.

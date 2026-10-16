# Julia Correctness Check

Status date: 2026-10-16

## Scope

This file summarizes the numerical correctness sweep and subsequent corrections for Julia
(`CUDA.jl`) ports in HeCBench.

Input set:
- `441` Julia ports with existing `ok` status in `.porting-state/coverage.db`
- CUDA was treated as the reference model
- Results were written by `tools/verify_numeric.py`

Tracking artifacts:
- `.porting-state/numeric_accuracy.db`
- `.porting-state/numeric_accuracy/reports/counts.txt`
- `.porting-state/numeric_accuracy/reports/mismatches.txt`
- `.porting-state/numeric_accuracy/reports/unverified.txt`
- `.porting-state/numeric_accuracy/reports/not_run.txt`
- `.porting-state/logs/numeric_accuracy_summary_2026-09-24.md`

## Results

```text
pass|native_pass_trusted|259
pass|numeric_stdout|41
mismatch|numeric_stdout|72
unverified|no_numeric_data|56
not_run|reference_fail|1
not_run|run_status|11
not_run|timeout|1
```

Summary:
- `300 / 441` passed either benchmark-native correctness checks or numeric
  stdout comparison.
- `72 / 441` reported mismatches and require investigation.
- `56 / 441` did not expose comparable numeric data and need benchmark-specific
  metadata, checksums, or artifact comparisons.
- `13 / 441` did not complete a CUDA/Julia numerical comparison during the
  sweep.

## Interpretation

`pass|native_pass_trusted` means CUDA and Julia both emitted benchmark-native
`PASS` verdicts and neither emitted `FAIL`.

`pass|numeric_stdout` means comparable non-timing numeric stdout values matched
within the verifier tolerance.

`mismatch|numeric_stdout` means comparable numeric stdout was found but CUDA and
Julia values differed beyond tolerance.

`not_run|reference_fail` means the CUDA reference emitted a native `FAIL`
verdict, so the Julia result cannot be compared against a passing CUDA
reference.

`unverified|no_numeric_data` means the benchmark ran but did not expose
comparable non-timing numeric output. These ports need benchmark-specific
numeric metadata or artifact comparison before numerical equivalence can be
claimed.

`not_run|run_status` and `not_run|timeout` mean the numerical sweep could not
complete the CUDA/Julia pair for that benchmark.

## Correction: atan2 (2026-10-16)

Fixed the Julia CPU-reference argument order for float32, int32, and int16
outputs to match the GPU launches and CUDA reference. Native verification now
checks all three output types and propagates failure through the process exit
status. The prior float-only threshold of 10 has been tightened to 1e-3;
integer RMSE must be zero.

Verification used Tesla V100 (sm_70), the local CUDA 12.0 toolkit, and canonical
CUDA arguments `10000000 100`. With CUDA `EXTRA_CFLAGS=--fmad=false`, both
implementations printed zero RMSE for all three output types; the numerical
verifier reported `PASS numeric_stdout` with zero absolute/relative difference,
and coverage reported `MATCH`. Numerical comparison tolerances were unchanged.
This compares the printed RMSE summaries rather than raw output arrays.

The default CUDA build with FMA enabled reported int32 RMSE 1.177096 and int16
RMSE 0.000548 while Julia reported zero. The passing record therefore applies
to the controlled build with contraction disabled. Reproducing this record
requires that build flag; default-build summary matching remains unresolved.
The original input generators still differ between CUDA and Julia.

Evidence, build commands, correction token accounting, and the local checkpoint
are under `.porting-state/numeric_accuracy/fixes/atan2/`. Original translation
usage and attribution are preserved.

## Next Work

1. Investigate the `72` mismatch rows first.
2. Add `numeric_check:` metadata or deterministic checksums for the `56`
   unverified rows.
3. Repair or retest the `13` not-run/reference-blocked rows.
4. Rerun `tools/verify_numeric.py` after each fix and update the local summary.

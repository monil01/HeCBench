# Julia Correctness Check

Status date: 2026-09-24

## Scope

This file summarizes the first numerical correctness sweep for Julia
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
pass|numeric_stdout|40
mismatch|numeric_stdout|73
unverified|no_numeric_data|56
not_run|reference_fail|1
not_run|run_status|11
not_run|timeout|1
```

Summary:
- `299 / 441` passed either benchmark-native correctness checks or numeric
  stdout comparison.
- `73 / 441` reported mismatches and require investigation.
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

## Next Work

1. Investigate the `73` mismatch rows first.
2. Add `numeric_check:` metadata or deterministic checksums for the `56`
   unverified rows.
3. Repair or retest the `13` not-run/reference-blocked rows.
4. Rerun `tools/verify_numeric.py` after each fix and update the local summary.

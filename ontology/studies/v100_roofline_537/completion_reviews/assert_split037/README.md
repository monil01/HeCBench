# ASSERT split profiling proposal — batch037

**Source reviewed, runtime unvalidated. No GPU/build/test execution was performed.**

`delivery.json` freezes three plans, helper and review SHA256. Root integration is
restricted to ASSERT CUDA.jl/KA/JACC via `--isolated-assertion-reviewed`. Original
selected correctness results/configurations, source kernels, geometry, installed
Julia executable and project dependencies are unchanged. Native ASSERT is untouched.

## Required collector integration

1. Validate review SHA, entry `assert/MODEL`, config/result SHA, exact helper/spec,
   commands and all execution bindings. Only then remove the single
   `requires_isolated_assertion_collector_phase_hooks` gate **in memory**. Preserve
   the original staged plan bytes/hash.
2. Hold the existing GPU lock across the complete pre/profile/post sequence. Use
   the existing child Landlock guard for each phase; check source hashes before
   and after phases. GPU identity, free-space reserve and foreign-activity checks
   remain mandatory. These helpers do not independently prove a lock/guard.
3. Run the exact reviewed `pre_profile_check_command`, adding only that phase's
   `HECBENCH_ASSERT_PHASE_AUTHORIZATION` environment marker from the review. This
   marker is a collector attestation, not a security primitive. Write the actual
   launcher result to provenance as **runtime checks**, not compiler/build/JIT.
   `pre_profile_checks.log` is persistent at slot top.
4. The helper runs **all five** fresh original `--negative` processes even if one
   fails. Full stdout/stderr and signed returncodes are persisted after each child
   before checking the final gate. Each child must return zero and produce exactly
   one `ASSERT_STATUS 710` token pair. Any failure forbids NCU. Collector operational
   timeout/foreign interruption may truncate the sequence; this remains failure,
   and partial receipts must be retained. No negative child runs under NCU.
5. NCU targets the exact original positive Julia command directly; no source-copy
   wrapper, no nested negative process. Its environment has no authorization
   marker. Keep original selector caveats: first positive function/geometry may
   select warmup; no whole timed-ROI coverage or arithmetic roofline claim follows.
6. After real NCU session quiescence, collector writes
   `positive_profile_outcome.json` with the exact schema in the review. It records
   the real returncode/stop reason, direct positive argv, immutable plan SHA,
   complete execution.log SHA/size and actual lock/guard/quiescence attestations.
   This file is not a counter-parser verdict or a substitute for PCI/source checks.
7. Run exact `post_profile_check_command` under the same guard/lock and marker;
   retain `post_profile_checks.log`. It rehashes raw negative streams, reparses real
   710 tokens, checks all five ordered original positive PERF_TRIAL rows, finite
   positive component times and sums, source stability and actual rc0/no-stop.
   Literal execution-only fields come from the bound original row.update source.
   Mandatory post success + `joined_execution_contract.json` precede acceptance.
   Counter parsing/PCI/metric validity remain separate collector gates.
8. Preserve raw phase outcomes and all partial diagnostics on every failure. Freeze
   negative_raw recursively using exact-byte/base64 evidence before scratch cleanup.

## Persistent receipt inventory

Per slot: `phase_spec.json`, `negative_prechecks.json`,
`negative_raw/{1..5}.stdout`, `.stderr`, `.combined.log`,
`positive_profile_outcome.json`, `joined_execution_contract.json`,
`pre_profile_checks.log`, `post_profile_checks.log`, ordinary execution.log,
plan/provenance/counter artifacts. JSON receipts are strict finite JSON.

Successful new per-model workflow: five negative runtime process contexts + one
positive profiled program, zero authored kernel changes or compiler/build commands.
Backend JIT invocation counts remain unknown. Negative contexts are intentionally
poisoned then exited and are never reused as positive numerical contexts.

## Scope and uncertainties

This is an **unproven profiling orchestration recovery**. Historical nested child
exit255/empty stdout remains unknown cause; do not assert NCU deliberately rejected
710, loader failure, Landlock failure or CUDA source failure without new evidence.
The accepted original timings/status rows are immutable. Instrumented times emitted
by this experiment never replace the original five-run performance series.

Source has no numerical outputs: all joined rows retain checked_values0, null
errors, no arrays, `no_numerical_outputs_execution_only`. The negative checks are
real execution/assertion evidence, not numeric agreement or full arithmetic work.
No actual negative or positive target was run during proposal preparation.

Static validation only: helper/staging source AST parsing and full declared-file
SHA comparisons. No unit tests, target imports, GPU/device queries or builds.

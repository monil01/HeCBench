# Finite V100 completion supervisor — root adoption required

New script: `tools/v100_roofline_completion.py`. Source-only implementation; no
supervisor, GPU, compiler/build, benchmark or tests were run by its author. Static
AST parsing and manual call/flow review only. No source/shared ledger edits beyond
this new tool; root controls adoption, all executions and Git.

## Invocation

Root must enumerate **all currently owned GPU controllers**, including the main,
finite followups and direct custom drivers, using exact PID plus Linux start ticks.
The tool never discovers/spawns another full campaign master. Do not enumerate CPU
refresh: it is forbidden and would create a wait cycle. Existing controllers may
exit before the supervisor starts; an absent/reused PID is not waited on or signaled.

```sh
python3 tools/v100_roofline_completion.py \
  --source-root /home/imo/HeCBench_julia \
  --raw-root /tmp/hecbench-roofline-v100-20261020 \
  --output-root /tmp/hecbench-roofline-v100-20261020/completion_SUPPLY_FRESH_ID \
  --reviews /home/imo/HeCBench_julia/ontology/studies/v100_roofline_537/approved_isolation_reviews_v7.json \
  --controller 1213116:SUPPLY_ACTUAL_START_TICKS \
  --controller SUPPLY_OTHER_OWNED_PID:SUPPLY_ACTUAL_START_TICKS \
  --max-retry-rounds 2 --freeze-every 10 --timeout 600 --execute
```

Placeholder values are deliberately invalid; do not copy without actual identities.
Compute start ticks from `/proc/PID/stat` after the final `)`: split field index19
(Linux field22). Standard owned campaign/followup/collect/retry/handoff controllers
are allowed. A direct custom driver requires `--controller-reviews FILE`, schema:

```json
{"controllers":{"PID:START_TICKS":{
  "command_sha256":"SHA256 of exact /proc/PID/cmdline bytes, including NULs",
  "source_reviewed_GPU_controller":true,
  "reason":"Exact reviewed custom finite GPU profiling controller; not CPU refresh"
}}}
```

PID UID must equal the executing unprivileged user. Refresh/completion processes
are rejected even if listed in the review. Do not infer historical ownership from
a currently absent PID. Root must supply a complete explicit monitoring frontier.

Without `--execute`, only private configuration/checkpoint/journal is prepared.
This is not a dry execution of commands or a claim of validation. Existing source
root/raw tree must be owned by current UID; output must be a new child of raw root.
No source roots, reviewed files or historical evidence are created/repaired.

## Finite state sequence

1. Exclusively lock raw-root `finite_completion_supervisor.lock`, then the private
   journal lock, so a second output directory cannot launch duplicate rounds.
2. Wait indefinitely for every exact controller identity to finish. Never send any
   signal. Foreign GPU contention waits remain the existing collector's normal
   behavior, with no extra supervisor deadline.
3. Summarize -> integrity audit -> require actual exit0 **and issues=[]**. Save exact
   audit and frontier inputs. Coordinator tool locks remain authoritative; no outer
   metadata lock is held across child tools (would deadlock their own locks).
4. Retry at most two finite rounds, fixed IDs **039,041**; **040 is never used**.
   Only selected frozen `profile_process_failed` whose actual profiler stop_reason
   is external GPU activity or wall_timeout are eligible. Reuse the existing retry
   tool, approved reviews, unchanged checks/math, explicit fresh batch IDs. It may
   use the existing source-qualified bounded-specialization selector for timeouts;
   geometry undercoverage remains disclosed. No new broad master is launched.
5. Retry launcher failure is preserved, round consumed once, then summarize/audit
   still runs; no silently repeated interrupted traversal. Other parser/build/path/
   replay/hardware failures do not become eligible merely by being unsuccessful.
6. Final existing refresh `--once` (no wait PID), verify each emitted internal tool
   returncode, then fresh summarize+zero-issue audit. Export fresh failure census,
   retained-family audit and all537x4 assessment using their own coordinator locks.
   Zero census issues/drift, all2148 assessment records and no assessment evidence
   issues are mandatory. All output command/log SHA receipts remain private.
7. Emit `ready_for_root_review.json` with denominators537/model, exact status/category
   distributions and remaining hardware/replay/plan/failed/pending records. This
   never means complete elapsed ROI, causal bottleneck or scientific validation.
   No automatic Git, push, paper mutation or publication is performed.

## Resume and previously started rounds

Reinvoke the **same exact options/tool bytes/reviews** plus `--resume`. Durable
independent step workers persist their own PID/start, child PID/start, command,
complete stdout/stderr, signed returncode and source/log SHA. They can finish after
an external supervisor disconnect. Resume waits an existing worker; it never
executes the same step a second time. A crash between launch intent and proof is
ambiguous and fails closed for manual review, not automatic repeat. All unfinished
steps are reconciled before launching new metadata work.

If a retry ID already has artifacts **not created by this supervisor**, explicit
`--existing-round-state 039=finished` (or041) is required to import it; this requires
an actual `retry_traversal_finished` event and retains journal/frontier SHA. For an
incomplete interrupted old round use `039=preserve`: consume the ID without retry,
retain artifacts and report remaining failures. This does not certify success.
A live external retry must itself be in the explicit controller frontier and
finish before import. Existing roots with no journal also require `preserve`.
Never change existing round states on a resumed configuration; configuration drift
requires root review. Supervisor/tool/review changes fail closed before new steps.

## Known limits

The monitor trusts root's completeness of supplied controllers; it does not signal
or infer unmanaged GPU work. Existing collector idle/identity/foreign-activity/write
confinement gates remain active on every retry. CPU refresh returns0 for some inner
failures, so this supervisor explicitly scans its retained JSON events and rejects
any failed substep/missing finished event. Metadata exports are sequential coherent
snapshots, not a new combined GPU execution or same-instant source/ROI certificate.
Original correctness runs/means and frozen attempts are retained unchanged.

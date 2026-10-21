# Conservative operational retry candidate — private, root adoption required

Authoritative source: `retry_candidate.py`, SHA and active-source diff recorded in
`delivery.json`. Active tools were **not edited**. Author executed read-only metadata
inventory mode only, without `--execute`; no GPU, compiler/build or tests. Source
syntax was inspected via AST and actual frozen metadata SHA/length joins were read.

## Why generic v1 can regress

`tools/v100_roofline_retry.py` re-runs the generic planner then collector with only
`--selection-overrides` for timeout retries. It does not carry through reviewed
`--large-artifacts`, `--application-replay-reviewed`, `--profile-window-reviewed`,
`--isolated-assertion-reviewed`, `--tensor-modes`, custom build/output overlays or
source-bound helper copies. Its `--timeout600` can also shorten a parent's explicit
requested/effective bound (collector's primary-mean floor cannot restore a larger
historical requested bound). Grouping several models under one invocation obscures
their distinct deadlines and methods.

The selected failure can remain an **older** generic attempt after a later repaired
or bounded attempt was frozen. Inspecting only selected flags is insufficient.
This candidate inventories **every frozen attempt for each eligible slot**, verifies
its captured plan/provenance and manifest compressed/raw SHA+length, and retains
that exact evidence. No current-source hashes backfill historical execution facts.

## Conservative eligibility and explicit dispositions

- Frontier remains selected `profile_process_failed` with recorded foreign-GPU or
  wall-timeout stop only. Other failures are not broadened into operational retries.
- Any noncanonical captured plan field, reviewed application/window/assert/tensor/
  large-artifact/selector method, or changed historical command/build/environment/
  ROI/contract signature yields `held_requires_fresh_source_bound_custom_method_reapplication`.
- Perlin/ZMDDFT/SU3 explicitly require large payload policy; ASSERT, system atomics,
  StreamUM and Rayleigh--Benard require case-specific method review. They never
  fall back to generic execution. Holds are not hardware/framework unsupported
  claims and do not alter accepted mathematics, evidence or terminal status.
- Old generic bounded selectors are conservatively held too. They could be admitted
  by a future exact standard-global-method equivalence review; this candidate does
  not guess that a source-counted/retained selector is interchangeable.
- A fresh generic plan can execute only if it is staged/no gates and normalized
  build argv, target argv, full environment, source cwd, timing region, config/result
  bytes, outcome and the entire execution-binding role/content set match the
  selected parent. **Dictionary keys and values** normalize scratch prefixes;
  canonical input/source paths and content hashes remain exact. No general path,
  numeric or semantic rewrite is allowed.
- Fresh stages that fail equivalence are preserved with an explicit disposition,
  not executed or silently replaced. The candidate never copies a custom method
  into a new directory without a new reviewed source-bound method adapter.
- Generic tool/helper/review bytes are bound before the traversal and checked
  immediately before/after every child command. Collector's own lock/write/source/
  numerical/foreign-activity gates remain mandatory.

## Deadlines

For each model slot, requested retry seconds are
`max(CLI timeout, every recorded historical requested_seconds/effective_seconds)`.
Ceil to integer. CLI and operational bound must be<=3600. If a historical bound is
larger, hold for explicit operational review; do not silently lower it. If an actual
historical profile lacks a deadline record, hold rather than invent a missing one.
Collector may raise a supplied timeout via its existing source-primary-mean rule,
never replace original performance means or certify a complete ROI. Each slot gets
its own collector invocation; one model's timeout cannot reset another's bound.

At this read-only snapshot, 169 selected stop slots existed. Five were held before
fresh preparation: columnarSolver/CUDA.jl, RayleighBenardConvection/native,
RayleighBenardConvection/CUDA.jl, RayleighBenardConvection/JACC and SLU/KA. The other
164 merely require prospective exact fresh-plan equivalence; they are **not already
approved or run**. Some historical floors exceed600s: bitcracker689;
columnarSolver1272/1297; HBC1118--1351; reviewed RBC941--1027 (RBC held separately).
The completed inventory is `review_snapshot_v3/method_preservation_review.json`.
Older v1/v2 inventory files remain development snapshots.

## Invocation and supervisor compatibility

After root source review/adoption, existing supervisor argv is compatible:

```
python3 ADOPTED_RETRY_PATH \
 --source-root /home/imo/HeCBench_julia \
 --raw-root /tmp/hecbench-roofline-v100-20261020 \
 --batch-id 039 \
 --reviews /home/imo/HeCBench_julia/ontology/studies/v100_roofline_537/approved_isolation_reviews_v7.json \
 --timeout 600 --freeze-every 10 --execute
```

Only IDs039/041 are accepted;040 remains reserved. Optional `--output-root` otherwise
defaults to `raw/retry_ID_method_review`. Read-only inventory should supply a fresh
private /tmp output as the author did, omitting `--execute`. Re-running an existing
round is forbidden if any retry journal/frontier/batch exists; root supervisor must
import/preserve the prior state, never reset it.

On execution, `raw/retry_039_frontier.json`, `raw/retry_039.jsonl` (likewise041) and
`retry_traversal_finished` naming remain available to existing supervisor import/
resume. Per-slot fresh plans are `batch039-retry-BENCH-MODEL/BENCH/MODEL/plan.json`.
The raw frontier freezes all dispositions, including holds. Detailed command logs,
SHA bindings and execution_dispositions.json persist privately. A finite traversal
is not scientific or ROI completion. No failed source/numerical/guard check is
waived; original artifacts, scope and performance samples remain immutable.

Root's already-running021 retains loaded old source unchanged. Re-adopting candidate
bytes requires a **fresh** supervisor configuration/source hash review, including
new042nativeSU3 in the explicitly owned controller frontier. This candidate itself
never discovers, duplicates or stops controllers or foreign GPU activity.

# V100 measurement operational-outcome census

Snapshot: 2026-10-21T18:25:07.795853+00:00. Integrity issues: 55. All four models retain denominator537; selected and latest outcomes are separate from repeated attempts. No percentages suppress pending or blocked slots.

| Model | Selected completed counters | Runtime NA | Pending/active | Hardware | Replay holds | Plan gates | Other selected | Frozen attempts | Profile launchers | Build launchers |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cuda | 357 | 1 | 115 | 5 | 1 | 19 | 39 | 429 | 407 | 408 |
| cuda_jl | 342 | 2 | 120 | 5 | 1 | 12 | 55 | 432 | 413 | 35 |
| ka | 339 | 2 | 120 | 5 | 1 | 13 | 57 | 434 | 411 | 34 |
| jacc | 340 | 2 | 120 | 5 | 1 | 13 | 56 | 432 | 411 | 34 |

Operational failures are profiling/build/runtime/isolation/dependency/parse observations, not translation failures. Cause remains unknown when only a generic nonzero exit is recorded. Profile/build launcher counts may include nested target/compiler/rank subprocesses; total backend compiler/JIT counts remain unknown. CPU-only recovery records add no execution. Runtime allocation/copy NA does not manufacture zero FLOPs or bytes. Successful kernel samples do not prove full ROI or causal bottlenecks.

Exact parent evidence and diagnostic snippets: all_frozen_attempts.json. Complete537×4 selected/latest rows: selected_latest_slots.json. All compressed/raw input SHAs and consistency scope: summary.json. Historical executed source identity is retained from captured provenance; current source files are never rehashed to certify past runs.

Run this exporter again to a new /tmp directory after future metadata updates; existing output directories are refused. No GPU, benchmark tests, compiler or target program is executed by this exporter.

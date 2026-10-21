# Proposed companion14/main integration

**No paper/source/shared-ledger changes made.** This packet describes the immutable
`measurement_failure_census/final_snapshot`, not the current running campaign.
Source progress: **2026-10-21 18:38:36.245563 UTC**. Source snapshot creation predates
lock acquisition/reading; do not interpret creation time as a simultaneous capture.
Every input is SHA-bound in tables/packet_summary.json. Six documented source
qualifications remain; zero integrity issues does not remove them.

## Recommended placement

1. Companion14, immediately after `Measurement failure taxonomy and audit`: use
   `tables/selected_table.tex` and `tables/all_frozen_attempts_table.tex`, with the
   explicit date retained. Use `companion14_proposed.tex` as a prose starting point.
   The first table has 537 model slots per version; the second has repeated frozen
   **record** units. Never sum the two views or call the historical counts unique
   benchmarks, actual device-JIT attempts or independent successful GPU processes.
2. Current sentence “Failed attempts retain raw counters and source bindings” is
   too broad for attempts stopped before a target/counter file exists. Prefer:
   “Available counter fragments, logs and captured bindings are retained; missing
   evidence remains explicit.” The source snapshot records absence rather than
   inventing a zero-length scientific observation or a cause.
3. Main: optional `main_optional_proposed.tex` method sentence only. Transient
   measurement availability is not an abstraction success ranking. Do not insert
   these dated counts as final results while the completion manager is running.
4. If the final manager emits a newer clean failure census, regenerate to a NEW
   packet output via prepare_packet.py. Update manuscript date/counts from its own
   exact input hashes, rather than editing table cells or treating this snapshot
   as the live census. Preserve this old packet.

## Exact distinctions and caveats

- Selected counter records: CUDA360 / CUDA.jl344 / KA341 / JACC342 (sum1387).
  Latest frozen counter records:358/338/336/336 (sum1368). All19 category differences
  are selected valid observations with a later failed retry, listed in
  tables/selected_latest_differences.csv. A failed new retry does not automatically
  invalidate an unchanged, correctly bound older selection.
- Historical 431/434/434/433 records include repeated attempts and the one KA
  CPU-only Heartwall derivation recovery. “1,732 profiling executions” would be
  incorrect. Actual *recorded new launcher* units: profiler420/425/421/422,
  build421/35/34/35. A launcher can have multiple targets/ranks/compilers; internal
  device-JIT and total child-process counts remain null/unknown.
- Forty-three frozen records have null profiler returncode:11/9/12/11. These are
  records without that process observation, not automatically missing raw logs,
  missing executions, compiler failures, zero work or evidence of a bug.
- Native has more recorded build launchers because this observation policy invokes
  native compilation often; do not infer native compile difficulty or compare it
  with Julia device compilation. No causally normalized compilation experiment
  was performed by this census.
- The compact “Build/helper/I/O/checker diagnostic” row combines six raw categories
  only for readable tables. Exact raw labels survive taxonomy_detailed.csv and
  codebook.json. The observations include a nonzero host checker child, permission
  text or wrapper rejection, but do not identify the unique underlying cause.
- Categories are mutually exclusive **exporter priorities**, not an exhaustive
  causal ontology. A retained record can have several raw diagnostic signatures;
  the full source all_frozen_attempts.json retains them. Do not add extra signature
  counts to the displayed exclusive categories.
- Pending/active475 and plan gates56 remain in the current denominator; no completed-
  cases-only percentage. Historical attempts have no “pending” placeholder rows,
  so that table is not a replacement coverage census.
- Hardware20 means five prerequisites propagated across four model assessments,
  not20 distinct hardware defects or framework limitations. Replay holds remain
  method constraints. Runtime N/A7 means no profiled SM kernel under that selected
  source contract, not failed numerical translation or observed zero memory/FLOPs.
- The scalar-FP counters exclude integer/address/control/special-function and some
  atomic arithmetic. An accepted counter record does not itself mean a qualified
  full benchmark roofline coordinate, complete elapsed ROI or causal bottleneck.
- No translation rejection/device-JIT failure rate is reconstructed from profiler
  process failures. Historical attempt counts, source repairs, current translation
  compiler loops and operational profiling recovery are different phases/units.

## Tokens and censoring

The source census carried no embedded shared-usage receipt at generation time;
its later closed raw boundary receipt is separate at
`measurement_failure_census/usage_receipt.json`. This paper-support phase has its
own prospective begin/end receipt delivered after closure. Neither packet infers
per-model consumption or adds token windows into a campaign total. Counts include
inherited/reused session context; cached input and reasoning output are subsets,
not extra consumption. Unknown historical per-target/billed-token quantities stay
unknown. A counter snapshot's attempt chronology is not a session billing ledger.

## Reproduction

```
python3 prepare_packet.py \
  --snapshot /tmp/roofline-completion-20261021/measurement_failure_census/final_snapshot \
  --output-root /tmp/NEW_FRESH_FAILURE_PAPER_TABLES
```

The generator reads frozen JSON/CSV only, does not import profiling libraries,
query a GPU, compile source, run benchmarks/tests or alter the source snapshot.
TeX uses booktabs; tables fit companion-wide placement (use table* if necessary).
No percentages or unknown-as-zero device/compiler counts are generated.

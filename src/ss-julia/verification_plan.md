# SS prospective full-source contract

Primary executes the exact recovered OpenCL1.2 text (707706 bytes), source pattern
`clEnqueueNDRangeKernel` (22 bytes), and all20000 repeats. Named R1 controls run
source naive mode on the original text and a dense one-miss input, overlapping
17-byte matches, stage1/stage2 rejection workloads, and mixed-case matches at
thread/workgroup/text boundaries. Source `ENABLE_2ND_LEVEL_FILTER` remains on;
CLI supports pattern length1 or >16, not short patterns2–16. Byte offsets and ASCII
case folding are intended, not Unicode character positions or biological search.

Before GPU study captures, declare exact integer acceptance (atol=rtol=0): full
actual input-array readbacks and deterministic initial output state/guards agree;
all group counts and every complete source-final sorted match position equal an
independent every-start ASCII naive oracle. No count-only or sampled acceptance.
Each raw physical append slot must map bijectively to one unique `(group,position)`
key in its source interval and satisfy the complete pattern; every expected key
must appear exactly once. Retain all full raw pre/post device allocation bytes,
source-final sorted positions, and the complete inverse permutation; reconstruct
the raw payload from canonical values/permutation/initial inactive sentinels and
require byte identity. Check every guard and inactive output slot separately.

The original `verify` itself compacts and sorts its complete valid results. The
primary numerical outputs are those source-final values and complete group counts,
with `raw_output_identity_claim:false`; raw atomic append ordering is not claimed
identical. This is complete keyed/source-final numerical agreement under the
current guide, distinct from raw identity and property-only membership summaries.
Do not serialize GPU work or hide raw value differences through a lossy mapping.

Conditional native reference: original initcheck observes uninitialized inactive
result-buffer D2H copies (88458errors, original validmatches36 stillpasses). Initialize
all output payloads before each full source execution and add aligned guards,
outside ROI. An executed CPU ASan witness proves source `memcpy` compaction
can overlap on legal dense per-group counts; owned `memmove` is the minimal host
repair. GPU `kernels.cu` stays byte-identical. Preserve original CLI/logs/source and
both witnesses. Set `native_reference_repaired:true`, canonical nonempty provenance
and exclusion from untouched-native aggregates before measurement. No observed
wrong original GPU search positions or framework limitation is inferred.

All models execute original file I/O, CPU Boyer–Moore–Horspool reference, full
selected GPU repeat loop, and source final compaction/sort/check. Native host
controller/C ABI is retained where used, with all actual selected kernels authored
and dispatched by the target framework, never native fallback. Match original
256-thread blocks and131072search positions/workgroup, source dynamic pattern
storage/static queue/counters and source synchronization (backend extensions
explicit). Actual source case/region/precision/API capabilities are documented.

ROI is the original synchronized whole repeat loop including host launch/C ABI
control and completion; allocations, deterministic initialization/guarding, input
readbacks, independent checks, raw capture and source CPU final sort are outside.
Full-source warmup before five complete checked executions; ignore only trial1's
elapsed time and average2–5. PrimaryR20000 plus namedR1 controls remain separate
per-case timing fields and summed common campaign timing. Preserve all5times and
any failures; no outlier removal or replacement of accepted series.

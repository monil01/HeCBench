# V100 roofline campaign: 537 benchmark names, four models

The [campaign protocol](protocol.json) and [progress ledger](PROGRESS.json) are
maintained by the coordinator. Read [POLICY.md](POLICY.md) before profiling;
[ledger_schema.json](ledger_schema.json) describes a reviewed stage record. The
ledger covers CUDA, CUDA.jl, KA and JACC for every name, including explicit
not-applicable, hardware-blocked, failed and incomplete-counter statuses. It does
not require a fictional scalar-FP point for every slot. The five existing native
hardware barriers remain unverified.

## Classifier

`tools/v100_roofline_classify.py` is a standard-library CPU-only parser for Nsight
Compute 2022 raw CSV (wide/long). It emits per-kernel scalar precision projections,
raw FP16/FP32/FP64 thread instruction counts, tensor-pipeline presence, direct
DRAM/L2 bytes, exact source-CSV/parser SHA256, and clock-adjusted modeled ceilings.

```bash
python3 tools/v100_roofline_classify.py \
  --csv PATH/counters.csv --context PATH/reviewed_scope.json \
  --output PATH/classification.json
```

The context binds actual executed source/config/result identities, semantic stage,
ROI membership and coverage; a generic kernel name is insufficient. Output
`schema_version:1` describes the per-kernel parser report; the ledger schema is
the enclosing scientific/provenance record, not an assertion that the report alone
contains every required stage fact. Root attaches and reviews the envelope.

FP16 conversion/peak is gated by `--fp16-lanes 1|2` plus
`--fp16-convention-evidence PATH_OR_ID`; do not select a factor without validated
V100 instruction/metric evidence. Tensor presence does not supply HMMA FLOPs.
Missing metrics remain null, zero traffic yields undefined DRAM intensity, mixed
classes remain separate projections, and no causal bottleneck is inferred solely
from `roofline_regime`. No equal-weight whole-benchmark averaging is performed.

The validated first-per-launch-configuration selector is
`--filter-mode per-launch-config --kernel-name-base mangled --kernel-id '::regex:.*:1'`. It is a sampling
selector, not complete-ROI evidence. Preserve original five checked trial times
and their mean runs 2–5; profiler replay duration never replaces them.

## Review accounting

[shared_review_usage.json](shared_review_usage.json) records observed shared
methodology review counters with the initial guide/script exposure explicitly
left-censored. Integration has its own window. Neither cost is translation effort;
cached-input and reasoning counters are subsets, not extra tokens. No GPU, Git,
sudo, OS mutation or primary-source changes occurred during review/integration.

## Live collection and audit exports

The serial `tools/v100_roofline_campaign.py` controller stages one benchmark at a time with `v100_roofline_prepare.py`, runs `v100_roofline_collect.py`, then losslessly freezes finished evidence with `v100_roofline_summarize.py`. Resume the recorded controller before starting another. The current collector PID/session is in `.porting-state/portable_julia/active_work_537.json`; the joint paper checkpoint includes its snapshot.

- `model_audit.csv`: all 2148 slots, selected configuration, immutable primary mean, profiling/coverage/bottleneck status and evidence link.
- `kernel_audit.csv`: every sampled kernel, including non-floating work; raw precision counts, DRAM/L2 traffic, replay time and stage qualifiers.
- `kernel_roofline_points.csv`: scalar per-precision DRAM projections; each row is a kernel sample, never a benchmark average.
- `kernel_hierarchical_points.csv`: separately labeled DRAM/L2 projections when corresponding counters exist.
- `kernel_tensor_roofline_points.csv`: separately typed FP16 input/FP32 accumulation tensor projections, only with prospective source/dependency approvals; target SASS observation remains explicitly qualified.
- `retained_roi_aggregates/`: complete source-counted scientific-family sums for admitted configurations; FLOPs/bytes and summed replay times are distinct from elapsed host ROI.
- `profiles/`: losslessly compressed raw counters/logs/plans/provenance and per-attempt manifests; later collectors also snapshot exact collector/parser sources. Historical profile attempts remain preserved.
- `measurement_events.jsonl` and `measurement_effort.json`: profiling infrastructure failures and shared inference accounting, separate from translation failure/cost records.

Readiness is versioned: initial1842 ready/286 blocked; hash-bound isolationv4 with512MiB input copies gives2013 ready/115 blocked. The full controller allows1024MiB per-model assets under a2GiB free-space reserve; readiness still does not certify a valid scientific ROI. Fixed-output, custom-build and multi-GPU cases stay blocked until their exact isolation/measurement method is reviewed.

## Evidence audit

Run the CPU-only audit under the campaign metadata lock:

```bash
python3 tools/v100_roofline_audit.py \
  --source-root /home/imo/HeCBench_julia \
  --output ontology/studies/v100_roofline_537/integrity_audit.json
```

It verifies compressed and raw manifest hashes, selected contract bytes, exact
collector/parser versions, and recomputes historical kernel derivations. It also
checks selected observation hashes and all 537 names / 2148 model slots. The
audit does not prove ROI membership, numerical correctness of disposable profiler
outputs, or historical source identity. Earlier captures without explicit PCI
checks retain that qualification; later raw rows must match the GPU UUID-to-PCI
mapping. Future builds and profiles use a child-only filesystem write guard; no
unrestricted fallback is permitted.

`atomicSystemWide` is held from kernel replay: CPU and GPU concurrently update
the same managed allocation. Its translation/property verification is unchanged;
a separate application-replay or single-pass measurement method is needed.

## Live execution safeguards

Resume the recorded collectors and CPU export supervisor from
`.porting-state/portable_julia/active_work_537.json`; do not duplicate the full
queue. All GPU work takes the shared lock, waits for an idle GPU and monitors
foreign GPU activity during profiling. On contention, terminate only the owned
fresh profiling session and retain the failure; never signal another workload.
Wait for owned descendants to exit before hashing outputs. A timeout enlarged
from the saved runtime is an operational limit, not a runtime prediction.

Select the best qualified capture per model, while exposing the latest attempt
and preserving all failures separately. Source-counted retained-trial selectors
require exact source/config/result bindings and an observed launch-count match.
They describe a fresh profiled scientific window; they do not turn cold replay
into the historical unprofiled trial or establish elapsed host ROI throughput.

Check the memory domain before assigning a bandwidth limit: host-mapped primary
data (including `zerocopy` and `halo-finder`) need host/PCIe analysis, so an HBM
roofline bottleneck prediction is not applicable. Keep raw DRAM/L2 diagnostics.
Tensor conversion requires a validated metric convention and a source-bound
per-kernel arithmetic-mode approval; record whether executed SASS was observed.
Keep scalar epilogue work and tensor work separate. Missing or unreviewed modes
remain unconverted rather than inheriting another kernel's peak.

Run `tools/v100_roofline_refresh.py` for frozen evidence integrity checks,
precision-aware figures, ontology and compact paper exports. An integrity audit
checks hashes and derivations; it does not prove scientific correctness, source
ROI membership or causal bottlenecks. Queue completion never implies complete
scientific analysis. Keep raw evidence outside the paper repository.


## Resume and interpret partial coverage

`PROGRESS.json` is the current measurement census; `NATIVE_POPULATION_AUDIT.json`
confirms that all 537 native CUDA directories are represented. Presence in the
registry does not mean a measurement completed. Inspect qualified-completion,
all-four-model and retained-phase counts separately. The live controller list and
sessions are in `.porting-state/portable_julia/active_work_537.json`; the periodic
CPU supervisor updates integrity, figures, ontology, publication and checkpoints.

Some first-specialization/geometry captures include warmup and source candidates.
A typed tensor approval confirms a physical work convention, not timed-region
membership. Partial captures, unavailable arithmetic modes, unsupported replay,
foreign-GPU contention and hardware barriers remain explicit. In particular,
SLU has a bounded first-specialization followup after expensive full-geometry
replay; that sampling change cannot certify an entire factorization trajectory.

No kernel samples are equally averaged into a benchmark classification, and the
geometric region is never silently promoted to a causal bottleneck. Preserve
primary five-run timings and mean runs2–5. New profiler failures are measurement
failures, separate from accepted translation results and compiler-feedback cost.

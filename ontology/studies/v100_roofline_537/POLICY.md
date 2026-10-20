# Precision-aware V100 roofline review

Read-only review, 20 October 2026. No GPU executions, source changes, ontology
changes, Git operations or OS mutations. Shared review effort is separate from
translation; the initial guide/profiler reads precede the prospective meter and
are left-censored. This campaign policy does not itself establish new measured roofline positions
or bottleneck diagnoses.

## 1. Denominator and outcome contract

Maintain a 537 x four-model assessment ledger (2148 intended slots), not 2148
mandatory plotted points. Every slot has a recorded status and reason: eligible
FP kernel, eligible mixed-stage component, non-FP diagnostic, no device kernel,
execution-only/no useful numerical work, hardware-blocked, profiling-failed,
unsupported counter, unreviewed mapping, or incomplete coverage. A language front
end with zero authored GPU functions may execute substantial vendor GPU work:
profile that work, preserve zero authored-function count, and identify the actual
DSO/kernel/backend. Conversely runtime allocation/free or an empty/assertion launch
does not create useful floating-point work. Never use a zero placeholder to
represent an unavailable point.

Use the selected current result/effective config and qualification, not directory
names or generic traits. Corrected-native, known-native-defect, adapted, property,
revised-acceptance and multidevice strata stay distinguishable. Existing 532
accepted names comprise 522 named numerical, five property and five execution;
the five native-hardware outcomes remain unverified/unprofileable on V100 unless
new actual evidence changes readiness. Full-corpus profiling does not extend the
scientific scope of any accepted numerical contract.

## 2. Kernel and ROI identity: prerequisite, not optional metadata

First obtain an ordered launch census for each executed case and measured ROI.
Bind configuration ID, selected result/config SHA, actual executed source/dependency
hashes, library realpath/version/SHA, device UUID, stage ID, semantic operator,
launch ordinal/iteration/case, grid/block/dynamic shared bytes and called stream.
Hash command, environment and profiler tool/version. Generic Julia names such as
JACC parallel_for are NOT semantic identities. Match source call site plus launch
ordinal/geometry and expected sequence, not just regex. Exclude initializers,
readback/audit kernels and JIT/setup outside the declared ROI. Some source fills,
memsets and copies ARE inside the timed ROI; preserve them in the census as such
but do not assign floating-point work to them.

For each scientific stage enumerate all launches and variable input/size/iteration
classes. A sampled launch is a sampled launch, with actual coverage declared.
Multiplying its counters by iteration count requires a separately justified
stationarity/work equivalence assumption. Adaptive iterations, active queue sizes,
thermodynamic branches, sparse occupancy and stochastic paths can invalidate that
assumption. Prefer measured full selected range or explicit early/mid/late samples
with coverage flags. First-warmup-only profiles are not steady-state/full-workload
profiles. A stage mapping can be one-to-many across front ends; identical function
names or counts are not required, semantic workload equivalence is.

Keep three clocks separate: unprofiled primary five-trial host ROI (mean runs 2–5),
profiled per-launch device duration, and profiled full-range elapsed duration.
Compute counter-derived rates only with the duration of the same measured scope.
Do not divide one kernel's FLOPs/traffic by full host time, sum times as an elapsed
range when streams overlap, or interpret a sum of isolated replay durations as
application time. Sum traffic/work across an exactly covered serial kernel group
and divide by that group's sum of profile durations for a clearly labelled
isolated-device aggregate. Full workflow overlays require actual full-range time
and coverage of transfers/host/vendor work, never a kernel-only bound on host work.

## 3. Work precision and counters

Record vectors of timed arithmetic: scalar FP16 lanes, FP32, FP64, tensor operations
by input/accumulation mode, integer/bitwise/address/control operations, atomics and
special functions. Storage precision, arithmetic precision, host calculations,
untimed fixture/JIT/RNG/oracle precision, and artifact encoding are separate roles.
Float64 serialization does not establish FP64 execution. Int32 serialized BF16 is
not integer arithmetic; FP16 storage does not establish FP16 accumulation or tensor
execution. CUDA API compute modes describe API intent, not proof of the emitted
instruction pathway. V100 does not acquire TF32/FP8 tensor hardware because a newer
API accepts a type or falls back to scalar conversion.

Use exact available architecture/tool metric names from installed query/section
outputs and record formulas. Predicated-on THREAD instruction counters are the
basis for scalar FLOPs: add + mul + 2*FMA at the appropriate precision. Do not
multiply thread counters by warp width again. Packed half2 counts need two active
lanes per scalar instruction; verify metric semantics before scaling. Tensor HMMA
counts require the actual Volta instruction shape/step and thread/warp metric unit;
never apply one Ampere formula or multiply a pre-derived FLOP metric twice.
Keep executed-hardware FLOPs separate from independently/source-derived useful
algorithm FLOPs (padding, recomputation and conversion differ). Complex arithmetic
is real operations of the executed precision, not one complex operation = one FLOP.
Explicitly list whether divide/sqrt/exp/log/reciprocal and SFU work are omitted from
the conventional add/mul/FMA ordinate. Their omission can make an apparent low
attainment mathematically correct but unsuitable for a compute-bound conclusion.
FP64 atomic addition deserves its own serialization/atomic throughput evidence;
a DFMA counter alone will not represent its cost.

Memory tiers use their own measured byte counters: DRAM read+write, L2 transactions,
and, if supported, L1/shared/local transactions. Never place L2 bytes under a DRAM
bandwidth ceiling or source theoretical bytes under a measured-traffic label.
Use direct sum byte counters when available; converting rate*time requires recorded
unit normalization and same-scope metrics. Transaction sectors are not useful
payload bytes, and local traffic includes private arrays/stack as well as possible
spills. Zero DRAM bytes => undefined/infinite DRAM intensity, not zero intensity;
retain a nullable point and another measured memory tier. Zero work => no log FP
roofline point. Missing/unsupported counters remain null, never zero.

## 4. Ceilings and classification

Record the exact V100 SKU/UUID/SM count/ECC, tool/driver/library/runtime, observed
clocks and power/thermal state. Record ceiling kind: datasheet theoretical,
Nsight modeled peak-sustained, or separate empirical calibration. These are not
interchangeable. No sudo, clock-setting privilege requests or OS changes are needed.
Use one explicitly selected ceiling protocol per comparison, with per-profile
clock-sensitive variants disclosed instead of quietly changing the ridge by model.
Query the installed section definitions and retain metric units/formulas. Existing
summarize_roofline.py uses a derived ffma x2 metric: in the inspected saved CSV its
10240 is 2*5120 peak FMA instructions/cycle, NOT this kernel's executed operation
count. Preserve that role and validate it from installed tool metadata before reuse.
The fixed useful-FLOP FP32 triad summary is not a generic mixed-precision counter
summarizer.

For homogeneous scalar class p and memory tier L:
I_p,L = measured executed FLOPs_p / measured bytes_L;
P_p = measured executed FLOPs_p / same-scope duration;
R_p,L = C_p / B_L; attainable envelope = min(C_p, B_L*I_p,L).
Store absolute values, attainment and I/R with provenance. Label placement
memory-side when I<R and compute-side when I>R, with a declared near-ridge band
(e.g. [0.8,1.25], a display convention rather than a statistical confidence band).
This is geometric positioning, NOT an actual memory-bound/compute-bound diagnosis.
A point far below the envelope may be latency-, dependency-, launch-, atomic-,
shared-bank-, branch-, SFU-, occupancy- or host-limited. Diagnoses require independent
counters and preferably a controlled intervention, with evidence/confidence links.

For mixed precision provide separate stage/class facets and the full operation
vector. Do not sum FP16+FP32+FP64 FLOPs and compare to whichever peak is largest.
Per-class projections are upper-bound diagnostics because other classes also cost
time; a single combined mixed roof needs a documented instruction/resource model.
Do not add the independent maxima of pipelines that share issue resources. For
mixed scalar/tensor, record both separately and validate nonoverlap/double counting.
Integer/crypto/string/graph work gets measured bytes/s, instructions/s, atomic
rates and latency/resource plots; no fabricated FP32 operation count. Small FP
substages in an integer-heavy search can have a component roofline, never an
unqualified FP roofline label for the whole benchmark.

## 5. Replay and coverage controls

NVIDIA documents that kernel replay flushes caches by default, can serialize or
change concurrency, and can fail for kernels requiring host responses between
passes. Specify replay mode, cache/clock policy, pass count, warmup/input snapshot,
and any serialization. For asynchronous host/GPU queues, persistent tasks, mapped
flags and Hogwild training, choose an actually supported application/range strategy
or mark profiling unsupported-for-this-contract; don't deadlock or serialize the
algorithm just to get a point. Preserve raw property-model differences. A full
application replay can also change endogenous schedules, so retain run identity,
output/property verdict and measured counter variability.

MultiGPU contracts need per-device profiles plus host/communication coverage.
CUDA_VISIBLE_DEVICES=6 single-V100 work is not a profile of a two-GPU native ROI.
Either root authorizes and records all devices or assess the slot with a clear
multi-device scope gap. Never present GPU6 counters as the complete multidevice job.

## 6. Existing script issues requiring root attention

- profile_reverse.py removes JACC name filtering entirely: first launch may be a
  fill/initializer. Keep as historical exploratory evidence, not target identity.
- profile_scientific_case.py and profile_shmembench.py are explicitly first-match,
  launch-count1, generally warmup profiles; generic parallel_for name is insufficient
  for full537 stage selection. Scientific script also restricts only three cases.
- profile_shmembench changes1000repetitions to1 and runs from source cwd; label the
  diagnostic and consider frozen-output locations rather than touching source cwd.
- summarize_roofline.py selects rows[-1] without checking kernel count/unit row,
  assumes nanoseconds, only FP32 useful work, computes traffic from rate*duration,
  and has no zero/missing guards. The version1 parser tools/v100_roofline_classify.py is unit-aware; preserve
  preserving this historical triad tool rather than silently changing old results.
- Generic top MMCSF traits still say planned/noacceptedtimings; selected mixed-
  precision configuration overrides/actual executed sources are authoritative.
- Hash validation must target actual nested executed variants and all declared
  dependencies; discover() canonical-child digest alone can bind an older primary
  while commands execute a qualified secondary folder.

Primary references (reviewed official documentation):
https://docs.nvidia.com/nsight-compute/ProfilingGuide/
https://docs.nvidia.com/nsight-compute/NsightComputeCli/
https://docs.nvidia.com/nsight-compute/ComputeTriage/
https://docs.nvidia.com/nsight-compute/NsightCompute/index.html

## 7. Validated selector and campaign checkpoint

Root validated `--filter-mode per-launch-config --kernel-id '::regex:.*:1'`:
the canary captures four rows for two kernel names with two grids. This selects
the first invocation per kernel/launch configuration, not a complete workload.
Keep the exact selector, full raw output and tested tool version with each profile.
A kernel name/grid may still represent several semantic stages, and a first
invocation may be warmup; assign stage identity and ROI membership independently.

Every joint checkpoint records all 537 × four status denominators, configuration/result
identities, selected and unassigned stages, launch/ROI coverage, precision/counter
formulas and missing metrics, profile/raw report paths and SHA256 values, model
source/environment/library bindings, GPU/tool identity, failed attempts/replay
restrictions, actual profiler durations and separate immutable primary five-run
means. Include open shared profiling effort windows, storage locations/retention
needs, next actions and paper/ontology export state. Root owns runner/progress and
measurement metadata; this policy/parser never overwrite primary timing/results.
The current authority remains the linked three main guides; translation completion
and frozen 447/release histories do not change when the profiling campaign starts.

## 8. Recorded sm70 arithmetic calibration

The source/counter/SASS receipts in
[calibration/v100_sm70_20261020/review.json](calibration/v100_sm70_20261020/review.json)
validate the V100/Nsight2022.4.1 physical packed conventions: HADD2 and HMUL2 each
count two scalar lane operations per predicated-on thread instruction; HFMA2
counts four. Each full4096-thread/128-step canary produces524288 corresponding
thread instructions. Scalar-half source also emits HFMA2, so the unused upper
lane remains physical executed work, not useful source work. The original half
canary did not collect HADD/HMUL counters; those missing values remain unknown,
even though the new separate add/multiply canaries establish the conversion.

The FP16-input/FP32-accumulation WMMA16x16x16/128-step canary yields1048576 scalar
FLOPs and2048 tensor-pipeline instructions:512FLOPs per measured instruction.
CPU SASS review confirms HMMA.884.F32.F32.STEP0--3. The parser requires an explicit
factor/evidence argument and sm70 context before enabling this tensor convention;
record target precision-mode review separately from the calibration scope.
Other architectures or unreviewed tensor instruction mappings remain unknown.
Tensor and scalar projections have separate ceilings; no aggregate peak is added.
All physical-work conversions are explicit opt-ins, not retroactive edits to old
profiles or an inference from storage/API types.

The parser additionally reports distinct L2 projections using measured
`lts__t_bytes.sum` and modeled peak
`lts__t_bytes.sum.peak_sustained * lts__cycles_elapsed.avg.per_second`.
Missing L2 peak/clock values leave that envelope unknown. Observed integer thread
instructions and SM/DRAM utilization are retained as evidence; neither utilization
nor geometric position alone assigns a causal bottleneck. Calibrated CLI flags
are `--fp16-lanes 2 --fp16-convention-evidence RECEIPT` and
`--tensor-flops-per-instruction 512 --tensor-convention-evidence RECEIPT`, with
`gpu_sm70: true` in the attached context. Original callers remain valid; old
immutable derived reports are not recomputed automatically.

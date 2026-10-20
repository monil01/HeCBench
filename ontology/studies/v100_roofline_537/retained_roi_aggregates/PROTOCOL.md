# Source-counted retained-trial kernel aggregation (version 1)

## Scope and acceptance

This exporter accepts only the declared triad N4194304/R100, BabelStream
N262144/dot/R10, and Jacobi N2048/20-step configurations. It uses the current
selected capture, never substitutes a failed latest attempt, and never scales
one sampled kernel by a source repeat count. A complete family requires an
unchanged source/config/result/selector recipe, successful whole profiler
process and original dual-reference numerical receipt, five fresh positive CPU
self-check rows, a source ordinal window for retained trial2, the exact expected
number of distinct raw kernel identities, and explicit per-kernel timed-stage
membership annotations. Source-ordinal proof is not a captured whole-run trace.
Preserved raw counter/manifest/selector/source hashes make each admission auditable.

## Allowed totals

For all actual selected scientific launches i, retain precision separately and
compute F_p = sum_i F_(i,p), B_DRAM = sum_i(read_i+write_i), B_L2 = sum_i L2_i,
and T_device_replay = sum_i duration_i. Scalar FP32/64 work is add+mul+2*FMA;
FP16/tensor work may not be inferred without calibrated arithmetic conventions.
This limited exporter rejects nonzero half/double/tensor arithmetic rather than
silently adding unlike peaks. Conventional scalar counts exclude atomic/SFU,
integer/address/control work. Kernel multiplicity is counted once per unique
raw identity; profiler hardware replay passes are not extra scientific launches.

The integrated work/traffic coordinate is I_p = F_p/B, **not the mean of kernel
intensities**. Device replay rate F_p/T_device_replay is labeled as such; summing
individual launch rates or equal-weight averaging rates is invalid. The source
runs these cases serially. If future stages overlap, summed duration is device
service time, not union busy time, makespan, or elapsed time. Raw cold-replay
traffic is not the uninstrumented warm-cache trajectory traffic. Complete
source-counted kernel work under this replay protocol can be shown independently
of any claim about full elapsed ROI behavior.

## Timing and ceilings

Original five checked trials remain unchanged: mean trials2..5 is reported per
internal repeat/step. To recover the declared host timer interval, multiply
reported seconds by source repeats (100/10/20). This host interval includes
launch/control/synchronization; Jacobi also includes scalar D2H, host sqrt and
error reset. Fresh profiler trial2 host clocks include collection/replay overhead.
Neither original nor instrumented host time may be used as the denominator for
these new replay counters. Show original means, fresh clock interval and summed
kernel replay duration as separate quantities. Their difference is not a pure
host overhead estimate under profiling.

Keep per-launch clock-adjusted modeled ceilings/ridges as ranges and per-launch
geometric regimes. Do not average ceiling FLOP/s, ridge, intensity or occupancy
into a benchmark diagnosis. No aggregate geometric regime or causal bottleneck
is asserted here: causal bottleneck remains unknown. For a future fully mapped
serial homogeneous stage, a clearly named model may form sum_i max(F_i/P_i,
B_i/BW_i) as a per-launch ideal-service lower bound; that is not an observed
elapsed-time roof or a measured sustained device peak. Mixed precisions require
separate work vectors/projections, not summed peaks.

## Completeness boundaries

Triad's named study contains only 100 serial scientific triad launches within the
host timer; source stream/tile/transfer benchmark is excluded. A complete family
therefore covers the named scientific device work, not the original full CLI or
elapsed-host ROI. BabelStream here means dot only: not six STREAM operations or
host final scalar reduction. N262144 working set fits L2; cold NCU DRAM traffic
must not be called normal STREAM bandwidth. Jacobi's 20 stencil calls may cover
all selected stencil computation, but timer-included resets/memsets, transfers
and host norms remain outside that counter family. Atomic accumulation also has
cost/work not represented by scalar add/mul/FMA totals. Neither complete stencil
family nor a memory-side projection proves full-solver/causal bottleneck.

Original accepted numerical outputs and dual-reference receipts are separate
from new profiler self-check logs. Disposable new outputs were manifest-recorded
then removed by the collector; no new direct-artifact correctness certification
is inferred. Old rejected/partial captures remain unchanged and unextrapolated.

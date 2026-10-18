# Sixteen-field AoS versus SoA accumulation

Both original native kernels are copied verbatim into native_kernels.inc,
with unchanged NUM_ELEMENTS4096 and COUNT4096. All versions preserve physical
AoS16fields/record and SoA16planes with4096capacity, UInt32modular arithmetic,
and16explicitfield updates periteration. CUDA.jl/KA/JACC share genuine
source-equivalent generated field updates, one authored kernel withtwo Val
layout specializations. Historical Julia only timed a dummy Float32touch
and is preserved, not treated as numerical reference evidence.

Two varied full-width fixtures (including overflow and bit-position patterns)
and active counts4096/4093 give8paths. All20full1,573,632-word artifacts are
bit-identical. Each has1,049,088actual source/initial-destination prefix words,
then524,544complete output/guard words. Everyfield, inactive tail andreadonly
source is checked by independent initial+4096*source modulo2^32, notanother
4096-step loop. All includes and fixture files are automatically hash-bound
before execution, including arbitrary .inc/.bin extensions.

Each path has a separately timed one-kernel launch plus context completion;
reset/transfers/setup/readback/oracle are outside. Aggregate is sum8windows,
allfivechecked trials, mean2..5; perpath rawtimes preserved. Sourcefullcapacity
and COUNT are retained, with a separately named legal4093-active tail. No
other CLI repeat regime is certified. Source timing is host-inclusive kernel
region, not event-only latency. Compiler alias/lowering/resource differences
are not isolated by semantic matching. No dynamic traffic/instruction/occupancy
profiles or measured roofline position are claimed. Small per-path working
set and repeatedupdates must not be labeled DRAM bandwidth from source bytes.

On the named cases the SoA mean is8.49--8.78times faster than AoS acrossall
four models. These are within-workload observed layout differences, not a
universal model ranking or a demonstrated bottleneck mechanism.

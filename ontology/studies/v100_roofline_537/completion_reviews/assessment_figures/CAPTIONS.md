# Suggested figure captions

**Coverage assessment.** Dated partial-campaign snapshot of 537 benchmark names
for each of CUDA, CUDA.jl, KA and JACC. (a) Model-specific counts distinguish at
least one observed FP geometric position, no conventional FP work counted,
source-reviewed inapplicability, hardware prerequisites, and pending or failed
profiling. Unknown categories remain explicit. (b) Each cell gives the number
of names with a qualified position divided by its scientific-family denominator;
the second line counts unmeasured or unresolved names. Known no-FP and source-N/A
cases are not treated as missing measurements. Positions can come from partial,
first-invocation or source-unassigned samples; coverage does not establish a
complete GPU/host ROI or a causal bottleneck. The frozen descriptor has no
source-N/A entries; the plotting code supports those future classifications.

**Integrated scientific-family intensity.** Ratio of summed executed FP32
add/multiply/FMA work to summed measured DRAM bytes over the complete selected
scientific kernel family: 100 Triad launches, 10 STREAM-dot launches and 20
Jacobi stencil launches per model. All 12 captures pass the strict source-counted
family audit. These are cold kernel-replay measurements, not mean kernel
intensities, ordinary elapsed-host throughput or causal bottleneck labels. STREAM
covers dot only; Jacobi's timer-included resets, scalar transfers and host norms
are outside the stencil counter family. Conventional FP totals also exclude
integer, special-function and floating-atomic costs. No speedup correlation is
inferred, and the older controlled Triad roofline plot remains separate.

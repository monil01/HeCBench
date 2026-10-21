# Why the three scientific families have different measured intensities

## Scope and evidence

This review covers the twelve already-admitted source-counted trial-2 families:
Triad (100 launches), STREAM dot (10) and Jacobi stencil (20), in CUDA, CUDA.jl,
KA and JACC. All use the existing successful captures and strict numerical
receipts. The analysis binds the strict audit, original source/environment files,
selected frozen manifests, raw counters and compiler/profiler provenance. It
recomputes the instruction-class totals directly from the archived CSVs.

**Evidence levels:** raw counter observations; exact source arithmetic accounting;
current pinned dependency-source inspection; and unresolved hypotheses are kept
separate. No selected-instance SASS/PTX/cubin is listed in these profile manifests,
and the selected scratch executables are no longer retained. Thus this report
has instruction-class counter evidence, **not an inspected disassembly**. No
compiler, JIT, GPU run, test, benchmark or shared-file modification was performed.

## 1. Jacobi: the entire 356.52M difference reconciles with reduction mapping

All versions update the same 2046 x 2046 interior for 20 fixed steps and compute
pointwise changes plus one convergence diagnostic per step. This is the named
fixed-step case, not full convergence/time-to-solution.

| Selected scalar work | Exact counted operations over 20 steps |
|---|---:|
| Four-neighbor averaging: 3 additions + 1 multiply | 334,889,280 |
| Pointwise difference and square: 1 subtract + 1 multiply | 167,444,640 |
| Common per-point subtotal | **502,333,920** |
| Native first reduction: 4 shuffle/add rounds on all 2048² threads | 335,544,320 |
| Native second reduction: 4 add rounds on 16 threads x 16,384 blocks | 20,971,520 |
| Additional native conventional FADD | **356,515,840** |
| Native measured add/multiply total | **858,849,760** |

The raw counters close this census exactly. Native has 691,405,120 FADD and
167,444,640 FMUL; every Julia version has 334,889,280 FADD and 167,444,640 FMUL.
All have zero FFMA in these captures. Native source uses an 18 x 18 shared tile,
two shuffle/add reductions and one atomic addition per block. CUDA.jl uses direct
global stencil loads and a conditional atomic update per point with positive
squared change; KA and JACC retain that per-point mapping.

**The metric is incomplete for comparing total reduction arithmetic.** Its
ordinary FADD/FMUL/FFMA counters exclude floating atomic operations. Native's
327,680 source-declared block atomic calls are omitted, as are Julia's
data-dependent point atomic calls. The difference therefore describes counted
instruction classes and a reduction-mapping adaptation, not more useful stencil
updates, an arithmetic precision change, or a complete total-FP-work comparison.
The same scientific norm can be accumulated by differently counted primitives.

Consequently native's executed-class intensity (about 1.320 FLOP/DRAM byte) and
CUDA.jl's (0.770) have different numerators. `core_work_diagnostic.json` shows an
explicitly auxiliary common-work normalization: stencil plus pointwise squared
change only, divided by all selected kernel traffic. Native and CUDA.jl then
lie near 0.772 and 0.770; JACC is about 0.411. This does not replace the frozen
executed-counter points and still omits all norm-reduction arithmetic.

At source level, native declares 5,234,688 tile/halo input float loads per step;
the direct-global Julia expression references 20,930,580 input floats. These are
logical operand accesses, not measured transactions or guaranteed emitted load
instructions. Observed DRAM reads are nearly equal: native 337.745 MB and
CUDA.jl 338.090 MB over the selected family. This observation does not identify
which cache/coalescing mechanism accounts for the reuse.

## 2. FMA contraction changes instruction counts without changing weighted FLOPs

Triad has 838,860,800 counted FP32 FLOPs in every version. CUDA records
419,430,400 FFMA instructions and no separate FADD/FMUL; each Julia version
records 419,430,400 FADD plus 419,430,400 FMUL and no FFMA. A fused multiply-add
counts as two conventional FLOPs, so this is equal weighted work but different
arithmetic instruction composition. Full outputs satisfy the declared numerical
contract; bitwise identity or identical rounding is not inferred.

STREAM dot gives the same lesson. Each family performs 2 x 262144 x 10 product/
accumulation FLOPs plus 256 x 255 x 10 block-tree additions: **5,895,680** counted
FLOPs. Native records 2,621,440 FFMA and 652,800 FADD. Julia records 2,621,440 FMUL
and 3,274,240 FADD. These counter observations support contraction differences;
without retained disassembly they do not identify a particular compiler pass or
prove a throughput cause. STREAM here is the N262144 dot case, not the full six-
operator/source-size benchmark; its small working set and cold replay matter.

## 3. JACC traffic is higher mainly in the measured write component

Decimal MB, summed over actual selected launches (not averages of intensities):

| Case/model | DRAM reads MB | DRAM writes MB | Total MB |
|---|---:|---:|---:|
| STREAM dot CUDA | 20.988 | 0.001 | 20.990 |
| STREAM dot CUDA.jl | 20.988 | 0.001 | 20.989 |
| STREAM dot KA | 21.005 | 0.001 | 21.006 |
| STREAM dot JACC | 21.092 | **101.234** | **122.325** |
| Jacobi CUDA | 337.745 | 313.049 | 650.794 |
| Jacobi CUDA.jl | 338.090 | 313.880 | 651.971 |
| Jacobi KA | 338.040 | 314.229 | 652.269 |
| Jacobi JACC | 341.949 | **881.735** | **1223.684** |

JACC's total DRAM traffic is about 5.83x native for dot and 1.88x for Jacobi.
The excess is overwhelmingly writes, while its counted arithmetic equals the
other Julia versions. L2 transaction totals also increase. These are actual
hardware-transaction differences under the cold-replay protocol; they are not
an equal-weight kernel summary or evidence of higher output size.

### What the source does establish

- Dot's JACC implementation uses `JACC.shared(template)` with a 256-element
  Float32 zero template. Installed JACC 1.4.0 CUDA source copies that vector
  **from global input into dynamic shared memory**, then synchronizes. If the
  full copy is emitted per block, it declares 2,621,440 logical global-read bytes
  across the family, plus shared-memory writes. The compiler can optimize it;
  logical source bytes are not a DRAM prediction. There is no source-declared
  global template write, so this copy alone does not account for the 101 MB of
  extra observed DRAM writes. All four dot kernels write the same 256 partials.
- The source-level dot output is 10,240 bytes over 10 launches, while native
  measured DRAM writes are only 1,280 bytes. This alone demonstrates that DRAM
  interval write bytes cannot be equated to source store count x element size.
  Cache/writeback/profiling timing must remain part of the measurement scope.
- Jacobi JACC does not call `JACC.shared`. KA and JACC both use Atomix atomics
  (JACC re-exports that macro), with matching conditional per-point logic.
  Therefore per-point atomic granularity alone cannot explain JACC versus KA,
  and the shared-template explanation does not apply to Jacobi.
- The selected JACC launch sources explicitly request 1024 shared-memory bytes
  for dot and zero for Jacobi/Triad. The earlier unrelated default-48KiB launch
  confound is not an explanation for these explicitly configured captures.
- Triad JACC has essentially the same DRAM traffic as the other models, despite
  also using a generic JACC callback. Thus these data do not support a universal
  JACC callback/dispatch traffic tax.

### What remains a hypothesis

Compiler local-memory/spill traffic, atomic lowering and cache/writeback/replay
behavior are plausible discriminators, not established causes. CUDA documents
that spills can use local memory backed by device memory and that PTX/cubin
inspection or compiler resource reports can identify that use. That general
mechanism does **not** establish a spill in these selected JACC kernels.
See the [CUDA 12.0.1 local-memory documentation](https://docs.nvidia.com/cuda/archive/12.0.1/cuda-c-programming-guide/index.html#local-memory).

A causal claim would require selected-instance disassembly and local load/store,
spill and atomic-transaction evidence (or a separately declared matched lowering
control). Those data were not retained or collected here. No new experiment is
represented as completed. The report does not assign a memory/compute bottleneck,
join counters to elapsed-host ROI rates, replace primary five-run means, or
attribute these differences to the abstraction independently of the actual
kernel mapping, compiler and profiling protocol.

## Paper-ready summary

The fixed-step Jacobi case illustrates why executed-work rooflines must be read
with the implementation mapping. Native CUDA's 858.85M counted FP32 FLOPs versus
502.33M for Julia reconcile exactly with its explicit shuffle/block reduction;
Julia instead uses point atomics, whose arithmetic is outside the selected
add/multiply/FMA counter classes. These are equal scientific update counts, not
complete arithmetic-work equivalence. Native Triad and STREAM dot also contract
multiply-adds while the Julia captures emit separate additions and multiplies,
leaving weighted FLOPs unchanged. JACC's higher dot/Jacobi DRAM traffic is mainly
write traffic; its arithmetic totals match the other Julia variants. Source
inspection shows that the template copy is not a declared global write and
that point-atomic granularity is shared with KA; neither directly accounts for
the JACC-specific write excess. Indirect compiler effects of these constructs
remain possible, and retained data do not establish the compiler/cache mechanism. These are source- and counter-supported observations,
not causal framework rankings or whole-ROI performance explanations.

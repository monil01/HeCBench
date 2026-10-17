# Miller–Rabin probable-prime testing

The native benchmark evaluates strong probable-prime predicates using base prefixes
of (2, 7, 61), with straightforward UInt64 modular multiplication/remainder and
an efficient UInt32 Montgomery-arithmetic implementation. This is computational
number theory and a potential cryptographic building block, not an implemented
cryptographic protocol, security claim, or floating-point scientific solver.

The historical Julia program is preserved in `reference_original/main.jl` and the
CUDA.jl child. It touched one GPU marker and printed canned output without doing
primality tests. Its PASS and zero times are not numerical or performance evidence.
The restored reference implements both source algorithms, including the 128-entry
inverse seed table, Hensel lifting, Montgomery overflow correction and actual
UInt32/UInt64 wraparound semantics. No floating-point surrogate is used.

## Input and accuracy contract

The fixture generator calls unchanged source `set_nprimes` and `set_nintegers`.
It preserves the fixed xorshift state, wheel increment policy, nominal 8/16/24/32
bit masks, and 100,000 values per width for each distribution. Prime incrementing
can exceed the nominal mask boundary, just as in the source; nominal width is not
a guaranteed strict upper bound. All values are odd and at least five. These are
synthetic benchmark distributions, not secure random generation or a uniform
sample of all primes. Prime generation and all CPU preparation are outside timing.

An independent Python arbitrary-integer `pow` oracle checks each base-prefix
predicate. Duplicate inputs are cached only in CPU oracle preparation; the GPU
still evaluates every original input. Each configuration retains four original
100,000-input datasets. Artifacts contain the exact three bases and 400,000 input
values, 1,200,000 untimed per-input/base-prefix decisions, and the four actual timed
aggregate counter results. The untimed evaluator calls the same source arithmetic
helpers; it supplements the source kernels' aggregate-only observable output and
is not claimed to be an array emitted by the timed native kernel. The full
per-input decisions, their sums, and all five direct dual-reference comparisons
must be exact. Integer classifications have no floating-point tolerance.

## Timed work

The timed native kernels remain unchanged: 391 blocks of 256 lanes, a guard for
100,000 active inputs, three base-prefix predicate calls and three global atomic
adds per input, including adds of zero. The aggregate is the sum across all three
prefixes, not merely the number passing the final three-base test. Each width's
counter reset and synchronization happen outside its timer. The kernel launch and
completion are timed. Ten repetitions per width are averaged, then the four width
means are summed, matching the source's reported sum of width-specific kernel
means. Five outer trials are retained and the final four averaged. Original
Makefile repetition 100 is reduced to ten for the named study; all input sizes,
widths, bases and algorithms remain present. Untimed full-output instrumentation,
CPU checking, transfers, serialization and fixture generation are excluded.

Input distribution changes early exits and modular-arithmetic work: prime inputs
usually execute all subtests, while many odd composites exit early. The source's
single contended counter also matters. Actual divergence, integer division cost,
register/local-memory behavior of the inverse table, cache traffic and atomic
bottlenecks require profiling. Floating-point FLOP roofline counts do not describe
these unsigned modular-arithmetic kernels.

## Completed results and effort

All four configurations pass across all four models. The separate saved-artifact audit checks all 80 artifacts byte-exactly against the actual bases/input prefix and independent arbitrary-integer predicate oracle. Prefix monotonicity, all-prime success and actual timed aggregate sums pass.

| Distribution / algorithm | CUDA (us) | CUDA.jl (us) | KA (us) | JACC (us) |
|---|---:|---:|---:|---:|
| prime_simple | 2367.034 | 2547.588 | 2561.353 | 2601.620 |
| prime_efficient | 713.407 | 333.727 | 346.855 | 361.503 |
| odd_simple | 1350.397 | 1470.705 | 1501.485 | 1508.382 |
| odd_efficient | 477.611 | 320.174 | 328.349 | 341.217 |

These are sums of four width-specific kernel means, not time per candidate or one width. All retained trials/variation are stored. Montgomery and straightforward algorithms have genuinely different integer operation sequences; input distributions alter early exits. No observed timing gap is assigned a hardware cause without profiling.

KA needed two program candidates: an explicit final return was rejected by the frontend kernel macro before device JIT; removing it allowed four actual device-JIT checks. Thus five frontend/process evaluations are not five device compilations. JACC passed one candidate and four device-JIT checks. Each has two kernel definitions, one timed and one validation-only. The shared arithmetic helper was authored/validated in the reference phase, not charged as novel target-only work. Original Julia restoration passed its first program candidate in four checks; native isolated fixture staging needed one configuration repair after exit3, with original native kernels unchanged. Internal compiler passes and specialization counts remain unknown. Usage windows include cached context and coordination; broad/child windows overlap and cannot be added as exclusive billed cost.

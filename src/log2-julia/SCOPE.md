# Requested-bit log2 approximation: corrected secondary configuration

The historical CUDA.jl file only launched a marker and printed placeholder
RMSE/timing text. Its source and Makefile are preserved in reference_original/.
New Julia implementations execute the genuine iterative-squaring algorithm.

The original native source stops before adding the requested last fractional
bit. Actual original GPU captures fail the predeclared pointwise analytic
budget at66574 points each, across19 requested precisions and8269 inputs.
The maximum absolute log2 error is approximately1.0. The minimal native
comparison change >= to > restores that bit; originals, exact patch, all
five rejected captures and their source hashes remain preserved. All accepted
results are corrected_reference_secondary, excluded from untouched-native
aggregates. No original failed configuration is certified or overwritten.

Every accepted capture contains8269 actual Float32 device inputs,19 host
precision-argument metadata values,157111 output values and two exact guards.
All20 accepted artifacts are bit-identical across CUDA/CUDA.jl/KA/JACC.
The independent CPU Float64 math.log2 oracle checks each point against the
predeclared budget 2^-p +16u +2ulp32(log2(x)), u=2^-24. This separates requested
fractional-bit truncation, recurrence-rounding allowance and output storage
resolution. A reported max_abs_error is analytic error over all precisions,
not the cross-GPU artifact difference (which is zero). The independent oracle
is Float64 libm, not a formal interval or arbitrary-precision certification.

Inputs remain in the original finite range[1,1e7], covering integers1..8193,
fractional witnesses and both Float32 neighbors around powers of two.
Subunit/negative/zero/huge-exponent inputs are excluded: the original exponent
and signed-shift normalization cannot be generalized silently. The study
covers all19 source precision values, not the original10-million-input CLI.

The timed region is the19-precision kernel-launch suite with17 internal
repetitions and device-wide completion. Setup/reset/readback/CPU checks and
artifact writes are excluded. Five trials are all checked; mean uses2..5.
Each model uses256-thread independent maps. Work varies with input and
requested precision; no static FLOP count, measured roofline position or
universal abstraction ranking is asserted.

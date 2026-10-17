# Mixture-of-experts routing

The previous Julia program executed a one-element marker and printed canned PASS and zero time. These ports execute row softmax and repeated top-k selection, preserving the native two-stage algorithm and lower-expert-ID tie rule. The source normalizes over all 384 experts and does not renormalize selected weights. For a flat row with k=8, the retained mass is 8/384, not one. A hypothetical constant-one expert mixture would therefore change by a factor of48 if a translator added subset renormalization. This is a mathematical contract witness, not an executed complete model.

## Scope and numerical checks

Eight named cases:257 tokens,384 experts, k=1,2,4,8, with bounded dyadic logits or large/tied logits. The latter include flat rows, multiple equal maxima and a single dominant expert. The original32768-token CLI, finished flags and expert-parallel subranges are not certified. Devices use Float32 probabilities/arithmetic and Int32 indices; Float64 is only the independent oracle and lossless mixed artifact encoding.

Every one of812,634 output values and797,246 exact input-prefix values is checked on all20 saved12,879,040-byte artifacts. These include complete softmax arrays, weights, expert indices and source-row mappings. Python natural-exp/math.fsum plus full stable sorting supplies an independent oracle; row sums, probability bounds, unique experts and weights equal selected probabilities are also checked. Maximum absolute error6.85e-9 is below the fixed atol1e-6/rtol1e-5. Integer outputs are exact; maximum ID2055 implies a unit mismatch cannot pass the binary floating budget. Reversing the tie rule changes3515 selected integers. Canonical input bytes are tied to actual GPU allocations by source/copy audit, not inferred from a checksum.

## Mapping and timing

Native kernels and source remain unchanged: CUB reductions,256 threads and257 token blocks. Julia implementations use explicit shared halving trees. Native compiled static shared bytes are52/80 and registers31/31; CUDA.jl/KA source shared policy is1024/2048 static, JACC1024/2048 dynamic. JACC copies256/512 Float32 template values plus a barrier per block; shared expert IDs use exactly representable Float32, while global IDs remain Int32. All device kernels use framework APIs. No measured Julia occupancy/traffic/registers or isolated abstraction effect is claimed.

Each of five trials records eight component means from20 synchronized kernel pairs. Only time1 is discarded; means use2–5, all correctness checks required. Allocation, initial-index reset, validation and full capture are outside timing. All timers finish with blocking device-wide completion, including all kernel streams.

## Effort

The current reference and both portable targets pass their first candidate/device evaluation, with two kernel definitions each. Eight cases and five trials are separate units. Historical translation consumption is unknown. A postverification Python array-type error in the artifact audit was repaired without a GPU rerun. Prospective processed-token windows include repeated/cached context and team/source/paper coordination; the broad workflow overlaps children and is not additive or billed cost. Retrospective raw-library stream review interleaves the annotation phase, while this benchmark itself already uses correct device-wide boundaries.

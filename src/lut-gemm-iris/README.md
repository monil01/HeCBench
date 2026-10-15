# LUT-GEMM through IRIS DMem

The original CLI cases (K=1024/4096/12288, N=4K), 8/4/3-bit suite, group
size 128, seed 123 and RNG reset/generation ordering are retained. Packed
weights and FP16 input/alpha/bias use DMem. The original 32-wide lookup
kernel body, 256-thread blocks, 2048-output tiles, half arithmetic, and
half2 atomic accumulation are preserved; only its exported name changes.
An IRIS clear task precedes each lookup launch. Host-side packing avoids
the original redundant K*N*bits float qW allocation; it encodes the identical
sign bits and dequantizes the same dense reference weights from those bits.

The mandatory reference is still the original real hipBLAS FP16 GemmEx,
with original descriptors/layout, dispatched using the shared IRIS vendor
bridge, IRIS-resolved GPU pointers and IRIS-owned stream. No benchmark GPU
memory is allocated/copied through native HIP. Errors in the vendor callback
fail the process through the bridge instead of silently succeeding.

The original reference layout and mean-error print are preserved, including
the fact that its mean error is not near zero. It does not print a PASS/FAIL
criterion. An independent host simulation reproduces packed-byte LUT
construction, bias/scale arithmetic and ascending tile accumulation. Every
native AND IRIS LUT value must match that SAME reference within the standard
addition-order forward-error bound 2*gamma_t*sum(abs(tile)), with half unit
roundoff u=2^-11, gamma_t=t*u/(1-t*u), t=K/32 (minimum floor 1e-3).
Half2 atomic tile accumulation order is nondeterministic. A separate, tighter
direct native/IRIS comparison requires atol=0.125/rtol=0.02, so the conservative
forward-error bound is not the only numerical check. Every hipBLAS output
must match exactly. The source has no original PASS tolerance to change.
IRIS_LUT_DUMP=1 emits both complete arrays. Timings include DMem/task/flush
overhead and are not vendor-kernel performance equivalents (P1).

Tested K=1024 and the original default K=4096 with repeat=2 rather than the
default repeat=100. K=12288 is not claimed as validated.

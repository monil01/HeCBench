# MF-SGD IRIS DMem port

The full original CPU reader/shuffling/grid partitioning/scheduling and binary
model writer are retained, along with LCG feature initialization, persistent
rank128 FP16 Hogwild SGD updates, learning-rate schedule, GPU half-to-float
conversion, model scaling and inverse permutation. All three partition paths
(single grid, first-level partition, double-buffered second-level partition)
remain present. CLI/default fields are preserved. The source documents only
rank128 support; unsupported ranks (including the unsafe legacy default80)
now fail before GPU access, matching the repaired CUDA/HIP baselines.

The explicit adapter owns every device allocation as `iris::DMem`; its pointer
keys are CPU mirror addresses, never native device allocations. Tasks flush
all written buffers. Host byte copies update/fetch DMem mirrors. No native GPU
allocation/copy/launch or CPU fallback for initialization, SGD, or conversion.
CPU data preprocessing and final model scaling are original source behavior.
P1: asynchronous stream/prefetch operations are represented by synchronous
DMem updates/tasks; all partition dataflow is retained but physical overlap
and performance are not equivalent to native streaming. No stream-overlap
claim is made.

Necessary native/source repairs, also applied here: protect RNG state writes
before the bounds check, allocate one initialization RNG state per launched
thread (old modulo-sharing raced), and specify width32 for HIP shuffles used
by logical32-lane workers. Original random-device seeding remains the default;
`HECBENCH_MF_SEED` is opt-in deterministic validation only. Setting
`HECBENCH_MF_VERIFY_INIT=1` independently checks all initialized FP16 values
against a CPU LCG/rounding oracle, not a benchmark computation fallback.

`make run` retains the recovered original Netflix run-script knobs and input
path via configurable TRAIN/MODEL; those Netflix assets remain unavailable.
Validation instead uses genuine GroupLens MovieLens100K ratings through the
existing general binary-file CLI (int32 user/item, float32 rating). Ratings
are not invented and data is not redistributed/committed. Archive/checksum,
source split membership, zero-based ID conversion and license are recorded in
local evidence. This is NOT validation of the Netflix/default-large workload.
Native HIP comparisons are against the explicitly repaired baseline, with
whole binary model and independently evaluated held-out predictions checked.
Parallel Hogwild execution is intentionally scheduling-nondeterministic, even
within one128-thread block. Strict whole-model validation uses the explicit
`HECBENCH_MF_VERIFY_SERIAL_WARPS` compile flag in BOTH native and IRIS kernels,
with4 workers/oneblock. It orders all four logical warps without dropping any
worker, iteration, or update. This diagnostic schedule is NOT the production
parallel schedule; original parallel comparisons and native self-variation
are retained without a false strict-parity claim. Use `make -B
HIPFLAGS='-O3 -DHECBENCH_MF_VERIFY_SERIAL_WARPS'` to build that diagnostic,
and `make -B` to restore production behavior. The independent HIP initialization
oracle models the observed gfx908 fused multiplication-to-FP16 single rounding,
avoiding a mistaken extra FP32 rounding at half-way cases.

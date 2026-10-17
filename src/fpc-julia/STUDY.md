# FPC study

## Scientific operation and scope

FPC estimates the number of payload bytes for common 32-bit word patterns. It produces one scalar byte total, not a serialized compressed stream. Neither compression metadata nor decompression is implemented by the original kernels. No floating-point codec or scientific dataset is identified in the source.

Two source kernels are measured separately: ordered short-circuit branch classification (`fpc`) and separately evaluated predicate-mask classification (`fpc2`). Both assign 1, 2 or 4 bytes per word, using zero, signed-byte/halfword, low-half-zero, two small halves, repeated byte and default cases. Every256-thread group uses a4-byte UInt32 shared atomic counter; its last lane merges the block count globally. CUDA.jl and KA retain this geometry; JACC adds one4-byte shared-template preload and barrier and requests exactly4 dynamic shared bytes.

## Inputs and numerical evidence

The named fixture contains262144 UInt64 containers whose high32bits are zero. Sixteen balanced literal patterns include signed boundaries and INT_MIN; an independently hand-enumerated byte-cost table predicts524288 bytes. Both nativeCPU FPCCompress and Julia helpers must match the table. Every trial compares the entire observable output: **one scalar**, exact against CPU and both native/CUDA.jl references. The input word count is not the number of checked outputs. The estimated payload excludes format metadata and is not evidence of an encoded lossless round trip.

The actual device input is2MiB; semantic uncompressed32-bit words total1MiB. This cache-resident bounded fixture does not establish DRAM performance or behavior on scientific distributions. The original random generator includes shifts beyond a32-bit operand width; that source-level nonportability is avoided, not silently assigned a portable RNG meaning. Original larger wordcounts, end-to-end offload and generator distributions remain unmeasured.

## Timing and workflow evidence

All4 models time GPU counter reset plus the selected classifier/atomic kernel,10 repetitions per trial, explicit JIT warmup then5 in-process trials. The reported time is mean runs2–5, retaining all samples. Allocation, H2D, scalar D2H and artifact/CPU checks are outside timing. This is an explicitly named GPU region rather than original native allocation/copy/free timing.

The historical CUDA.jl reference called its branch kernel for both labels and used a shared tree. Its original source is archived; the current reference restores the distinct mask kernel and native atomic mapping. Reference preparation required2 program candidates and4 checks across2 configurations because a shared fixture-generator missing space caused a Julia parse error before device JIT. One semantic reference mapping candidate and one shared fixture syntax repair are separately recorded. Each KA/JACC target passed its first program candidate in2 actual checks (one per source configuration), with no target repairs. JACC intentionally reuses the reviewed KA mapping and shared classifier math.

## Usage accounting

Reference audit, KA translation, JACC translation and annotation have prospective consumed-usage windows. These include cached/repeated context, reasoning, coordination and shared work, not exclusive source-generation cost or monetary billing. The broad workflow window overlaps phases and must not be added to them. Partial saved-source proxies remain separate; historical CUDA.jl generation cost is unknown. End markers are captured in a separate round after phase results and metadata.

## Timing dispersion

Both JACC configurations contain a retained millisecond-scale sample amid tens-of-microseconds trials. Their large means are therefore unstable, not evidence of a stable abstraction slowdown. All4 retained values remain in the reported means; no preferred rerun or outlier deletion. KA branch CV is0.516. Cause is unprofiled.

{
  "fpc": {
    "cuda": {
      "retained_trial_seconds": [
        9.3441e-06,
        7.72e-06,
        7.7601e-06,
        7.5541e-06
      ],
      "coefficient_of_variation": 0.10349845492365838,
      "outlier_removed": false
    },
    "cuda_jl": {
      "retained_trial_seconds": [
        1.05971e-05,
        8.882e-06,
        8.4211e-06,
        8.3461e-06
      ],
      "coefficient_of_variation": 0.11595614188253843,
      "outlier_removed": false
    },
    "ka": {
      "retained_trial_seconds": [
        3.21483e-05,
        1.46531e-05,
        1.29342e-05,
        1.28741e-05
      ],
      "coefficient_of_variation": 0.5160175553976958,
      "outlier_removed": false
    },
    "jacc": {
      "retained_trial_seconds": [
        8.71068e-05,
        0.0062297766,
        7.11647e-05,
        6.35336e-05
      ],
      "coefficient_of_variation": 1.9083298298024929,
      "outlier_removed": false
    }
  },
  "fpc2": {
    "cuda": {
      "retained_trial_seconds": [
        7.5861e-06,
        7.502e-06,
        7.4041e-06,
        8.1741e-06
      ],
      "coefficient_of_variation": 0.045186727722222646,
      "outlier_removed": false
    },
    "cuda_jl": {
      "retained_trial_seconds": [
        1.16332e-05,
        1.00551e-05,
        9.5491e-06,
        9.0511e-06
      ],
      "coefficient_of_variation": 0.11105150644970296,
      "outlier_removed": false
    },
    "ka": {
      "retained_trial_seconds": [
        2.18592e-05,
        1.06881e-05,
        1.04531e-05,
        9.8751e-06
      ],
      "coefficient_of_variation": 0.43652257780628767,
      "outlier_removed": false
    },
    "jacc": {
      "retained_trial_seconds": [
        0.0052291103,
        6.97396e-05,
        6.14785e-05,
        6.08526e-05
      ],
      "coefficient_of_variation": 1.9055232003356484,
      "outlier_removed": false
    }
  }
}

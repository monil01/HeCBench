# GEAM transpose study

## Operation and source scope

Despite the GEAM name, the native benchmark exercises only α=1,β=0,opA=transpose,opB=identity: a pure matrix transpose. Both original FP32 and FP64 modes are measured separately. No general scaled matrix addition is claimed. No physical/scientific data application is identified; this is matrix layout movement useful to downstream linear algebra.

The shared fixture is a nonconstant bounded integer pattern in a1025x1537 matrix. All1575425 entries, including partial32x32 edge tiles, are checked exactly against an independent CPU row/column permutation and both references, all5trials. FP32 and FP64 need no arithmetic rounding in this case.

## Independent implementation mappings

Native CUDA and genuine existing CUDA.jl delegate the whole transpose to cuBLAS S/Dgeam. Current studyβ0B aliases input in both references; original Julia used a separate zeroB, mathematically irrelevant atβ0 and preserved in original source. Only entrypoint guard and matched fixture adapter are added; no semantic reference repair is needed. Vendor internal grid, tile, scratch and register policy are unknown. Earlier pinned-environment cuBLAS resolution/version/hash evidence is linked, not misrepresented as contemporaneous GEAM process maps.

KA and JACC implement a custom32x32 tile with33-element padded leading dimension,32x8 lanes,49x33groups,4values perlane. Both load/store guards cover partialedge tiles, while all lanes execute barriers. KA static local memory is4224B FP32 or8448B FP64. JACC explicitly requests these same dynamic sizes and additionally stages a33x32 zero-template matrix with an initial barrier. That extra logical load traffic is not measured DRAM traffic.

## Performance protocol and interpretation

All4 models time10 repeated transposes pertrial with one endpoint synchronization, explicit JIT/librarywarmup then5in-process trials, mean runs2–5. Resets/allocations/copies/checks are outside timing. Native original synchronizes everycall, so its exact fence frequency is not claimed reproduced. Native/CUDA.jl vendor algorithms versus custom portable tiles are an explicit performance confound; same mathematical operation does not imply same internal kernel.

Input+output footprints are12.60MB FP32 and25.21MB FP64, exceeding V1006MiB L2, but a DRAM bottleneck is not established without counters. Useful floatingpoint work is zero: report logical read+write bytes/time rather than invented GFLOP rates or FP64 compute-roofline positioning. This numerator excludes template/internal traffic and is an effective application bandwidth, not measured memory transactions. Source16384x16384default and other scale regimes remain unmeasured.

## Effort and usage

Each KA/JACC program passed its first generic source candidate in2actual compiler/fullnumerical checks (FP32/64 specializations),0repairs. JACC starts with successful KA mapping available. Reference audit is separately1candidate/2checks/2nativebuilds, guard/adapteronly; historical Julia generation costunknown. Prospective consumed-usage windows include context/cache and coordination; partial source proxies remain separate. Broadworkflow overlaps phases and includes a short Gaussian fixture-generator packaging interruption after KAEND/beforeJACCBEGIN; that work is excluded from both target phases and not retroactively reassigned.

## All retained timing samples

{
  "f32": {
    "cuda": {
      "retained_trial_seconds": [
        2.41972e-05,
        2.40243e-05,
        2.37943e-05,
        2.39062e-05
      ],
      "coefficient_of_variation": 0.007185262232590898,
      "max_to_min_ratio": 1.0169326267215257,
      "outlier_removed": false
    },
    "cuda_jl": {
      "retained_trial_seconds": [
        3.30443e-05,
        3.32984e-05,
        3.44683e-05,
        3.29753e-05
      ],
      "coefficient_of_variation": 0.02078462695170767,
      "max_to_min_ratio": 1.0452763128766076,
      "outlier_removed": false
    },
    "ka": {
      "retained_trial_seconds": [
        4.13374e-05,
        3.05313e-05,
        2.89123e-05,
        3.04323e-05
      ],
      "coefficient_of_variation": 0.17490396893977783,
      "max_to_min_ratio": 1.4297513515009184,
      "outlier_removed": false
    },
    "jacc": {
      "retained_trial_seconds": [
        0.0001874158,
        0.0001863687,
        0.0001865777,
        0.0001862017
      ],
      "coefficient_of_variation": 0.002887716398311557,
      "max_to_min_ratio": 1.0065203486326924,
      "outlier_removed": false
    }
  },
  "f64": {
    "cuda": {
      "retained_trial_seconds": [
        4.25594e-05,
        4.22684e-05,
        4.22684e-05,
        4.27804e-05
      ],
      "coefficient_of_variation": 0.005857083358517083,
      "max_to_min_ratio": 1.0121130679183505,
      "outlier_removed": false
    },
    "cuda_jl": {
      "retained_trial_seconds": [
        5.91325e-05,
        5.08755e-05,
        5.04424e-05,
        5.02195e-05
      ],
      "coefficient_of_variation": 0.08199772856777555,
      "max_to_min_ratio": 1.1774808590288632,
      "outlier_removed": false
    },
    "ka": {
      "retained_trial_seconds": [
        4.63244e-05,
        4.33484e-05,
        4.25144e-05,
        4.25524e-05
      ],
      "coefficient_of_variation": 0.041231330582389925,
      "max_to_min_ratio": 1.089616694578778,
      "outlier_removed": false
    },
    "jacc": {
      "retained_trial_seconds": [
        0.0002050309,
        0.0002033758,
        0.0002033839,
        0.0002043219
      ],
      "coefficient_of_variation": 0.003933684798518061,
      "max_to_min_ratio": 1.0081381363957758,
      "outlier_removed": false
    }
  }
}

All four retained samples stay in each mean; variation does not establish a mechanism for any framework gap.

# MF-SGD: qualified source one-partition training study

Original cuMF_SGD source and notices are preserved under `original_native/`. Current code does not modify `src/mf-sgd-cuda`. The README documents recommender-system matrix factorization, rank 128, FP16 factors and FP32 warp-reduced SGD. Native source supports asynchronous Hogwild updates, so a unique parallel model array is not assumed.

## Native readiness and source qualification

The exact original sized worker-state initializer wrote 244 logical UInt32 suffix words when 12 workers were launched as 256 threads. A physically oversized guard allocation safely captured every actual write. `corrected_native/` moves the assignment inside the existing size check; every valid state is unchanged and all guards pass. See `original_rng_guard_witness_review.json`, `corrected_rng_guard_witness_review.json`, and the canonical `reference_correction_review.json`. This is an explicitly corrected-native stratum, excluded from untouched-native aggregates, not an original whole-CLI failure claim.

The original feature initializer also shares mutable RNG states across threads. Current same-seed/same-initial-state repeats produced up to 19,231 differing half words at 943 feature rows; actual RNG/feature bytes and guards are retained. This is a source policy/nondeterminism observation, not proof of a wrong random distribution. General parallel Hogwild outputs vary even with fixed actual initial features and selector streams: the first native series differed in 329,777 of 336,000 factor words, maximum difference 0.1866455. These observations are not framework failures.

## Inputs and measured scope

The exact cached public MovieLens 100k ZIP is 4,924,029 bytes, SHA256 50d2a982c66986937beb9ffb3aa76efe955bf3d5c6b761f4e3a7cd717c6a3229. Current work verified the cache without an HTTP request. `asset_provenance.json` records the actual 80k training / 20k held-out member and converted binary hashes. IDs are converted from 1-based to 0-based; rating values are exact Float32; timestamps are unused by the source binary format. This is an additional named real input, not the source README's Netflix/default-large dataset.

`study_prepare_inputs.cu` uses the source model preparation, actual random row maps and actual GPU-initialized FP16 readbacks to create the frozen shared input fixture. They are not reconstructed from a deterministic-seed claim. `prepared_inputs_audit.json` independently checks every rating permutation/scaling, all maps/inverses/rates and feature values before training. Raw ratings and the captured 1,893,212-byte input fixture are required small assets.

The complete default **one-partition SGD-update function** runs 30 epochs, rank 128, original update-vector size 128, with the source default of 12 workers and a separate 128-worker case. All allocations, worker RNG, dynamic rates, input copies, persistent parallel updates, readbacks, GPU Half-to-Float32 conversion and destruction are included. Original model postscale/unshuffle and all train/test predictions execute outside that source update-function ROI, with complete values retained. Multipartition/async-prefetch paths, the original Netflix CLI and model-building time are explicitly unmeasured. This is a source-path qualification, not an unsupported-framework claim.

## Evidence strength

This is **property equivalence, not raw model numerical identity**. Complete actual raw models, predictions, physical GPU/CPU guards and input readbacks are retained. Primary comparison values include exact actual input bits, every source selector, selection counts and predeclared property flags; they do not replace raw models with an equality claim.

The predeclared properties require finite/changed factors, exact source selectors/final states, unchanged guards, strictly decreased training squared error and held-out dot-only RMSE at most 3.0. The header b is retained but not added to the diagnostic dot prediction, following the source SGD residual r-dot. These gates are not a convergence, unique-trajectory, serialization or recommendation-accuracy certificate.

All 5 source GPU definitions are exercised: 3 timed training-pipeline definitions plus the original unbounded RNG/feature kernels on deterministic nonaliasing capability controls. A 12-warp control with independent rows rank 128 update checks 3,072 exact-rational gradient components against a derived FP32/FP16-rounding bound; 62,720 unselected components must stay bit-exact. The v2 control includes wide Half exponents/subnormal values. The valid earlier fixed-point v1 series is preserved separately; the coverage upgrade was chosen before any target translation/timing and changed no tolerance/property criterion or training algorithm.

A separate C++ checker, without CUDA/original solver code, recomputes all Half conversions and Float32 scale/unshuffle values bit-exactly, and every train/test prediction and objective inlong double with a 64-bit mantissa. `audit_complete_evidence.py` joins every physical GPU/CPU guard and payload, every used input and full selector/count stream to the actual primary/scientific artifacts. The unused uninitialized tail of the source 1024-rate buffer remains fully captured as bytes; only its defined consumed/initialized prefix is claimed identical.

## Current verified native reference

`study_config_native_v2.json`; `native_readiness_receipt.json`; `native_v2_complete_evidence_audit.json`. Five checked trials follow full warmup; timings use only trials 2–5. The mean 0.132354542 s is the **sum of two complete SGD-update function regions**, with observer allocation/context/trace covariates. Original printed self-timings include raw capture and are not substituted for this measured value. No per-worker isolated timing or native whole-CLI claim is made.

Compile the independent CPU checker with an existing compiler (no installation):

```sh
g++ -O3 -std=c++17 -fno-fast-math src/mf-sgd-julia/study_model_oracle.cpp -o /tmp/mf-sgd-model-oracle
```

Run GPU configurations only through `tools/verify_portable_julia.py` and the existing global study lock on GPU 6. Raw data/executables remain local under STATE or /tmp. Exact sources/runtime assets are bound in each configuration. All three Julia versions are complete: `all4_final_receipt.json`, `final_source_binding_audit.json`, each child `verification.json`/`translation_accounting.json`, and `paper_observations.md`/`.json` record the final scope. Canonical all-four configuration is `study_config_jacc.json`; all three target complete-evidence audits pass. Native preparation is separate from target translation.

| Native CUDA | CUDA.jl | KA | JACC |
|---:|---:|---:|---:|
| 0.132355 s | 0.138979 s | 0.143079 s | 0.142172 s |

These are the composite means of trials 2–5 for the two configured worker settings, with property equivalence and source-repair qualifications. Complete learned models and predictions remain available, with no raw-model identity or universal performance claim.

CUDA.jl required three program evaluations; KA and JACC each required one, inheriting the validated operator and host harness. The generic 64-bit Julia shuffle works at its correct one-based lane 1: the earlier source-lane-0 mapping caused observed compile/runtime errors. The provisional framework-gap diagnosis is withdrawn in `failure_diagnosis_adjudications.json`, with all raw failures and exact event-line bindings preserved.

Observed context-inclusive translation windows are CUDA.jl 9,121,990, KA 2,688,960 and JACC 2,801,849 tokens. The native-readiness window is 11,177,360 separately; annotations are separate, and nested verification windows are not additional tokens. Shared-harness and inherited-context strata are explicit. No system/package upgrade or privileged command was performed.

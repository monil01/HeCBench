# SVD3x3: qualified fixed-accuracy factorization ports

CUDA.jl, KA and JACC each execute their own authored point kernel, using the mechanically transcribed Float32/UInt32 scalar algorithm. No GPU SVD library or native GPU function is delegated to by a Julia target. The shared C++ checker runs on the CPU outside the timed region.

## Evidence and accuracy scope

The original four-sweep approximation passes its shared CPU checker but fails the independently predeclared normalized reconstruction budget of 0.002 on 1,300 of 1,048,576 actual SoA matrices. This describes accuracy relative to the selected budget; it does not establish that the intentionally fixed-sweep approximation is a source coding bug. The one-line eight-sweep intervention passes every unchanged property budget on that full dataset. Preserve both diagnostic series. This qualified reference is excluded from untouched-native performance comparisons.

The primary four-version contract covers batches of 1, 31, 256, 257, 4,097 and 65,537 matrices, including analytic rank/degeneracy controls and declared original-data selections. It retains full actual inputs, initial outputs, readonly readbacks, U/S/V outputs and allocation guards. Singular-vector sign and subspace freedom require property equivalence, with normalized singular values compared across models. No raw-vector identity or full-million-matrix target certificate is claimed.

`raw_state_audit_20trials.json` independently rebuilds all tensor checks from 480 saved raw files: 1,403,580 complete factorizations. `study_plan.json`, `predeclared_contract.json`, `completion_scope_review.json` and `paper_observations.json` define the scope and RQ links. Actual arrays remain in ignored local evidence storage.

## Measurement and effort

Each of five checked trials sums six seven-launch region means with whole-context completion. The source 256-thread groups and `n/256+1` group count are preserved. Discard only the first trial's time and average the last four. Allocation, transfers, independent checks and raw I/O are outside these regions. Native fast-math and ordinary typed Julia arithmetic are explicit rounding-policy covariates.

The CUDA.jl path includes a wrong environment, a host syntax rejection, an unmapped C helper and a compiled XOR-as-exponentiation numerical failure. These produce five program checks, three device-JIT processes, three submitted device bundles and two numerical bodies. KA and JACC pass their first candidates after ordered helper reuse. Per-version observed counters are in each child `translation_accounting.json`; shared native/paper windows overlap and must not be added as exclusive cost categories.

## Regeneration

1. Run `fetch_original_assets.py` for the exact size/MD5/SHA-bound DVC archive.
2. Run `generate_assets.py` with NumPy/LAPACK to regenerate twelve measured input and singular-value files. A fresh-directory CPU regeneration reproduced all twelve hashes.
3. Preserve the measured configurations; construct a fresh configuration for a new environment and run all four versions through `tools/verify_portable_julia.py`.
4. Run `audit_saved_evidence.py` against the new result and its explicitly identified raw-state directory. The current auditor selects the one completed five-trial series per model in this study's local raw directory; isolated reproduction directories require an explicit path adaptation and a new audit receipt.

The independent checker and regeneration scripts supplement actual GPU runs. They do not substitute for numerical evidence or populate unknown historical token fields.

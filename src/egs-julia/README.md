# CUDA EGS photon transport study

This study uses the complete source photon transport and detector accumulation pipeline, with the original 32³ voxel phantom, 512² detector, 80 kVp spectrum and cross-section tables. It models photons only. It does not certify electron transport, absorbed dose, clinical accuracy, or the original 100-million-history CLI.

## Preserved source and corrected reference

Original CUDA files remain unchanged in `../egs-cuda/` and are copied byte for byte under `reference_history/`. `configured_native/EGS.h` selects one block and one 32-thread warp; the production horizon remains 32,768 iterations. The source activity termination then drains particles and runs the final empty batch. The native reference uses 32,768 / 64 / 1 iterations and starts 43,728 photons: the requested history threshold of one is not the actual number of histories.

An original CPU histogram initializer reads beyond its allocation. `spectrum_reference_provenance.json` records the original ASan witness and the minimal correction: 145 histogram intervals have 146 edges, so alias construction and sampling use 145 columns. Unused alias-array tails are initialized explicitly. All original GPU function bodies are unchanged. This is a conditional corrected-native study, excluded from untouched-native aggregates.

## Regenerate original scientific assets

From the repository root:

```sh
python3 src/egs-julia/regenerate.py --data-root .porting-state/portable_julia/external_assets/egs/data
```

The script downloads only the original declared DVC data object, checks size 466,485 bytes, MD5 `c8b24576e18d0a0108c9ee4d9debea8f` and SHA256 `b069c8078089ba6d1d4a5b68522ad9caae8828dae903a50312b90374872dc2b8`, then extracts only the nine hash-checked scientific files. No downloaded code executes. Their hashes are in `original_dvc_asset_provenance.json`. The separately recovered author archive supplies identical scientific file bytes.

## Evidence and limits

`host_preparation.cu` runs original CPU table/phantom preparation and exposes guarded allocations. Julia implementations borrow those buffers, author all GPU computations in Julia, and do not call the native simulation or detector kernels. CUDA collectives and device-specific math are explicit backend extensions.

Each trial captures every actual prepared input allocation, all mutable phase outputs, and the full normalized category/total images. `artifact_layout.json` describes the mixed encoding. The initial prefix preserves every payload bit as exactly representable UInt32 words; integer output fields use exact byte values. Floating output fields are compared numerically under a declared budget.

Independent checks cover all detector-pixel accumulations, particle population conservation, finite/energy/direction invariants, canonical scalar RNG state membership, source emission geometry/energy, and Compton/Rayleigh identities. They are not an independent complete EGSnrc transport solver or a statistical cross-section validation.

All five trials are checked. Only trial one's time is discarded; trials two through five are averaged. The measured region is the sum of separately context-completed simulation and accumulation segments. Captures, CPU checks, reset, allocation and host normalization are outside the region. The observation pauses and reduced launch geometry make this a named study workload, not original continuous CLI performance.

The first native five-trial receipt is `20261019T203737202499Z_egs`: all checks pass, mean 0.2390462512 seconds. All four models are verified in final receipt `20261019T205107425923Z_egs`; source/dependency and all 20 artifact hashes were independently audited. CUDA.jl / KA / JACC means are 0.3249475800 / 0.3273687928 / 0.3294649613 seconds.

## Fresh full study on the recorded environment

After regenerating the data above, generate a relocated configuration and run the central verifier:

```sh
python3 src/egs-julia/make_config.py --data-root .porting-state/portable_julia/external_assets/egs/data --output /tmp/egs_fourway.json
python3 tools/verify_portable_julia.py --config /tmp/egs_fourway.json
```

The central verifier owns the GPU lock; do not nest another lock around it. These commands create a new series and do not replace the archived accepted series. The pinned Julia environments and CUDA 12.0 sm70 compiler are required. The original faulty initializer can be reproduced with `probe_source_bounds.cu` under host ASan; that entrypoint now explicitly includes the archived original initializer. Historical error logs retain their original source paths. `configuration_history_review.json` distinguishes the pre-repair launch plan from the current corrected-reference contract.

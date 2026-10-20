# t-SNE: bounded full-state four-way study

## Selected scope

Original60000-point MNIST precomputed KNN32 inputs,130×130 interpolation boxes,3 nodes/axis, **one complete update**. All24 active original CUDA entry definitions are represented by25 mathematical cases in one generic Julia index body. The unused BroadcastRowVector definition is not certified. Four Fourier passes are source custom direct DFT kernels, not cuFFT. Complete programs are **hybrid host control**, with own-framework device kernels; there is no native GPU kernel delegation.

Final result: `.porting-state/portable_julia/runs/20261020T070225578007Z_tsne/result.json`. Each model has an untimed warmup and five checked in-process executions; mean trials2–5 is0.08478019025 /0.92132640325 /0.910263911 /0.9957545565 seconds (CUDA/CUDA.jl/KA/JACC). Original source input IO and observation are excluded consistently. Framework host allocation, retention, reduction and GC behavior remain covariates; this is not a pure frontend comparison.

## Numerical evidence and exclusions

All20 complete captures contain32,563,847Float32 words each (130,255,388B), including3,960,000 actual device input words checked bit-exact. All35 fields have full independent saved-array checks;28 were original mandatory gates, seven are explicitly post-execution supplemental checks. See `final_saved_full_audit.json`. Full CPU oracle uses Float64 sparse/entropy/interpolation formulas and Complex128 FFT rather than replaying direct DFT source.

The **revised contract** scales only four Fourier fields by fixed CPU-derived powers of two: kernel by2^20 and three other workspaces by2^36. Codec proof confirms exact raw GPU bitstream round trips for every trial. All other31 fields are raw unchanged. Every value is compared, not a checksum substitute. This normwise coordinate choice follows an observed strict raw pointwise rejection; original failures and series remain preserved. `power2_codec.py --decode` recovers the raw binary output. Integer grid categories and all input bits are exact.

The strict three-update contract failed integer gridding/interpolation comparisons and remains failed. Original1000-update CLI actually exited0 with finite final text but has **no full trajectory certificate**. Target perplexity50 is infeasible for31 nonself neighbors: source capped200-step search yields31. Source normalization-axis simplification and high-dimensional KNN generation are not validated application quality. No claim of full scientific t-SNE quality or convergence.

**Safety coverage:** selected captures have no explicit output/allocation redzones. Original unchanged I1 native memcheck12.9 reported0 errors; this is separate diagnostic evidence, not all-model bounds/guard coverage. Report this remaining requirement explicitly. Native GPU arithmetic is unchanged; the named observer/integration adapter and revised metric exclude the result from untouched-native aggregates.

## Cold CPU regeneration and fresh configuration

No package or OS mutation is needed with existing pinned dependencies. Python requires NumPy2.5.4 (`cpu_requirements.txt`). From the repository root, use whitespace-free output paths:

```sh
PYTHONPATH=/tmp/pasc-julia-plot-tools python3 src/tsne-julia/regenerate_assets.py --output /tmp/tsne-cold/assets --proof /tmp/tsne-cold/assets_proof.json
PYTHONPATH=/tmp/pasc-julia-plot-tools python3 src/tsne-julia/build_native.py --output /tmp/tsne-cold/native --pythonpath /tmp/pasc-julia-plot-tools
python3 src/tsne-julia/prepare_cold_configs.py --assets /tmp/tsne-cold/assets --native-build /tmp/tsne-cold/native --output /tmp/tsne-cold/configs --pythonpath /tmp/pasc-julia-plot-tools
```

Actual cold CPU execution reproduced all62 binary assets/432,355,216B exactly and rebuilt the owned native observer successfully. Fresh all-four configuration preparation was verified; **no additional cold GPU run is claimed**. Original measured observer contains an absolute Python audit path; the cold builder's narrow process-local Python launcher relocates that CPU-only call without editing GPU code or installing system files. Generated CPU binaries and multi-gigabyte raw GPU captures stay local; deterministic recipes/hash inventories are essential artifacts. Source project clones pin CUDA6.2.1/KA0.9.45/JACC1.4.0 with only `CUDACore.nonblocking_synchronization=false`; no shared environment mutation.

## Effort accounting

CUDA.jl6 program evaluations include1 frontend syntax rejection,1 actual device compiler identifier repair,1 environment worker failure and2 numerical/contract rejections;5 reached device JIT. KA/JACC each one candidate/check after shared controller/contract inheritance, with no arithmetic repair. Target counter windows total10,949,555 /2,123,794 /2,469,809 tokens; input/cached/output/reasoning breakdowns remain in immutable boundary events and manifests. Shared native preparation, verification and annotation are separate. These context-inclusive counters are not exclusive billed costs or recovered historical cost; heterogeneous revised scopes are calibration-ineligible.

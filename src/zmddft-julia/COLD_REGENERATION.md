# Cold regeneration (CPU assets, then an explicitly fresh verification series)

No large binary input/oracle or actual GPU capture belongs in source Git. The committed essential assets are the deterministic generators, source coefficient table in ../zmddft-cuda/main.cu, study_plan.json byte counts/SHA256, pinned CPU dependency and generation_reproduction.json. Current input/oracles total 1,879,048,192 bytes plus the 4096-byte coefficient table. Generated primary artifacts are 1,073,746,496 bytes per model/trial; all three actual stage outputs are retained, not checksums or sampled elements.

In the executed environment, from repository root:

```sh
PYTHONPATH=/tmp/pasc-julia-plot-tools OPENBLAS_NUM_THREADS=1 python3 src/zmddft-julia/generate_fixtures.py
PYTHONPATH=/tmp/pasc-julia-plot-tools OPENBLAS_NUM_THREADS=1 python3 src/zmddft-julia/reproduce_assets.py
```

On a fresh machine, install cpu_requirements.txt in a CPU Python environment and use that interpreter instead of the task-local PYTHONPATH. Exact Python/NumPy/build/platform provenance is recorded separately. The coefficient and dyadic-input bytes are deterministic; FFT oracle hashes are the observed pinned CPU-build result. A different CPU/library build may change the lowest oracle bits; preserve this archived study_plan.json and declare a new fixture/configuration rather than overwriting old evidence or claiming identical hashes.

The committed three Julia entrypoints are already generated. To reproduce them independently, run generate_kernels.py --model cuda_jl, then --model ka, then --model jacc in a copied source tree; reproduce_assets.py does this in a temporary tree and compares every output byte without changing measured sources.

The primary config references immutable local captured reference bundles. A fresh checkout should not pretend to have those captures. After regenerating CPU assets and preparing the pinned Julia environments/toolkit, create a fresh four-model configuration:

```sh
python3 src/zmddft-julia/prepare_cold_config.py
PYTHONPATH=/tmp/pasc-julia-plot-tools OPENBLAS_NUM_THREADS=1 python3 tools/verify_portable_julia.py --config src/zmddft-julia/study_config_cold.json
```

This last command intentionally launches a **new**, named five-trial series on all four models. It was not run during the CPU-only annotation/reproduction audit. The verifier locks GPU6 internally; do not wrap it in an outer flock. Original dataset shape 256³ complex Float64, source64threads/16384blocks, all three stages and R100 remain unchanged. Large comparison scans are bounded and retain all hashes/prefix checks; optional NumPy acceleration does not sample values. Available GPU memory should cover four 256MiB data buffers plus normal compiler/runtime storage; local saved primary artifacts alone occupy about 21.5GB logical bytes, plus four untimed full-stage impulse captures (about3.2GB).

# Pinned Julia math environment

This separate project extends the existing portable study environment with SpecialFunctions2.9.0 for ordinary erf and other supported mathematical functions. CUDA6.2.1, KernelAbstractions0.9.45 and (for JACC) JACC1.4.0 are preserved by Pkg.PRESERVE_ALL; Manifest.toml records the complete exact dependency graph. LocalPreferences.toml selects the existing local CUDA12 toolkit. Julia version:1.12.6.

Resolve or instantiate this project without updating the original _julia_ka_env/_julia_jacc_env. No GPU feature support is inferred merely from successful package resolution; each benchmark must compile and verify its actual numerical contract under the shared GPU6 flock protocol. Record this project and manifest hashes in each adopting study configuration. Setup logs and original/new SHA256s are local STATE/portable_julia/math_environments.json.

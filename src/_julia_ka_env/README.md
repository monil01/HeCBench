# KA study environment

Pinned Julia 1.12.6 environment: CUDA 6.2.1, KernelAbstractions 0.9.45,
StaticArrays and Atomix. Manifest.toml records exact transitive versions.
LocalPreferences.toml selects installed CUDA toolkit (12.0 on study host),
required for Tesla V100 sm70; default CUDA 13 artifacts do not support V100.

Use `julia --project=src/_julia_ka_env SCRIPT.jl ...` from repository root.
GPU commands must acquire `/tmp/hecbench-julia-study-gpu.lock` and set
`CUDA_VISIBLE_DEVICES=6`; runner handles both itself.

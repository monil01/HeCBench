# JACC study environment

Pinned Julia 1.12.6 environment: CUDA 6.2.1, JACC 1.4.0, StaticArrays.
Manifest.toml records exact transitive versions. LocalPreferences.toml selects
local CUDA toolkit and the JACC CUDA backend. Default CUDA 13 artifacts cannot
execute on this host's Tesla V100 sm70 GPUs.

Use `julia --project=src/_julia_jacc_env SCRIPT.jl ...` from repository root.
GPU commands acquire `/tmp/hecbench-julia-study-gpu.lock` and set
`CUDA_VISIBLE_DEVICES=6`; runner handles both itself.

APIs checked in installed JACC 1.4.0: `JACC.parallel_for(N,f,args...)`,
`JACC.LaunchSpec(;sync=false,threads=...)`, `JACC.sync_workgroup()`.
`JACC.shared(vector)` copies the whole vector to dynamic shared storage;
it is not implicitly a per-block tile. Inspect the CUDA extension before
porting shared-memory kernels and record any workaround or semantic change.

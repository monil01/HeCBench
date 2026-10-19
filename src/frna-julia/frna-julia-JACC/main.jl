using JACC
include(joinpath(@__DIR__,"..","thermodynamic_device.jl"))
include(joinpath(@__DIR__,"..","stage_device.jl"))
# All scientific recurrences are Julia. CUDA static shared memory and block
# barriers explicitly preserve the source reduction via JACC's CUDA backend.
@inline function init_kernel(idx,c,n)
    i=Int32(idx)-Int32(1)
    @inbounds c.w5[i+Int32(1)]=0
    @inbounds c.w3[i+Int32(1)]=0
    nothing
end
@inline function hairpin_kernel(idx,c,n,d)
    @inbounds rna_scalar_stage(Val(1),c,n,d,Int32(idx)-Int32(1))
    nothing
end
@inline function exterior_kernel(idx,c,n,d)
    @inbounds rna_scalar_stage(Val(3),c,n,d,Int32(idx)-Int32(1))
    nothing
end
@inline function multibranch_kernel(idx,c,n,d)
    @inbounds rna_scalar_stage(Val(4),c,n,d,Int32(idx)-Int32(1))
    nothing
end
@inline function W_kernel(idx,c,n,d)
    @inbounds rna_scalar_stage(Val(6),c,n,d,Int32(idx)-Int32(1))
    nothing
end
@inline function bulge_kernel(idx,c,n,d)
    lane=Int32(CUDA.threadIdx().x)-Int32(1)
    cell=Int32(CUDA.blockIdx().x)-Int32(1)
    tx=rem(lane,Int32(16))
    ty=div(lane,Int32(16))
    tid=tx*Int32(16)+ty
    @inbounds rna_reduce_cuda(Val(2),c,n,d,cell,tx,ty,lane,tid)
    nothing
end
@inline function coaxial_kernel(idx,c,n,d)
    lane=Int32(CUDA.threadIdx().x)-Int32(1)
    cell=Int32(CUDA.blockIdx().x)-Int32(1)
    tx=lane
    ty=Int32(0)
    tid=lane
    @inbounds rna_reduce_cuda(Val(5),c,n,d,cell,tx,ty,lane,tid)
    nothing
end
@inline function WM_kernel(idx,c,n,d)
    lane=Int32(CUDA.threadIdx().x)-Int32(1)
    cell=Int32(CUDA.blockIdx().x)-Int32(1)
    tx=lane
    ty=Int32(0)
    tid=lane
    @inbounds rna_reduce_cuda(Val(7),c,n,d,cell,tx,ty,lane,tid)
    nothing
end
@inline function wl_coax_kernel(idx,c,n,d)
    lane=Int32(CUDA.threadIdx().x)-Int32(1)
    cell=Int32(CUDA.blockIdx().x)-Int32(1)
    tx=lane
    ty=Int32(0)
    tid=lane
    @inbounds rna_reduce_cuda(Val(8),c,n,d,cell,tx,ty,lane,tid)
    nothing
end
@inline function w5w3_kernel(idx,c,n,d)
    lane=Int32(CUDA.threadIdx().x)-Int32(1)
    cell=Int32(0)
    tx=lane
    ty=Int32(0)
    tid=lane
    @inbounds rna_reduce_cuda(Val(9),c,n,d,cell,tx,ty,lane,tid)
    nothing
end
# Complete launch dimensions per case prevent mutable inferred-grid reuse.
const RNA_NS=(77,12,9,44,13)
const INIT_SPECS=Dict(n=>JACC.launch_spec(;threads=1,blocks=n+1,shmem_size=0,sync=false) for n in RNA_NS)
const SCALAR_SPECS=Dict(n=>JACC.launch_spec(;threads=1,blocks=n,shmem_size=0,sync=false) for n in RNA_NS)
const BLOCK_SPECS=Dict(n=>JACC.launch_spec(;threads=256,blocks=n,shmem_size=0,sync=false) for n in RNA_NS)
const FINAL_SPEC=JACC.launch_spec(;threads=256,blocks=1,shmem_size=0,sync=false)
launch_init(c,n)=JACC.parallel_for(INIT_SPECS[Int(n)],Int(n)+1,init_kernel,c,n)
function launch_step(c,n,d)
    JACC.parallel_for(SCALAR_SPECS[Int(n)],Int(n),hairpin_kernel,c,n,d)
    JACC.parallel_for(BLOCK_SPECS[Int(n)],256*Int(n),bulge_kernel,c,n,d)
    JACC.parallel_for(SCALAR_SPECS[Int(n)],Int(n),exterior_kernel,c,n,d)
    JACC.parallel_for(SCALAR_SPECS[Int(n)],Int(n),multibranch_kernel,c,n,d)
    JACC.parallel_for(BLOCK_SPECS[Int(n)],256*Int(n),coaxial_kernel,c,n,d)
    JACC.parallel_for(SCALAR_SPECS[Int(n)],Int(n),W_kernel,c,n,d)
    JACC.parallel_for(BLOCK_SPECS[Int(n)],256*Int(n),WM_kernel,c,n,d)
    JACC.parallel_for(BLOCK_SPECS[Int(n)],256*Int(n),wl_coax_kernel,c,n,d)
    JACC.parallel_for(FINAL_SPEC,256,w5w3_kernel,c,n,d)
    nothing
end
include(joinpath(@__DIR__,"..","study_common.jl"))
rna_study()

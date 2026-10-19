include(joinpath(@__DIR__,"..","thermodynamic_device.jl"))
include(joinpath(@__DIR__,"..","stage_device.jl"))
function init_kernel(c,n)
    i=Int32(blockIdx().x)-Int32(1)
    @inbounds c.w5[i+Int32(1)]=0
    @inbounds c.w3[i+Int32(1)]=0
    return
end
function hairpin_kernel(c,n,d)
    @inbounds rna_scalar_stage(Val(1),c,n,d,Int32(blockIdx().x)-Int32(1));return
end
function bulge_kernel(c,n,d)
    tx=Int32(threadIdx().x)-Int32(1);ty=Int32(threadIdx().y)-Int32(1)
    @inbounds rna_reduce_cuda(Val(2),c,n,d,Int32(blockIdx().x)-Int32(1),tx,ty,tx+Int32(16)*ty,tx*Int32(16)+ty);return
end
function exterior_kernel(c,n,d)
    @inbounds rna_scalar_stage(Val(3),c,n,d,Int32(blockIdx().x)-Int32(1));return
end
function multibranch_kernel(c,n,d)
    @inbounds rna_scalar_stage(Val(4),c,n,d,Int32(blockIdx().x)-Int32(1));return
end
function coaxial_kernel(c,n,d)
    lane=Int32(threadIdx().x)-Int32(1)
    @inbounds rna_reduce_cuda(Val(5),c,n,d,Int32(blockIdx().x)-Int32(1),lane,Int32(0),lane,lane);return
end
function W_kernel(c,n,d)
    @inbounds rna_scalar_stage(Val(6),c,n,d,Int32(blockIdx().x)-Int32(1));return
end
function WM_kernel(c,n,d)
    lane=Int32(threadIdx().x)-Int32(1)
    @inbounds rna_reduce_cuda(Val(7),c,n,d,Int32(blockIdx().x)-Int32(1),lane,Int32(0),lane,lane);return
end
function wl_coax_kernel(c,n,d)
    lane=Int32(threadIdx().x)-Int32(1)
    @inbounds rna_reduce_cuda(Val(8),c,n,d,Int32(blockIdx().x)-Int32(1),lane,Int32(0),lane,lane);return
end
function w5w3_kernel(c,n,d)
    lane=Int32(threadIdx().x)-Int32(1)
    @inbounds rna_reduce_cuda(Val(9),c,n,d,Int32(0),lane,Int32(0),lane,lane);return
end
launch_init(c,n)=(@cuda threads=1 blocks=n+1 init_kernel(c,n))
function launch_step(c,n,d)
    @cuda threads=1 blocks=n hairpin_kernel(c,n,d)
    @cuda threads=(16,16) blocks=n bulge_kernel(c,n,d)
    @cuda threads=1 blocks=n exterior_kernel(c,n,d)
    @cuda threads=1 blocks=n multibranch_kernel(c,n,d)
    @cuda threads=256 blocks=n coaxial_kernel(c,n,d)
    @cuda threads=1 blocks=n W_kernel(c,n,d)
    @cuda threads=256 blocks=n WM_kernel(c,n,d)
    @cuda threads=256 blocks=n wl_coax_kernel(c,n,d)
    @cuda threads=256 blocks=1 w5w3_kernel(c,n,d)
    nothing
end
include(joinpath(@__DIR__,"..","study_common.jl"))
rna_study()

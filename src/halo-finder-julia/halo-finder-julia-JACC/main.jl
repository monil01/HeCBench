using JACC
include(joinpath(@__DIR__,"..","force_device.jl"))
# Framework dispatch with explicit CUDA cached-load/rn-rsqrt/warp/atomic extensions.
@inline function force_kernel(i,j,c,n,m,cut2,soft2,fcoeff,xstride,ystride)
    @inbounds halo_step(c,n,m,cut2,soft2,fcoeff,Int32(i)-Int32(1),Int32(j)-Int32(1),xstride,ystride)
    nothing
end
# Complete grids are keyed independently: no mutable inferred-grid reuse.
const HALO_JACC_SPECS=Dict((gx,gy)=>JACC.launch_spec(;threads=(32,4),blocks=(gx,gy),shmem_size=0,sync=false) for gx in (1,3,32) for gy in 1:3)
function launch_halo(a,c)
    spec=HALO_JACC_SPECS[(Int(a.gx),Int(a.gy))]
    JACC.parallel_for(spec,(Int(a.gx)*32,Int(a.gy)*4),force_kernel,c,a.count,a.count1,a.cut2,a.soft2,a.fcoeff,Int32(a.gx*32),Int32(a.gy*4))
    nothing
end
include(joinpath(@__DIR__,"..","study_common.jl"))
halo_study()

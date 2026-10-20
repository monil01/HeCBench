using KernelAbstractions
include(joinpath(@__DIR__,"..","force_device.jl"))
# Explicit CUDA extensions retain source cached vector loads, correct-rsqrt,
# warp shuffles and Float32 atomics; the scientific body is authored Julia.
const HALO_KA_BACKEND=CUDA.CUDABackend(always_inline=true)
@kernel function force_kernel(c,n,m,cut2,soft2,fcoeff,xstride,ystride)
    global_index=@index(Global,Cartesian)
    xidx=Int32(global_index[1])-Int32(1);yidx=Int32(global_index[2])-Int32(1)
    @inbounds halo_step(c,n,m,cut2,soft2,fcoeff,xidx,yidx,xstride,ystride)
end
function launch_halo(a,c)
    # Selected KA CUDA backend exposes maxthreads/always_inline, not the native
    # minimum-7-blocks-per-SM compilation hint; this difference is recorded.
    force_kernel(HALO_KA_BACKEND,(32,4))(c,a.count,a.count1,a.cut2,a.soft2,a.fcoeff,Int32(a.gx*32),Int32(a.gy*4);ndrange=(Int(a.gx)*32,Int(a.gy)*4))
    nothing
end
include(joinpath(@__DIR__,"..","study_common.jl"))
halo_study()

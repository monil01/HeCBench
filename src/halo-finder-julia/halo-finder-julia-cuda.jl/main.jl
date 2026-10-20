include(joinpath(@__DIR__,"..","force_device.jl"))
function force_kernel(c,n,m,cut2,soft2,fcoeff)
    xidx=(Int32(blockIdx().x)-Int32(1))*Int32(blockDim().x)+Int32(threadIdx().x)-Int32(1)
    yidx=(Int32(blockIdx().y)-Int32(1))*Int32(blockDim().y)+Int32(threadIdx().y)-Int32(1)
    xstride=Int32(blockDim().x)*Int32(gridDim().x);ystride=Int32(blockDim().y)*Int32(gridDim().y)
    @inbounds halo_step(c,n,m,cut2,soft2,fcoeff,xidx,yidx,xstride,ystride)
    return
end
function launch_halo(a,c)
    # Source __launch_bounds__(128,7) is mirrored explicitly for CUDA.jl.
    @cuda always_inline=true maxthreads=128 blocks_per_sm=7 threads=(a.bx,a.by) blocks=(a.gx,a.gy) force_kernel(c,a.count,a.count1,a.cut2,a.soft2,a.fcoeff)
    nothing
end
include(joinpath(@__DIR__,"..","study_common.jl"))
halo_study()

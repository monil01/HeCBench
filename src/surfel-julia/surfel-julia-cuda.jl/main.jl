using CUDA
@inline block_barrier()=CUDA.sync_threads()
include(joinpath(@__DIR__,"..","study_kernel_bodies.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
function base!(s,N,f,w,h,d)
 x=(Int32(blockIdx().x)-Int32(1))*Int32(16)+Int32(threadIdx().x)-Int32(1);y=(Int32(blockIdx().y)-Int32(1))*Int32(16)+Int32(threadIdx().y)-Int32(1);base_body!(x,y,s,N,f,w,h,d);return
end
function tile!(s,N,f,w,h,d)
 ix=Int32(threadIdx().x)-Int32(1);iy=Int32(threadIdx().y)-Int32(1);x=(Int32(blockIdx().x)-Int32(1))*Int32(16)+ix;y=(Int32(blockIdx().y)-Int32(1))*Int32(16)+iy;shared=CuStaticSharedArray(Float32,1792);tile_body!(x,y,iy*Int32(16)+ix,s,N,f,w,h,d,shared);return
end
function run!(tiled,s,N,f,w,h,d,tmp)
 if tiled;@cuda threads=(16,16) blocks=(2,2) tile!(s,N,f,w,h,d)
 else;@cuda threads=(16,16) blocks=(2,2) base!(s,N,f,w,h,d)
 end
 return
end
study(run!)

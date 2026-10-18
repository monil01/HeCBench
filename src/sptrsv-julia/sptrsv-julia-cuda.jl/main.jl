using CUDA
include(joinpath(@__DIR__,"..","study_device.jl"))
function solve!(rp,ci,v,flags,m,b,x,warpmap,len)
 sums=CUDA.CuStaticSharedArray(Float64,256)
 localid=Int32(CUDA.threadIdx().x)-Int32(1);globalid=(Int32(CUDA.blockIdx().x)-Int32(1))*Int32(256)+localid
 solve_body!(rp,ci,v,flags,m,b,x,warpmap,len,localid,globalid,sums)
 return nothing
end
function launch(rp,ci,v,flags,m,b,x,warpmap,len)
 CUDA.@cuda threads=256 blocks=cld(Int(len)-1,8) solve!(rp,ci,v,flags,m,b,x,warpmap,len)
end
include(joinpath(@__DIR__,"..","study_common.jl"))
study(CuArray,launch)

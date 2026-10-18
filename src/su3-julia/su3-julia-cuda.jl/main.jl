using CUDA
include(joinpath(@__DIR__,"..","study_device.jl"))
function su3!(a,b,c,n)
 id=(Int32(CUDA.blockIdx().x)-Int32(1))*Int32(CUDA.blockDim().x)+Int32(CUDA.threadIdx().x)-Int32(1)
 su3_body!(a,b,c,n,id)
 return nothing
end
function launch(a,b,c,n,threads)
 CUDA.@cuda threads=threads blocks=Int(n) su3!(a,b,c,n)
 nothing
end
include(joinpath(@__DIR__,"..","study_common.jl"))
study(CuArray,launch)

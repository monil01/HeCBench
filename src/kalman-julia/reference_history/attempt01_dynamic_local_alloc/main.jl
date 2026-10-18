using CUDA
include(joinpath(@__DIR__,"..","kalman_math.jl"))
function kalman_full!(args)
 bid=Int32((blockIdx().x-1)*blockDim().x+threadIdx().x-1)
 kalman_site!(bid,args...)
 return
end
launch(c,args)=CUDA.@cuda threads=256 blocks=cld(Int(c.ns),256) kalman_full!(args)
include(joinpath(@__DIR__,"..","study_common.jl"))
kalman_study(launch)

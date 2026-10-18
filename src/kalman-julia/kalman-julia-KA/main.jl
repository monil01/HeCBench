using CUDA,KernelAbstractions
include(joinpath(@__DIR__,"..","kalman_math.jl"))
@kernel unsafe_indices=true function kalman_full!(args)
 bid=Int32(@index(Global,Linear)-1)
 kalman_site!(bid,args...)
end
function launch(c,args)
 kalman_full!(CUDABackend(),256)(args;ndrange=cld(Int(c.ns),256)*256)
 nothing
end
include(joinpath(@__DIR__,"..","study_common.jl"))
kalman_study(launch)

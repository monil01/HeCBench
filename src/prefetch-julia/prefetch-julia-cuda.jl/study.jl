include("main.jl")
include(joinpath(@__DIR__,"..","study_common.jl"))
function launch!(n,a,b)
 @cuda threads=256 blocks=cld(Int(n),256) stream=CUDA.default_stream() add_cuda!(n,a,b)
 nothing
end
prefetch_study(launch!)

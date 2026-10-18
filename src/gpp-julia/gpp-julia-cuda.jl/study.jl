include("main.jl")
include(joinpath(@__DIR__,"..","study_common.jl"))
function launch(c)
 CUDA.@cuda threads=32 blocks=(c.gx,c.gy) solver_kernel!(c.args...)
end
gpp_study(launch)

using CUDA
include(joinpath(@__DIR__,"..","study_common.jl"))
include(joinpath(@__DIR__,"..","study_reference_kernels.jl"))
function run!(which,args...)
 if which==:init;@cuda threads=64 blocks=65536 kernel5_init!(args...)
 elseif which==:move;@cuda threads=64 blocks=65536 kernel5_move!(args...)
 else;@cuda threads=64 blocks=65536 kernel5_check!(args...)
 end
 return
end
study(run!)

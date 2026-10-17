include_string(@__MODULE__,first(split(read(joinpath(@__DIR__,"main.jl"),String),"function cal_num_less_cpu")),joinpath(@__DIR__,"main.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
function factory(diagonal,off,counts,buffers)
return idx->begin
@cuda threads=256 blocks=1 cal_num_eigenvalue_interval_kernel!(counts,buffers[idx],diagonal,off,UInt32(256))
@cuda threads=256 blocks=1 recalculate_eigen_intervals_kernel!(buffers[3-idx],buffers[idx],counts,diagonal,off,UInt32(256),0.001f0)
end
end
eigenvalue_study(factory)

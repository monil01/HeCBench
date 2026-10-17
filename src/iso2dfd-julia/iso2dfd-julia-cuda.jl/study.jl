using CUDA
source=read(joinpath(@__DIR__,"main.jl"),String)
include_string(@__MODULE__,first(split(source,"exit(main(ARGS))")),joinpath(@__DIR__,"main.jl"))
function launch_ref!(next,prev,vel,coef,nr,nc)
    @cuda threads=(16,16) blocks=(cld(nc,16),cld(nr,16)) iso_2dfd_kernel!(next,prev,vel,coef,nr,nc)
end
include(joinpath(@__DIR__,"..","study_common.jl"))
iso_study(launch_ref!)

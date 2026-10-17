using CUDA,KernelAbstractions
include(joinpath(@__DIR__,"..","study_math.jl"))
@kernel function multinomial_ka!(dest,nd,nc,samples,values,sd,sc)
    lane=@index(Local,Linear);block=@index(Group,Linear)-1
    if lane==1;multinomial_row!(Int32(block),dest,nd,nc,samples,values,sd,sc);end
end
function launch_ka!(args...)
    multinomial_ka!(get_backend(args[1]),256)(args...;ndrange=512*256)
end
include(joinpath(@__DIR__,"..","study_common.jl"));multinomial_study(launch_ka!)

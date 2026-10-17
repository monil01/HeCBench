using CUDA,KernelAbstractions
include(joinpath(@__DIR__,"..","study_math.jl"))
@kernel function daphne_ka!(cp,x,y,z,v,n)
    i=@index(Global,Linear)
    project_item!(i,cp,x,y,z,v,n)
end
function launch_ka!(cp,x,y,z,v,n)
    daphne_ka!(get_backend(x),256)(cp,x,y,z,v,n;ndrange=Int(n))
end
include(joinpath(@__DIR__,"..","study_common.jl"));daphne_study(launch_ka!)

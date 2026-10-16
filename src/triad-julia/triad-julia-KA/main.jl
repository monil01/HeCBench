using CUDA,KernelAbstractions
@kernel function triad_ka!(a,b,c,s,n)
    i=@index(Global,Linear)
    if i<=n
        @inbounds c[i]=a[i]+s*b[i]
    end
end
function launch_ka!(a,b,c,s,n)
    triad_ka!(get_backend(c),128)(a,b,c,s,n;ndrange=n)
end
include(joinpath(@__DIR__,"..","study_common.jl"))
triad_study(launch_ka!)

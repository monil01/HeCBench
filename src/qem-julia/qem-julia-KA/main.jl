using KernelAbstractions
include(joinpath(@__DIR__,"..","quartic_behavior_math.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
@kernel function qr_ka(n,a)
 i=@index(Global,Linear)
 if i<=n;qem_qr!(i,a);end
end
@kernel function solve_ka(n,a)
 i=@index(Global,Linear)
 if i<=n;qem_solve!(i,a);end
end
function launch!(a,n)
 backend=KernelAbstractions.get_backend(a[1])
 qr_ka(backend,64)(n,a;ndrange=Int(n))
 solve_ka(backend,64)(n,a;ndrange=Int(n))
end
study_main(launch!)

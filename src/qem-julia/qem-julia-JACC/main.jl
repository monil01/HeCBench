using JACC
include(joinpath(@__DIR__,"..","quartic_behavior_math.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
@inline function qr_jacc(i,n,a)
 if i<=n;qem_qr!(i,a);end
 return
end
@inline function solve_jacc(i,n,a)
 if i<=n;qem_solve!(i,a);end
 return
end
function launch!(a,n)
 backend=JACC.get_backend(:cuda)
 spec=JACC.LaunchSpec{typeof(backend)}(;threads=64,blocks=cld(n,64),shmem_size=0,sync=false)
 JACC.parallel_for(qr_jacc,spec,Int(n),n,a)
 JACC.parallel_for(solve_jacc,spec,Int(n),n,a)
end
study_main(launch!)

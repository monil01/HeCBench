include(joinpath(@__DIR__,"..","quartic_behavior_math.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
function study_qr(n,a)
 i=(blockIdx().x-Int32(1))*blockDim().x+threadIdx().x
 if i<=n;qem_qr!(i,a);end
 return
end
function study_solve(n,a)
 i=(blockIdx().x-Int32(1))*blockDim().x+threadIdx().x
 if i<=n;qem_solve!(i,a);end
 return
end
function launch!(a,n)
 @cuda threads=64 blocks=cld(n,64) study_qr(n,a)
 @cuda threads=64 blocks=cld(n,64) study_solve(n,a)
end
study_main(launch!)

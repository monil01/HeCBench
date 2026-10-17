include(joinpath(@__DIR__,"..","study_reference_kernel.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
function run!(a,oa,b,ob,c,oc,types,w,omega)
 @cuda threads=(1,HEIGHT) blocks=(256,1) lbm_kernel!(a,oa,b,ob,c,oc,types,w,omega)
 nothing
end
study(run!)

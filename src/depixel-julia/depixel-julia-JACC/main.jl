using CUDA,JACC
include(joinpath(@__DIR__,"..","depixel_math.jl"))
function dep_check_kernel!(i,a,t,w,h)
 dep_check!(a,t,Int32(i)-Int32(1),w,h)
 return
end
function dep_eliminate_kernel!(i,t,o,w,h)
 dep_eliminate!(t,o,Int32(i)-Int32(1),w,h)
 return
end
include(joinpath(@__DIR__,"..","study_common.jl"))
const DEP_SPECS=Dict(n=>JACC.LaunchSpec{typeof(JACC.get_backend(:cuda))}(;threads=256,blocks=n÷256,shmem_size=0,sync=false) for n in (256,512,1024))
function dep_launch!(a,t,o,w,h,n)
 spec=DEP_SPECS[n]
 JACC.parallel_for(dep_check_kernel!,spec,n,a,t,w,h)
 JACC.parallel_for(dep_eliminate_kernel!,spec,n,t,o,w,h)
end
depixel_study(dep_launch!)

using CUDA,KernelAbstractions
include(joinpath(@__DIR__,"..","depixel_math.jl"))
@kernel unsafe_indices=true function dep_check_kernel!(a,t,w,h)
 i=Int32(@index(Global,Linear))-Int32(1)
 dep_check!(a,t,i,w,h)
end
@kernel unsafe_indices=true function dep_eliminate_kernel!(t,o,w,h)
 i=Int32(@index(Global,Linear))-Int32(1)
 dep_eliminate!(t,o,i,w,h)
end
include(joinpath(@__DIR__,"..","study_common.jl"))
const DEP_KCHECK=dep_check_kernel!(CUDABackend(),256)
const DEP_KELIM=dep_eliminate_kernel!(CUDABackend(),256)
function dep_launch!(a,t,o,w,h,n)
 DEP_KCHECK(a,t,w,h;ndrange=n)
 DEP_KELIM(t,o,w,h;ndrange=n)
end
depixel_study(dep_launch!)

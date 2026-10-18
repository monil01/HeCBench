using CUDA,KernelAbstractions
@kernel function vectoradd!(a,b,c,n)
 i=@index(Global,Linear)
 if i<=n;@inbounds c[i]=a[i]+b[i];end
end
const add_kernel=vectoradd!(CUDABackend(),256)
function launch(a,b,c,n,s)
 add_kernel(a,b,c,n;ndrange=cld(Int(n),256)*256)
 nothing
end
include("../study_common.jl")
allocation_study(launch)

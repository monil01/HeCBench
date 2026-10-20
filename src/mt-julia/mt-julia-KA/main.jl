using CUDA,KernelAbstractions
const MODEL="ka"
include("../study_device.jl")
@kernel function kernel_mt!(parameters,output,n)
 gid=Int32(@index(Global,Linear))-Int32(1)
 body_mt!(gid,parameters,output,n)
end
@kernel function kernel_boxmuller!(output,n)
 gid=Int32(@index(Global,Linear))-Int32(1)
 body_boxmuller!(gid,output,n)
end
function dispatch_mt!(p,r,n)
 kernel_mt!(get_backend(r),128)(p,r,n;ndrange=4096)
 nothing
end
function dispatch_boxmuller!(r,n)
 kernel_boxmuller!(get_backend(r),128)(r,n;ndrange=4096)
 nothing
end
include("../study_common.jl")
main()

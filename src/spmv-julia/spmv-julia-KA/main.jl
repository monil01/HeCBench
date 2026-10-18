using CUDA,KernelAbstractions
include("../study_device.jl")
@kernel function dense_ka!(a,x,y,n)
 i=@index(Global,Linear)
 dense_body!(a,x,y,n,i)
end
launch(a,x,y,n)=dense_ka!(CUDA.CUDABackend(),256)(a,x,y,n;ndrange=cld(n,256)*256)
include("../study_common.jl")
run_study(launch)

using CUDA,KernelAbstractions
include("../portable_body.jl")
@kernel function pointwise_kernel(h,b,th,ti,bias,lg,ho,io,ci,co)
 idx=@index(Global,Linear)
 pointwise_body!(Int32(idx-1),h,b,th,ti,bias,lg,ho,io,ci,co)
end
const object=pointwise_kernel(CUDABackend(),256)
launch(h,b,th,ti,bias,lg,ho,io,ci,co)=object(Int32(h),Int32(b),th,ti,bias,lg,ho,io,ci,co;ndrange=h*b)
include("../study_common.jl")
run_study(launch)

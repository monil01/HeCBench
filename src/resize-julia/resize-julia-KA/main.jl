using CUDA,KernelAbstractions
include("../resize_device.jl")
@kernel unsafe_indices=true function resize_kernel!(mode,out,inp,outsize,oh,ow,ih,iw,fy,fx)
 resize_site!(mode,out,inp,Int32(@index(Global,Linear))-Int32(1),Int32(29184*256),outsize,oh,ow,ih,iw,fy,fx)
end
const RK=resize_kernel!(CUDABackend(),256)
launch(c)=(RK(c.mode,c.output,c.input,UInt64(c.no),c.oh,c.ow,c.ih,c.iw,c.fy,c.fx;ndrange=29184*256);nothing)
include("../study_common.jl")
resize_study(launch)

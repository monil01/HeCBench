using CUDA,KernelAbstractions
include("../study_portable_common.jl")
@kernel function urng_kernel(input,output,n)
 i=Int32(@index(Global,Linear));lane=Int32(@index(Local,Linear));shuffle=@localmem Int32 (4096,)
 if i<=n;@inbounds output[i]=urng_pixel(input[i],lane,shuffle);end
end
function factory(input,output,n)
 k=urng_kernel(CUDABackend(),256)
 ()->k(input,output,n;ndrange=Int(cld(n,Int32(256)))*256)
end
urng_study(factory)

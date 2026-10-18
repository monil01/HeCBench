using CUDA,KernelAbstractions,Printf
include("../study_device.jl")
@kernel function snake_kernel(A,B,O,N,T)
 i=@index(Global,Linear)
 if i<=N;@inbounds O[i]=snake_filter(A,B,(Int32(i)-Int32(1))*Int32(8),T);end
end
function factory(A,B,O,N)
 k=snake_kernel(CUDABackend(),256)
 T->k(A,B,O,Int32(N),Int32(T);ndrange=cld(N,256)*256)
end
include("../study_common.jl")
snake_study(factory)

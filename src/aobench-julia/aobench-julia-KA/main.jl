using CUDA,KernelAbstractions
include("../study_device.jl")
include("../study_common.jl")
@kernel function ao_kernel!(img)
 x,y=@index(Global,NTuple)
 render_lane!(Int32(x-1),Int32(y-1),img)
end
function factory(img)
 k=ao_kernel!(CUDABackend(),(16,16));()->k(img;ndrange=(256,256))
end
ao_study(factory)

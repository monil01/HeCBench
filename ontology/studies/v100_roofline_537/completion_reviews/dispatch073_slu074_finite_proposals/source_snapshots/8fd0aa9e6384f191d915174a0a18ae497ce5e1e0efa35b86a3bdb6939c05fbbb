using CUDA,KernelAbstractions
include("../study_common.jl")
@kernel function empty_kernel()
end
const kernel_object=empty_kernel(CUDABackend(),1)
launch()=kernel_object(;ndrange=1)
dispatch_study(launch)

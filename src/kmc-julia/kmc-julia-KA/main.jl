using CUDA,KernelAbstractions
# The native benchmark is cuBLAS-based: own KA allocation, explicit CUDA vendor interop.
# Zero authored device kernels; no fictitious @kernel translation or portability claim.
allocate(x)=begin
    a=KernelAbstractions.allocate(CUDABackend(),eltype(x),length(x))
    copyto!(a,x)
    a
end
include("../study_common.jl")
runstudy(allocate)

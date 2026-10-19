using CUDA,KernelAbstractions
include(joinpath(@__DIR__,"..","study_common.jl"))
function ccl_ka_allocate(x)
    a=KernelAbstractions.allocate(CUDABackend(),eltype(x),(length(x),))
    copyto!(a,x)
    a
end
ccl_execute(ccl_ka_allocate)

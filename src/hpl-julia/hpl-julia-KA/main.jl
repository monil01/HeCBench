using CUDA,KernelAbstractions
const HPL_BACKEND=CUDA.CUDABackend()
function framework_array(host)
    dev=KernelAbstractions.allocate(HPL_BACKEND,eltype(host),size(host))
    copyto!(dev,host)
    dev
end
include(joinpath(@__DIR__,"..","study_common.jl"))
hpl_study()

using KernelAbstractions,CUDA
include(joinpath(@__DIR__,"..","study_common.jl"))
function study_allocate(A)
 d=KernelAbstractions.allocate(CUDABackend(),ComplexF64,size(A));copyto!(d,A);d
end
# Explicit CUDA cuFFT interoperability: KA allocates backend storage; no KA device FFT is claimed.
study_main(study_allocate)

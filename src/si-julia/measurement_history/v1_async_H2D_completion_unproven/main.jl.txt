using CUDA,Libdl,SHA
# The source has no authored device kernels. Genuine CUDA.jl allocation/copies
# integrate with the exact native cuBLAS handle/API rather than inventing @cuda.
si_allocate(n)=CUDA.CuArray{Float32}(undef,n)
si_release(a)=CUDA.unsafe_free!(a)
include(joinpath(@__DIR__,"..","study_common.jl"))
si_study()

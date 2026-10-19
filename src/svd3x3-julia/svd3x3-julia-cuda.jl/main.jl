using CUDA
include("../device_common.jl")
allocate_model(a)=CuArray(a)
function factor_kernel!(input,output,n)
    gid=(blockIdx().x-Int32(1))*blockDim().x+threadIdx().x-Int32(1)
    factor_one!(input,output,n,Int32(gid))
    return nothing
end
launch_svd(w)=(@cuda threads=256 blocks=w.n÷256+1 factor_kernel!(w.di,w.dout,Int32(w.n)))
include("../study_common.jl")
study_main("cuda_jl")

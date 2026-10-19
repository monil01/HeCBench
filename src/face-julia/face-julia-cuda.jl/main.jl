using CUDA
const MODEL="cuda_jl"
include("../study_device.jl")
function address_kernel!(args...)
 i=(Int32(CUDA.blockIdx().x)-Int32(1))*Int32(CUDA.blockDim().x)+Int32(CUDA.threadIdx().x)-Int32(1)
 address_body!(args...,i)
 return
end
launch_address(a,b,p,w,n)=CUDA.@cuda threads=256 blocks=cld(n,256) address_kernel!(a,b,p,w,n)
include("../study_common.jl")
main()

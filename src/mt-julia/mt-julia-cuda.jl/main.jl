using CUDA
const MODEL="cuda_jl"
include("../study_device.jl")
function kernel_mt!(parameters,output,n)
 gid=(Int32(blockIdx().x)-Int32(1))*Int32(blockDim().x)+Int32(threadIdx().x)-Int32(1)
 body_mt!(gid,parameters,output,n)
 nothing
end
function kernel_boxmuller!(output,n)
 gid=(Int32(blockIdx().x)-Int32(1))*Int32(blockDim().x)+Int32(threadIdx().x)-Int32(1)
 body_boxmuller!(gid,output,n)
 nothing
end
function dispatch_mt!(p,r,n)
 CUDA.@cuda threads=128 blocks=32 kernel_mt!(p,r,n)
 nothing
end
function dispatch_boxmuller!(r,n)
 CUDA.@cuda threads=128 blocks=32 kernel_boxmuller!(r,n)
 nothing
end
include("../study_common.jl")
main()

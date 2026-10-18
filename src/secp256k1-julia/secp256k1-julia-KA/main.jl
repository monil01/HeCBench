using CUDA,KernelAbstractions
include(joinpath(@__DIR__,"..","field_math.jl"))
@kernel unsafe_indices=true function secp_kernel!(out,points)
 bytes=secp_compute(points)
 @inbounds for j in 1:32;out[j]=bytes[j];end
end
const ST_KERNEL=secp_kernel!(CUDABackend(),1)
function study_launch!(f)
 ST_KERNEL(f[2],f[1];ndrange=1)
end
include(joinpath(@__DIR__,"..","study_common.jl"))
study_main("ka",study_launch!)

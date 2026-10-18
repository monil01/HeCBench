using CUDA
include(joinpath(@__DIR__,"..","field_math.jl"))
function secp_kernel!(out,points)
 bytes=secp_compute(points)
 @inbounds for j in 1:32;out[j]=bytes[j];end
 return
end
function study_launch!(f)
 @cuda threads=1 blocks=1 secp_kernel!(f[2],f[1])
end
include(joinpath(@__DIR__,"..","study_common.jl"))
if abspath(PROGRAM_FILE)==(@__FILE__);study_main("cuda_jl",study_launch!);end

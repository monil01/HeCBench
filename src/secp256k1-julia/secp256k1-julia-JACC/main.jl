using CUDA,JACC
include(joinpath(@__DIR__,"..","field_math.jl"))
function secp_callback(i,out,points)
 bytes=secp_compute(points)
 @inbounds for j in 1:32;out[j]=bytes[j];end
 nothing
end
const BACKEND=JACC.get_backend(:cuda)
const SPEC=JACC.LaunchSpec{typeof(BACKEND)}(;threads=1,blocks=1,shmem_size=0,sync=false)
function study_launch!(f)
 JACC.parallel_for(secp_callback,SPEC,1,f[2],f[1])
end
include(joinpath(@__DIR__,"..","study_common.jl"))
study_main("jacc",study_launch!)

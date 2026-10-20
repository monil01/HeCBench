using CUDA,Libdl,SHA
include(joinpath(@__DIR__,"..","device_bodies.jl"))
function xl_launch(op,pbf,mapping,n,D,out,Q)
    entries=(cuda_mat_J_PI,cuda_mat_K_PI,cuda_mat_J_PI_dp,cuda_mat_K_PI_dp)
    0<=op<4||error("Unknown source entry")
    kernel=entries[Int(op)+1]
    @cuda threads=(8,8) blocks=(Int(n),Int(n)) stream=XL_STREAM kernel(pbf,mapping,n,D,out,Q)
    nothing
end
include(joinpath(@__DIR__,"..","study_common.jl"))
xl_study()

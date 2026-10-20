using CUDA,Libdl,SHA,JACC
# JACC owns the two-dimensional dispatch; original CUDA backend indices/shared/barriers remain explicit.
include(joinpath(@__DIR__,"..","device_bodies.jl"))
const XL_JACC_SPECS=Dict{Tuple{Int,Int},Any}()
@inline function xl_jacc_J_sp!(unused_i,unused_j,pbf,mapping,n,D,out,Q)
    cuda_mat_J_PI(pbf,mapping,n,D,out,Q)
    nothing
end
@inline function xl_jacc_K_sp!(unused_i,unused_j,pbf,mapping,n,D,out,Q)
    cuda_mat_K_PI(pbf,mapping,n,D,out,Q)
    nothing
end
@inline function xl_jacc_J_dp!(unused_i,unused_j,pbf,mapping,n,D,out,Q)
    cuda_mat_J_PI_dp(pbf,mapping,n,D,out,Q)
    nothing
end
@inline function xl_jacc_K_dp!(unused_i,unused_j,pbf,mapping,n,D,out,Q)
    cuda_mat_K_PI_dp(pbf,mapping,n,D,out,Q)
    nothing
end
function xl_launch(op,pbf,mapping,n,D,out,Q)
    entries=(xl_jacc_J_sp!,xl_jacc_K_sp!,xl_jacc_J_dp!,xl_jacc_K_dp!)
    0<=op<4||error("Unknown source entry")
    spec=get!(XL_JACC_SPECS,(Int(op),Int(n))) do
        JACC.launch_spec(;threads=(8,8),blocks=(Int(n),Int(n)),shmem_size=0,sync=false,stream=XL_STREAM)
    end
    JACC.parallel_for(spec,(8*Int(n),8*Int(n)),entries[Int(op)+1],pbf,mapping,n,D,out,Q)
    nothing
end
include(joinpath(@__DIR__,"..","study_common.jl"))
xl_study()

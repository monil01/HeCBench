using CUDA,Libdl,SHA,KernelAbstractions
# KA owns dispatch. Source CUDA block indices/shared storage/barriers remain explicit backend use.
include(joinpath(@__DIR__,"device_bodies.jl"))
const XL_KA_BACKEND=CUDA.CUDABackend(always_inline=true)
@kernel unsafe_indices=true function xl_ka_J_sp!(pbf,mapping,n,D,out,Q)
    li=@index(Local,Cartesian);gi=@index(Group,Cartesian)
    cuda_mat_J_PI(pbf,mapping,n,D,out,Q,Int32(li[1])-Int32(1),Int32(li[2])-Int32(1),Int32(gi[1])-Int32(1),Int32(gi[2])-Int32(1))
end
@kernel unsafe_indices=true function xl_ka_K_sp!(pbf,mapping,n,D,out,Q)
    li=@index(Local,Cartesian);gi=@index(Group,Cartesian)
    cuda_mat_K_PI(pbf,mapping,n,D,out,Q,Int32(li[1])-Int32(1),Int32(li[2])-Int32(1),Int32(gi[1])-Int32(1),Int32(gi[2])-Int32(1))
end
@kernel unsafe_indices=true function xl_ka_J_dp!(pbf,mapping,n,D,out,Q)
    li=@index(Local,Cartesian);gi=@index(Group,Cartesian)
    cuda_mat_J_PI_dp(pbf,mapping,n,D,out,Q,Int32(li[1])-Int32(1),Int32(li[2])-Int32(1),Int32(gi[1])-Int32(1),Int32(gi[2])-Int32(1))
end
@kernel unsafe_indices=true function xl_ka_K_dp!(pbf,mapping,n,D,out,Q)
    li=@index(Local,Cartesian);gi=@index(Group,Cartesian)
    cuda_mat_K_PI_dp(pbf,mapping,n,D,out,Q,Int32(li[1])-Int32(1),Int32(li[2])-Int32(1),Int32(gi[1])-Int32(1),Int32(gi[2])-Int32(1))
end
function xl_launch(op,pbf,mapping,n,D,out,Q)
    entries=(xl_ka_J_sp!,xl_ka_K_sp!,xl_ka_J_dp!,xl_ka_K_dp!)
    0<=op<4||error("Unknown source entry")
    kernel=entries[Int(op)+1](XL_KA_BACKEND,(8,8))
    kernel(pbf,mapping,n,D,out,Q;ndrange=(8*Int(n),8*Int(n)))
    nothing
end
include(joinpath(@__DIR__,"..","study_common.jl"))
xl_study()

using CUDA,KernelAbstractions
include(joinpath(@__DIR__,"..","warp_network.jl"))
@kernel unsafe_indices=true function collision_flags!(a,out)
 lane=Int32(@index(Local,Linear))-Int32(1)
 out[lane+Int32(1)]=collision_flag(a[lane+Int32(1)],lane)
end
@kernel unsafe_indices=true function collision_masks!(a,out)
 lane=Int32(@index(Local,Linear))-Int32(1);mask=collision_mask(a[lane+Int32(1)],lane)
 if lane==0;out[1]=mask;end
end
include(joinpath(@__DIR__,"..","study_common.jl"))
const CK_FLAGS=collision_flags!(CUDABackend(),32)
const CK_MASK=collision_masks!(CUDABackend(),32)
function launch_collision!(a,f,m)
 CK_FLAGS(a,f;ndrange=32)
 CK_MASK(a,m;ndrange=32)
end
collision_study(launch_collision!)

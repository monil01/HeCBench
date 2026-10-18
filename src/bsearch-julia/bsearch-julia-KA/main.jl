using CUDA,KernelAbstractions
include(joinpath(@__DIR__,"..","search_math.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
@kernel unsafe_indices=true function search_ka!(a,z,r,n,mode::Val{MODE}) where {MODE}
 i=@index(Global,Linear)
 if i<=512
  @inbounds r[i]=MODE==1 ? search_basic(a,z[i],n) : search_walk(a,z[i],n,search_seed(n),mode)
 end
end
@kernel unsafe_indices=true function search_shared_ka!(a,z,r,n)
 i=@index(Global,Linear);tid=@index(Local,Linear)
 seed=@localmem UInt64 (1,)
 if tid==1;@inbounds seed[1]=search_seed(n);end
 @synchronize
 if i<=512;@inbounds r[i]=search_walk(a,z[i],n,seed[1],Val(3));end
end
function study_launch!(a,z,r,n,mode::Val{MODE}) where {MODE}
 if MODE==4
  search_shared_ka!(CUDABackend(),256)(a,z,r,n;ndrange=512)
 else
  search_ka!(CUDABackend(),256)(a,z,r,n,mode;ndrange=512)
 end
end
study_main(study_launch!)

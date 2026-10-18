using CUDA,JACC
include(joinpath(@__DIR__,"..","search_math.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
const SEARCH_BACKEND=JACC.get_backend(:cuda)
const SEARCH_PRIVATE=JACC.LaunchSpec{typeof(SEARCH_BACKEND)}(;threads=256,blocks=2,shmem_size=0,sync=false)
const SEARCH_SHARED=JACC.LaunchSpec{typeof(SEARCH_BACKEND)}(;threads=256,blocks=2,shmem_size=8,sync=false)
const SEARCH_TEMPLATE=CUDA.zeros(UInt64,1)
function search_jacc!(i,a,z,r,n,mode::Val{MODE},template) where {MODE}
 if MODE==1
  @inbounds r[i]=search_basic(a,z[i],n)
 elseif MODE==4
  seed=JACC.shared(template)
  if (i-Int32(1))%Int32(256)==0
   @inbounds seed[1]=search_seed(n)
  end
  JACC.sync_workgroup()
  @inbounds r[i]=search_walk(a,z[i],n,seed[1],Val(3))
 else
  @inbounds r[i]=search_walk(a,z[i],n,search_seed(n),mode)
 end
 nothing
end
function study_launch!(a,z,r,n,mode::Val{MODE}) where {MODE}
 spec=MODE==4 ? SEARCH_SHARED : SEARCH_PRIVATE
 JACC.parallel_for(search_jacc!,spec,Int32(512),a,z,r,n,mode,SEARCH_TEMPLATE)
end
study_main(study_launch!)

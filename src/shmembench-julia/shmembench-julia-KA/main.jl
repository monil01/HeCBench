using CUDA,KernelAbstractions
include("../study_portable_common.jl")
@kernel unsafe_indices=true function shmem_kernel(output)
 tid=Int32(@index(Local,Linear))-Int32(1);gid=(Int32(@index(Group,Linear))-Int32(1))*Int32(256)+tid+Int32(1);shared=@localmem SHMEM_VEC4 (1536,)

 @inbounds shared[tid+Int32(1)]=shmem_init(tid)
 @inbounds shared[tid+Int32(257)]=shmem_init(tid+Int32(1))
 @inbounds shared[tid+Int32(513)]=shmem_init(tid+Int32(3))
 @inbounds shared[tid+Int32(769)]=shmem_init(tid+Int32(7))
 @inbounds shared[tid+Int32(1025)]=shmem_init(tid+Int32(13))
 @inbounds shared[tid+Int32(1281)]=shmem_init(tid+Int32(17))
 @synchronize
 for j in 1:1024
  shmem_swap!(shared,tid+Int32(1),tid+Int32(257))
  shmem_swap!(shared,tid+Int32(513),tid+Int32(769))
  shmem_swap!(shared,tid+Int32(1025),tid+Int32(1281))
  @synchronize
  shmem_swap!(shared,tid+Int32(257),tid+Int32(513))
  shmem_swap!(shared,tid+Int32(769),tid+Int32(1025))
  @synchronize
 end
 @inbounds output[gid]=shmem_reduce(shared,tid)
end
function factory(output)
 k=shmem_kernel(CUDABackend(),256)
 ()->k(output;ndrange=1024*256)
end
shmem_study(factory)

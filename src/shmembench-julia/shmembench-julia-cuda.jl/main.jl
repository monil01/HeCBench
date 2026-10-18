using CUDA
include("../study_common.jl")
function shmem_kernel(output)
 tid=Int32(threadIdx().x)-Int32(1);gid=(Int32(blockIdx().x)-Int32(1))*Int32(256)+tid+Int32(1);shared=CuStaticSharedArray(SHMEM_VEC4,1536)

 @inbounds shared[tid+Int32(1)]=shmem_init(tid)
 @inbounds shared[tid+Int32(257)]=shmem_init(tid+Int32(1))
 @inbounds shared[tid+Int32(513)]=shmem_init(tid+Int32(3))
 @inbounds shared[tid+Int32(769)]=shmem_init(tid+Int32(7))
 @inbounds shared[tid+Int32(1025)]=shmem_init(tid+Int32(13))
 @inbounds shared[tid+Int32(1281)]=shmem_init(tid+Int32(17))
 sync_threads()
 for j in 1:1024
  shmem_swap!(shared,tid+Int32(1),tid+Int32(257))
  shmem_swap!(shared,tid+Int32(513),tid+Int32(769))
  shmem_swap!(shared,tid+Int32(1025),tid+Int32(1281))
  sync_threads()
  shmem_swap!(shared,tid+Int32(257),tid+Int32(513))
  shmem_swap!(shared,tid+Int32(769),tid+Int32(1025))
  sync_threads()
 end
 @inbounds output[gid]=shmem_reduce(shared,tid)
 return
end
factory(output)=()->(@cuda threads=256 blocks=1024 shmem_kernel(output))
shmem_study(factory)

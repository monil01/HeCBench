using CUDA,JACC
include("../study_portable_common.jl")
@inline function shmem_kernel(i,output,template)
 tid=Int32((i-1)%256);gid=Int32(i);shared=JACC.shared(template)

 @inbounds shared[tid+Int32(1)]=shmem_init(tid)
 @inbounds shared[tid+Int32(257)]=shmem_init(tid+Int32(1))
 @inbounds shared[tid+Int32(513)]=shmem_init(tid+Int32(3))
 @inbounds shared[tid+Int32(769)]=shmem_init(tid+Int32(7))
 @inbounds shared[tid+Int32(1025)]=shmem_init(tid+Int32(13))
 @inbounds shared[tid+Int32(1281)]=shmem_init(tid+Int32(17))
 JACC.sync_workgroup()
 for j in 1:1024
  shmem_swap!(shared,tid+Int32(1),tid+Int32(257))
  shmem_swap!(shared,tid+Int32(513),tid+Int32(769))
  shmem_swap!(shared,tid+Int32(1025),tid+Int32(1281))
  JACC.sync_workgroup()
  shmem_swap!(shared,tid+Int32(257),tid+Int32(513))
  shmem_swap!(shared,tid+Int32(769),tid+Int32(1025))
  JACC.sync_workgroup()
 end
 @inbounds output[gid]=shmem_reduce(shared,tid)
 return
end
const BACKEND=JACC.get_backend(:cuda)
function factory(output)
 template=CuArray(fill(ntuple(_->VecElement(0f0),Val(4)),1536))
 spec=JACC.LaunchSpec{typeof(BACKEND)}(;threads=256,blocks=1024,shmem_size=24576,sync=false)
 ()->JACC.parallel_for(spec,1024*256,shmem_kernel,output,template)
end
shmem_study(factory)

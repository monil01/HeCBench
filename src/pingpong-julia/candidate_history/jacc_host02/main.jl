using CUDA, JACC
function increment_mpi(i,d,n)
 q=Int32(i-1)
 while q<n
  @inbounds d[q+1]+=1.0
  q+=Int32(1024*256)
 end
 return
end
function increment_nccl(i,d,n)
 q=Int32(i-1)
 while q<n
  @inbounds d[q+1]+=1.0
  q+=Int32(1024*256)
 end
 return
end
allocate_raw(h)=JACC.Array(h)
const SPEC=JACC.launch_spec(threads=256,blocks=1024,shmem_size=0,sync=false)
function increment!(d,n,variant)
 if variant==0
  JACC.parallel_for(SPEC,1024*256,increment_mpi,d,Int32(n))
 else
  JACC.parallel_for(SPEC,1024*256,increment_nccl,d,Int32(n))
 end
 return
end
include("../study_common.jl")

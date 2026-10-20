using CUDA, KernelAbstractions
@kernel function increment_mpi(d,n)
 i=@index(Global,Linear)
 q=Int32(i-1)
 while q<n
  @inbounds d[q+1]+=1.0
  q+=Int32(1024*256)
 end
end
@kernel function increment_nccl(d,n)
 i=@index(Global,Linear)
 q=Int32(i-1)
 while q<n
  @inbounds d[q+1]+=1.0
  q+=Int32(1024*256)
 end
end
function allocate_raw(h)
 a=KernelAbstractions.allocate(CUDABackend(),Float64,(length(h),));copyto!(a,h);return a
end
function increment!(d,n,variant)
 if variant==0
  increment_mpi(CUDABackend(),256)(d,Int32(n);ndrange=1024*256)
 else
  increment_nccl(CUDABackend(),256)(d,Int32(n);ndrange=1024*256)
 end
 return
end
include("../study_common.jl")

using CUDA
function increment_mpi(d,n)
 i=Int32((blockIdx().x-1)*blockDim().x+threadIdx().x-1)
 while i<n
  @inbounds d[i+1]+=1.0
  i+=Int32(1024*256)
 end
 return
end
function increment_nccl(d,n)
 i=Int32((blockIdx().x-1)*blockDim().x+threadIdx().x-1)
 while i<n
  @inbounds d[i+1]+=1.0
  i+=Int32(1024*256)
 end
 return
end
allocate_raw(h)=CuArray(h)
function increment!(d,n,variant)
 if variant==0
  CUDA.@cuda threads=256 blocks=1024 increment_mpi(d,Int32(n))
 else
  CUDA.@cuda threads=256 blocks=1024 increment_nccl(d,Int32(n))
 end
 return
end
include("../study_common.jl")

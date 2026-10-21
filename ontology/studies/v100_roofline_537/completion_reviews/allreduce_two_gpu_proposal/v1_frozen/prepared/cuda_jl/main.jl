using CUDA
function add_kernel!(src,out,offset,n)
 i=Int32((blockIdx().x-1)*blockDim().x+threadIdx().x-1)
 stride=Int32(blockDim().x*gridDim().x)
 while i<n
  @inbounds out[offset+i+1]=src[i+1]+out[offset+i+1]
  i+=stride
 end
 return
end
reduce(c,offset,n)=(@cuda threads=256 blocks=32 add_kernel!(c.tmp,c.output,Int32(offset),Int32(n)))
alloc(v)=CuArray(v)
include("/tmp/roofline-completion-20261021/allreduce_two_gpu_proposal/prepared/study_common.jl")
study(alloc,reduce)

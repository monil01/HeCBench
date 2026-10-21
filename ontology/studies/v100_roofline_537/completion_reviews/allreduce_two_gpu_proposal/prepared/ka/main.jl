using CUDA,KernelAbstractions
@kernel function add_kernel!(src,out,offset,n)
 i=Int32(@index(Global,Linear)-1)
 while i<n
  @inbounds out[offset+i+1]=src[i+1]+out[offset+i+1]
  i+=Int32(8192)
 end
 nothing
end
const BACKEND=CUDABackend()
reduce(c,offset,n)=add_kernel!(BACKEND,256)(c.tmp,c.output,Int32(offset),Int32(n);ndrange=8192)
function alloc(v)
 a=KernelAbstractions.allocate(BACKEND,Float32,(length(v),));copyto!(a,v);a
end
include("/tmp/roofline-completion-20261021/allreduce_two_gpu_proposal/prepared/study_common.jl")
study(alloc,reduce)

using CUDA
import JACC
JACC.@init_backend
function add_kernel!(i,src,out,offset,n)
 index=Int32(i-1)
 while index<n
  @inbounds out[offset+index+1]=src[index+1]+out[offset+index+1]
  index+=Int32(8192)
 end
 return
end
const SPEC=JACC.launch_spec(threads=256,blocks=32,shmem_size=0,sync=false)
reduce(c,offset,n)=JACC.parallel_for(SPEC,8192,add_kernel!,c.tmp,c.output,Int32(offset),Int32(n))
function alloc(v)
 a=JACC.zeros(Float32,length(v));copyto!(a,v);a
end
include(joinpath(@__DIR__,"..","study_common.jl"))
study(alloc,reduce)

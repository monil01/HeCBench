using CUDA,JACC
import Atomix
include(joinpath(@__DIR__,"..","study_common.jl"))
function mx_reduce!(index,values,minbits,n::Int32,dim::Int32)
 i=Int32(index)
 if i<=n
  value=mx_value(values,Int32(4)+(i-Int32(1))*dim,dim)
  @inbounds old=minbits[5]
  while value<reinterpret(Float32,old)
   observed=(Atomix.@atomicreplace :monotonic minbits[5] old => reinterpret(UInt32,value)).old
   observed==old && break
   old=observed
  end
 end
 return
end
function mx_capture!(index,values,out,n::Int32,dim::Int32)
 i=Int32(index)
 if i<=n;@inbounds out[4+i]=mx_value(values,Int32(4)+(i-Int32(1))*dim,dim);end
 return
end
function factory(values,minbits,out,n,dim)
 spec=JACC.launch_spec(;threads=256,shmem_size=0,sync=false)
 reduce! =()->JACC.parallel_for(spec,n,mx_reduce!,values,minbits,n,dim)
 capture! =()->JACC.parallel_for(spec,n,mx_capture!,values,out,n,dim)
 reduce!,capture!
end
michalewicz_study(factory)

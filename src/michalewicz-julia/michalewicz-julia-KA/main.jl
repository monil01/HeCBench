using CUDA,KernelAbstractions
import Atomix
include(joinpath(@__DIR__,"..","study_common.jl"))
@kernel function mx_reduce!(values,minbits,n::Int32,dim::Int32)
 i=Int32(@index(Global,Linear))
 if i<=n
  value=mx_value(values,Int32(4)+(i-Int32(1))*dim,dim)
  @inbounds old=minbits[5]
  while value<reinterpret(Float32,old)
   observed=(Atomix.@atomicreplace :monotonic minbits[5] old => reinterpret(UInt32,value)).old
   observed==old && break
   old=observed
  end
 end
end
@kernel function mx_capture!(values,out,n::Int32,dim::Int32)
 i=Int32(@index(Global,Linear))
 if i<=n;@inbounds out[4+i]=mx_value(values,Int32(4)+(i-Int32(1))*dim,dim);end
end
function factory(values,minbits,out,n,dim)
 reducekernel=mx_reduce!(CUDABackend(),256);capturekernel=mx_capture!(CUDABackend(),256)
 reduce! =()->reducekernel(values,minbits,n,dim;ndrange=n)
 capture! =()->capturekernel(values,out,n,dim;ndrange=n)
 reduce!,capture!
end
michalewicz_study(factory)

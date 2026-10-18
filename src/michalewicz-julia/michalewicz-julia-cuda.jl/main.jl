using CUDA
include(joinpath(@__DIR__,"..","study_common.jl"))
function mx_reduce!(values,minbits,n::Int32,dim::Int32)
 i=(blockIdx().x-Int32(1))*blockDim().x+threadIdx().x
 if i<=n
  value=mx_value(values,Int32(4)+(i-Int32(1))*dim,dim)
  @inbounds old=minbits[5]
  while value<reinterpret(Float32,old)
   observed=CUDA.atomic_cas!(pointer(minbits,5),old,reinterpret(UInt32,value))
   observed==old && break
   old=observed
  end
 end
 return
end
function mx_capture!(values,out,n::Int32,dim::Int32)
 i=(blockIdx().x-Int32(1))*blockDim().x+threadIdx().x
 if i<=n;@inbounds out[4+i]=mx_value(values,Int32(4)+(i-Int32(1))*dim,dim);end
 return
end
function factory(values,minbits,out,n,dim)
 reduce! =()->(@cuda threads=256 blocks=cld(n,256) mx_reduce!(values,minbits,n,dim))
 capture! =()->(@cuda threads=256 blocks=cld(n,256) mx_capture!(values,out,n,dim))
 reduce!,capture!
end
michalewicz_study(factory)

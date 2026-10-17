using CUDA
include(joinpath(@__DIR__,"..","study_common.jl"))
function kernel!(a,b,output,na,nb,slot)
 i=(blockIdx().x-1)*blockDim().x+threadIdx().x
 if i<=na
  @inbounds point=a[i]
  closest=floatmax(Float32)
  for j in 1:nb
   @inbounds other=b[j]
   dx=point[1]-other[1];dy=point[2]-other[2]
   closest=min(closest,dx*dx+dy*dy)
  end
  @inbounds bits=output[slot]
  while closest>reinterpret(Float32,bits)
   next=reinterpret(UInt32,closest)
   previous=CUDA.atomic_cas!(pointer(output,slot),bits,next)
   previous==bits&&break
   bits=previous
  end
 end
 return
end
function run!(a,b,out,na,nb)
 @cuda threads=256 blocks=cld(na,256) kernel!(a,b,out,na,nb,Int32(1))
 @cuda threads=256 blocks=cld(nb,256) kernel!(b,a,out,nb,na,Int32(2))
 return
end
study(run!)

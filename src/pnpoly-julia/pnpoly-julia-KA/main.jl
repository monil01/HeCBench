using CUDA,KernelAbstractions
include(joinpath(@__DIR__,"..","study_common.jl"))
@kernel function kernel!(points,vertex,out,::Val{K}) where K
 t=@index(Local,Linear);group=@index(Group,Linear)
 if K==0
  base_i=(group-1)*256+t
  if base_i<=16384
   @inbounds q=points[base_i];c=false;k=600
   for j in 1:600
    @inbounds c=xor(c,cross(q,vertex[j],vertex[k]));k=j
   end
   @inbounds out[base_i]=Int32(c)
  end
 else
  i=(group-1)*256*K+t
  if i<=16384
   qs=ntuple(z->(@inbounds points[i+256*(z-1)]),Val(K));cs=ntuple(z->false,Val(K));k=600
   for j in 1:600
    @inbounds vj=vertex[j];vk=vertex[k]
    old=cs;cs=ntuple(z->xor(old[z],cross(qs[z],vj,vk)),Val(K));k=j
   end
   for z in 1:K
    @inbounds out[i+256*(z-1)]=Int32(cs[z])
   end
  end
 end
end
run!(p,v,out,mode)=kernel!(get_backend(out),256)(p,v,out,mode;ndrange=256*64)
study(run!)

using CUDA,JACC
include(joinpath(@__DIR__,"..","study_common.jl"))
function kernel!(index,points,vertex,out,::Val{K}) where K
 t=mod(index-1,256)+1;group=div(index-1,256)+1
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
run!(p,v,out,mode)=JACC.parallel_for(JACC.launch_spec(threads=256,blocks=64,shmem_size=0,sync=false),16384,kernel!,p,v,out,mode)
study(run!)

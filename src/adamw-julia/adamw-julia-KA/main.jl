using CUDA,KernelAbstractions
include(joinpath(@__DIR__,"..","study_common.jl"))
@kernel function kernel!(p,g,ms,vs,m,v,template,c2,ss)
 t=@index(Local,Linear);group=@index(Group,Linear);shared=@localmem Float32 (128,)
 i=(group-1)*64+t
 pl,pr,ml,mr,vl,vr=update(i,group,p,g,ms,vs,m,v,c2,ss)
 @inbounds begin
  p[2i-1]=pl;p[2i]=pr;shared[t]=max(ml,mr);shared[t+64]=max(vl,vr)
 end
 @synchronize
 stride=32
 while stride>=1
  if t<=stride
   @inbounds shared[t]=max(shared[t],shared[t+stride]);shared[t+64]=max(shared[t+64],shared[t+64+stride])
  end
  @synchronize
  stride>>=1
 end
 @inbounds em=shared[1];vm=shared[65]
 if t==1;@inbounds ms[group]=em;vs[group]=vm;end
 @synchronize
 @inbounds m[i]=pack_native(quant(EM,EC,ml/em),quant(EM,EC,mr/em));v[i]=pack_native(quant(VM,VC,vl/vm),quant(VM,VC,vr/vm))
end
function run!(args...)
 for (_,c2,ss) in PARAMS
  kernel!(get_backend(args[1]),64)(args...,c2,ss;ndrange=4096)
 end
 return
end
study(run!)

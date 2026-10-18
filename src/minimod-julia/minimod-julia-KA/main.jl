using CUDA,KernelAbstractions
include("../wave_device.jl")
@kernel unsafe_indices=true function wave_kernel!(u,v,vp,phi,eta,b,c0,c,hd,ispml)
 t=@index(Local,NTuple);g=@index(Group,NTuple);tx=Int(t[1])-1;ty=Int(t[2])-1;tz=Int(t[3])-1;bx=Int(g[1])-1;by=Int(g[2])-1;bz=Int(g[3])-1
 ss=@localmem Float32 (4096,)
 wave_zero!(ss,tx,ty,tz);@synchronize
 loc=wave_load!(ss,u,tx,ty,tz,bx,by,bz,b);@synchronize
 wave_update!(ss,u,v,vp,phi,eta,loc,b,c0,c,hd,ispml)
end
@kernel unsafe_indices=true function source_kernel!(v,value)
 point_source!(v,value)
end
const WK=wave_kernel!(CUDABackend(),(8,8,8));const SK=source_kernel!(CUDABackend(),1)
function launch(a)
 u,v,phi,vp,eta=a.vs
 for step in 1:a.ns
  for (i,b) in enumerate(WAVE_REGIONS)
   WK(u,v,vp,phi,eta,b,a.c0,a.c,a.hd,Val(i!=4);ndrange=(b[6]-b[5],b[4]-b[3],b[2]-b[1]))
  end
  SK(v,a.source[step+1];ndrange=1);u,v=v,u
 end
 nothing
end
include("../study_common.jl")
wave_study(launch)

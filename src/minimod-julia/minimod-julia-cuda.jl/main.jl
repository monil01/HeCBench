using CUDA
include("../wave_device.jl")
function wave_kernel(u,v,vp,phi,eta,b,c0,c,hd,ispml)
 tx=Int(threadIdx().x)-1;ty=Int(threadIdx().y)-1;tz=Int(threadIdx().z)-1;bx=Int(blockIdx().x)-1;by=Int(blockIdx().y)-1;bz=Int(blockIdx().z)-1
 ss=CuStaticSharedArray(Float32,4096);wave_zero!(ss,tx,ty,tz);sync_threads();loc=wave_load!(ss,u,tx,ty,tz,bx,by,bz,b);sync_threads();wave_update!(ss,u,v,vp,phi,eta,loc,b,c0,c,hd,ispml);nothing
end
source_kernel(v,value)=(point_source!(v,value);nothing)
function launch(a)
 u,v,phi,vp,eta=a.vs
 for step in 1:a.ns
  for (i,b) in enumerate(WAVE_REGIONS)
   @cuda threads=(8,8,8) blocks=((b[6]-b[5])÷8,(b[4]-b[3])÷8,(b[2]-b[1])÷8) wave_kernel(u,v,vp,phi,eta,b,a.c0,a.c,a.hd,Val(i!=4))
  end
  @cuda threads=1 blocks=1 source_kernel(v,a.source[step+1]);u,v=v,u
 end
 nothing
end
include("../study_common.jl")
wave_study(launch)

using CUDA
include("../metropolis_device.jl")
function metropolis_kernel(S,H,A,I,h,B,alt)
 tx=Int(threadIdx().x)-1;ty=Int(threadIdx().y)-1;tz=Int(threadIdx().z)-1;bx=Int(blockIdx().x)-1;by=Int(blockIdx().y)-1;bz=Int(blockIdx().z)-1
 ss=CuStaticSharedArray(Int32,1080);c,st,inc=tile_load!(ss,S,H,A,I,tx,ty,tz,bx,by,bz,alt);sync_threads();st=metropolis_phase!(ss,c,h,B,st,inc,Val(0));sync_threads();st=metropolis_phase!(ss,c,h,B,st,inc,Val(1));sync_threads();tile_store!(ss,S,A,I,c,st,inc);nothing
end
function launch(c)
 for step in 1:c.ns,alt in 0:1
  @cuda threads=(16,4,4) blocks=(2,2,8) metropolis_kernel(c.vs...,c.h,c.B,alt)
  CUDA.synchronize()
 end
 nothing
end
include("../study_common.jl")
metropolis_study(launch)

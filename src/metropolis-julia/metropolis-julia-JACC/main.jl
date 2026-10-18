using CUDA,JACC
include("../metropolis_device.jl")
const TEMPLATE=CUDA.zeros(Int32,1080)
const SPEC=JACC.LaunchSpec{typeof(JACC.get_backend(:cuda))}(;threads=(16,16),blocks=(2,16),shmem_size=4320,sync=false)
function metropolis_kernel!(i,j,S,H,A,I,h,B,alt,template)
 tx=(i-1)%16;bx=(i-1)÷16;ty=((j-1)%16)%4;tz=((j-1)%16)÷4;by=((j-1)÷16)%2;bz=((j-1)÷16)÷2
 ss=JACC.shared(template);c,st,inc=tile_load!(ss,S,H,A,I,tx,ty,tz,bx,by,bz,alt);JACC.sync_workgroup()
 st=metropolis_phase!(ss,c,h,B,st,inc,Val(0));JACC.sync_workgroup()
 st=metropolis_phase!(ss,c,h,B,st,inc,Val(1));JACC.sync_workgroup()
 tile_store!(ss,S,A,I,c,st,inc);nothing
end
function launch(c)
 for step in 1:c.ns,alt in 0:1
  JACC.parallel_for(metropolis_kernel!,SPEC,(32,256),c.vs...,c.h,c.B,alt,TEMPLATE);CUDA.synchronize()
 end
 nothing
end
include("../study_common.jl")
metropolis_study(launch)

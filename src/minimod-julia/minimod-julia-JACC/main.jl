using CUDA,JACC
include("../wave_device.jl")
const TEMPLATE=CUDA.zeros(Float32,4096)
const SOURCE_SPEC=JACC.LaunchSpec{typeof(JACC.get_backend(:cuda))}(;threads=1,blocks=1,shmem_size=0,sync=false)
function wave_kernel!(i,j,u,v,vp,phi,eta,b,c0,c,hd,ispml,template)
 tx=(i-1)%8;bx=(i-1)÷8;ty=((j-1)%64)%8;tz=((j-1)%64)÷8;ny=(b[4]-b[3])÷8;by=((j-1)÷64)%ny;bz=((j-1)÷64)÷ny
 ss=JACC.shared(template);wave_zero!(ss,tx,ty,tz);JACC.sync_workgroup();loc=wave_load!(ss,u,tx,ty,tz,bx,by,bz,b);JACC.sync_workgroup();wave_update!(ss,u,v,vp,phi,eta,loc,b,c0,c,hd,ispml);nothing
end
source_kernel!(i,v,value)=(point_source!(v,value);nothing)
function launch(a)
 u,v,phi,vp,eta=a.vs
 for step in 1:a.ns
  for (i,b) in enumerate(WAVE_REGIONS)
   gx=(b[6]-b[5])÷8;gy=(b[4]-b[3])÷8;gz=(b[2]-b[1])÷8
   spec=JACC.LaunchSpec{typeof(JACC.get_backend(:cuda))}(;threads=(8,64),blocks=(gx,gy*gz),shmem_size=16384,sync=false)
   JACC.parallel_for(wave_kernel!,spec,(gx*8,gy*gz*64),u,v,vp,phi,eta,b,a.c0,a.c,a.hd,Val(i!=4),TEMPLATE)
  end
  JACC.parallel_for(source_kernel!,SOURCE_SPEC,1,v,a.source[step+1]);u,v=v,u
 end
 nothing
end
include("../study_common.jl")
wave_study(launch)

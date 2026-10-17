using CUDA,JACC
JACC.@init_backend
include(joinpath(@__DIR__,"..","study_common.jl"))
function mean_base(index,data,out)
 tid=Int32(index-1)
 if tid<10000
 @inbounds begin
 # Actual source D3, N10000, squared radius60 and Gaussian denominator32.
 row=3tid;x=data[row+1];y=data[row+2];z=data[row+3];a=b=c=total=0f0
  for j in Int32(0):Int32(9999)
   u=data[3j+1];v=data[3j+2];w=data[3j+3];dx=x-u;dy=y-v;dz=z-w;distance=(dx*dx+dy*dy)+dz*dz
   if distance<=60f0
    weight=exp(-distance/32f0);a+=weight*u;b+=weight*v;c+=weight*w;total+=weight
   end
  end
 out[row+1]=a/total;out[row+2]=b/total;out[row+3]=c/total
 end
 end
end
function mean_tiled(index,data,out,template)
 tid=Int32(index-1);local_data=JACC.shared(template)
 @inbounds begin
 # Actual source D3, N10000, squared radius60 and Gaussian denominator32.
 row=3tid;x=data[row+1];y=data[row+2];z=data[row+3];a=b=c=total=0f0
  lid=tid%64
  for tile in Int32(0):Int32(156)
   other=tile*64+lid
   if other<10000
    local_data[3lid+1]=data[3other+1];local_data[3lid+2]=data[3other+2];local_data[3lid+3]=data[3other+3];local_data[193+lid]=1f0
   else
    local_data[3lid+1]=0f0;local_data[3lid+2]=0f0;local_data[3lid+3]=0f0;local_data[193+lid]=0f0
   end
   JACC.sync_workgroup()
   for j in Int32(0):Int32(63)
    u=local_data[3j+1];v=local_data[3j+2];w=local_data[3j+3];valid=local_data[193+j];dx=x-u;dy=y-v;dz=z-w;distance=(dx*dx+dy*dy)+dz*dz
    if distance<=60f0*valid
     weight=exp(-distance/32f0);a+=weight*u;b+=weight*v;c+=weight*w;total+=weight*valid
    end
   end
   JACC.sync_workgroup()
  end
 if tid<10000;out[row+1]=a/total;out[row+2]=b/total;out[row+3]=c/total;end
 end
end
function factory(a,b,mode)
 tiled=mode=="tiled";spec=JACC.launch_spec(;threads=64,blocks=157,shmem_size=tiled ? 1024 : 0,sync=false);template=CUDA.zeros(Float32,256)
 ()->begin
  x=a;y=b
  for iteration in 1:1000
   if tiled;JACC.parallel_for(spec,10048,mean_tiled,x,y,template);else;JACC.parallel_for(spec,10048,mean_base,x,y);end
   CUDA.synchronize();x,y=y,x
  end
 end
end
if abspath(PROGRAM_FILE)==@__FILE__;meanshift_study(factory);end

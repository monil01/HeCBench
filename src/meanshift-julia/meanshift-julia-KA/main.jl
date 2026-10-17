using CUDA,KernelAbstractions
include(joinpath(@__DIR__,"..","study_common.jl"))
@kernel function mean_base(data,out)
 tid=Int32(@index(Global,Linear))-Int32(1)
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
@kernel function mean_tiled(data,out)
 tid=Int32(@index(Global,Linear))-Int32(1);local_data=@localmem Float32 (256,)
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
   @synchronize
   for j in Int32(0):Int32(63)
    u=local_data[3j+1];v=local_data[3j+2];w=local_data[3j+3];valid=local_data[193+j];dx=x-u;dy=y-v;dz=z-w;distance=(dx*dx+dy*dy)+dz*dz
    if distance<=60f0*valid
     weight=exp(-distance/32f0);a+=weight*u;b+=weight*v;c+=weight*w;total+=weight*valid
    end
   end
   @synchronize
  end
 if tid<10000;out[row+1]=a/total;out[row+2]=b/total;out[row+3]=c/total;end
 end
end
function factory(a,b,mode)
 backend=KernelAbstractions.get_backend(a);kernel=mode=="base" ? mean_base(backend,64) : mean_tiled(backend,64)
 ()->begin
  x=a;y=b
  for iteration in 1:1000;kernel(x,y;ndrange=10048);CUDA.synchronize();x,y=y,x;end
 end
end
if abspath(PROGRAM_FILE)==@__FILE__;meanshift_study(factory);end

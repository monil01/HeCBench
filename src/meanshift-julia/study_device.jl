@inline function meanshift_body(tid,data,out,local_data,tiled)
 @inbounds begin
 # Actual source D3, N10000, squared radius60 and Gaussian denominator32.
 row=3tid;x=data[row+1];y=data[row+2];z=data[row+3];a=b=c=total=0f0
 if tiled
  lid=tid%64
  for tile in Int32(0):Int32(156)
   other=tile*64+lid
   if other<10000
    local_data[3lid+1]=data[3other+1];local_data[3lid+2]=data[3other+2];local_data[3lid+3]=data[3other+3];local_data[193+lid]=1f0
   else
    local_data[3lid+1]=0f0;local_data[3lid+2]=0f0;local_data[3lid+3]=0f0;local_data[193+lid]=0f0
   end
   sync_mean()
   for j in Int32(0):Int32(63)
    u=local_data[3j+1];v=local_data[3j+2];w=local_data[3j+3];valid=local_data[193+j];dx=x-u;dy=y-v;dz=z-w;distance=(dx*dx+dy*dy)+dz*dz
    if distance<=60f0*valid
     weight=exp(-distance/32f0);a+=weight*u;b+=weight*v;c+=weight*w;total+=weight*valid
    end
   end
   sync_mean()
  end
 else
  for j in Int32(0):Int32(9999)
   u=data[3j+1];v=data[3j+2];w=data[3j+3];dx=x-u;dy=y-v;dz=z-w;distance=(dx*dx+dy*dy)+dz*dz
   if distance<=60f0
    weight=exp(-distance/32f0);a+=weight*u;b+=weight*v;c+=weight*w;total+=weight
   end
  end
 end
 if tid<10000;out[row+1]=a/total;out[row+2]=b/total;out[row+3]=c/total;end
 end;return
end

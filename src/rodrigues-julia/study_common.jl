using CUDA,Printf
const RN=4099
const RG=12
const RGUARD=12345.25f0
@inline function rotate_point!(d,i,angle,wx,wy,wz,::Val{WIDTH}) where WIDTH
 @inbounds begin
  s,c=sincos(angle);mc=1f0-c;j=Int32(12)+(i-Int32(1))*Int32(WIDTH);x=d[j+1];y=d[j+2];z=d[j+3]
  m1=c+wx*wx*mc;m2=wz*s+wx*wy*mc;m3=-wy*s+wx*wz*mc
  m4=-wz*s+wx*wy*mc;m5=c+wy*wy*mc;m6=wx*s+wy*wz*mc
  m7=wy*s+wx*wz*mc;m8=-wx*s+wy*wz*mc;m9=c+wz*wz*mc
  d[j+1]=x*m1+y*m2+z*m3;d[j+2]=x*m4+y*m5+z*m6;d[j+3]=x*m7+y*m8+z*m9
  if WIDTH==4;d[j+4]=0f0;end
 end
 return
end
function rodrigues_study(factory)
 pts=collect(reinterpret(Float32,read(joinpath(@__DIR__,"points_f32.bin"))));oracle=collect(reinterpret(Float64,read(joinpath(@__DIR__,"oracle_f64.bin"))))
 v=Float32[-.3,-.6,.15];norm=1f0/sqrt(v[1]*v[1]+v[2]*v[2]+v[3]*v[3]);axes=[v.*norm,Float32[0,0,1]]
 allok=true;ma=0.0
 for trial in 0:5
  prefix=Float32[];outputs=Float32[];times=Float64[];ok=true;offset=0;trialerr=0.0
  for width in [3,4],axis in axes,angle in Float32[0,.5,Float32(pi/2)],repeats in [1,17]
   W=RN*width+2RG;initial=fill(RGUARD,W)
   for i in 1:RN
    initial[RG+(i-1)*width+1:RG+(i-1)*width+3]=pts[3i-2:3i]
    if width==4;initial[RG+i*width]=Float32((i%17)-8)/8;end
   end
   params=Float32[RGUARD,RN,repeats,width,angle,axis...,RGUARD];d=CuArray(initial);dp=CuArray(params);launch! =factory(d,dp,Val(width))
   if trial==0;launch!(Int32(RN));CUDA.device_synchronize(blocking=true);offset+=3RN;continue;end
   before=Array(d);par=Array(dp);ok &=reinterpret(UInt32,before)==reinterpret(UInt32,initial)&&reinterpret(UInt32,par)==reinterpret(UInt32,params)
   CUDA.device_synchronize(blocking=true);start=time_ns()
   for r in 1:repeats;launch!(Int32(RN));end
   CUDA.device_synchronize(blocking=true);push!(times,(time_ns()-start)*1e-9/repeats)
   out=Array(d);afterparams=Array(dp);ok &=reinterpret(UInt32,afterparams)==reinterpret(UInt32,params)
   ok &=all(==(RGUARD),out[1:RG])&&all(==(RGUARD),out[end-RG+1:end]);budget=5e-6+8e-7*repeats;an=sqrt(sum(Float64(x)^2 for x in axis));unit=Float64.(axis)./an
   for i in 1:RN
    a=Float64.(out[RG+(i-1)*width+1:RG+(i-1)*width+3]);p=Float64.(pts[3i-2:3i]);ref=oracle[offset+3i-2:offset+3i];err=maximum(abs.(a.-ref));trialerr=max(trialerr,err)
    ok &=all(isfinite,a)&&err<=budget&&abs(sqrt(sum(abs2,a))-sqrt(sum(abs2,p)))<=3budget&&abs(sum(a.*unit)-sum(p.*unit))<=3budget
    if width==4;ok &=reinterpret(UInt32,out[RG+i*width])==UInt32(0);end
   end
   append!(prefix,before);append!(prefix,par);append!(outputs,out);offset+=3RN
  end
  trial==0&&continue
  allok &=ok;ma=max(ma,trialerr);artifact=vcat(prefix,outputs);path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"output_trial$(trial).bin");write(path,artifact)
  @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":%.12g,\"max_rel_error\":0,\"timing_region\":\"sum24_separate_inplace_kernel_hostwindows_divided_by_R\",\"jit_excluded\":true,\"path_times_seconds\":[%s],\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float32\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,sum(times),ok ? "true" : "false",length(artifact),trialerr,join(times,","),path,length(artifact))
 end
 allok||error("quaternion/norm/projection/guard/input verification failed");println("PASS")
end

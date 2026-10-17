const EM=(-0.8875f0,-0.6625f0,-0.4375f0,-0.2125f0,-0.0775f0,-0.0325f0,-0.0055f0,0.0f0,0.0055f0,0.0325f0,0.0775f0,0.2125f0,0.4375f0,0.6625f0,0.8875f0,1.0f0,)
const EC=(-0.775f0,-0.55f0,-0.325f0,-0.145f0,-0.055f0,-0.019f0,-0.00275f0,0.00275f0,0.019f0,0.055f0,0.145f0,0.325f0,0.55f0,0.775f0,0.94375f0,)
const VM=(0.0625f0,0.125f0,0.1875f0,0.25f0,0.3125f0,0.375f0,0.4375f0,0.5f0,0.5625f0,0.625f0,0.6875f0,0.75f0,0.8125f0,0.875f0,0.9375f0,1.0f0,)
const VC=(0.09375f0,0.15625f0,0.21875f0,0.28125f0,0.34375f0,0.40625f0,0.46875f0,0.53125f0,0.59375f0,0.65625f0,0.71875f0,0.78125f0,0.84375f0,0.90625f0,0.96875f0,)
const PARAMS=((reinterpret(Float32,UInt32(1036831952)),reinterpret(Float32,UInt32(1023510188)),reinterpret(Float32,UInt32(1008981768)),),(reinterpret(Float32,UInt32(1044549472)),reinterpret(Float32,UInt32(1027023248)),reinterpret(Float32,UInt32(1001158287)),),)
@inline function quant(qmap,mid,x)
 low=Int32(0);high=Int32(15)
 x<=qmap[1]&&return low
 qmap[16]<=x&&return high
 while low<high
  m=(low+high)>>1
  if qmap[m+1]<=x;low=m+1;else;high=m;end
 end
 return mid[low]<x ? low : low-1
end
@inline function update(i,block,p,g,ms,vs,m,v,c2,ss)
 @inbounds begin
  mb=m[i];vb=v[i];ml=EM[Int(mb&0x0f)+1]*ms[block];mr=EM[Int((mb>>4)&0x0f)+1]*ms[block];vl=VM[Int(vb&0x0f)+1]*vs[block];vr=VM[Int((vb>>4)&0x0f)+1]*vs[block]
  gl=g[2i-1];gr=g[2i];ml=.9f0*ml+(1f0-.9f0)*gl;mr=.9f0*mr+(1f0-.9f0)*gr;vl=.999f0*vl+(1f0-.999f0)*(gl*gl);vr=.999f0*vr+(1f0-.999f0)*(gr*gr)
  pl=p[2i-1]*(1f0-1f-3*1f-2)-ss*(ml/(sqrt(vl)/c2+1f-8));pr=p[2i]*(1f0-1f-3*1f-2)-ss*(mr/(sqrt(vr)/c2+1f-8))
 end
 return pl,pr,ml,mr,vl,vr
end
@inline pack_native(l,r)=UInt8((l&15)|(r&240))
function fixture()
 n=4096;p=Float32[Float32(mod(i,31)-15)/16f0 for i in 0:2n-1];g=Float32[Float32(mod(i,17)-8)/64f0 for i in 0:2n-1];ms=Float32[Float32(mod(i,7)+1)/16f0 for i in 0:63];vs=Float32[Float32(mod(i,5)+1)/32f0 for i in 0:63];m=UInt8[UInt8((mod(i,8)+8)|((mod(3i,8)+8)<<4)) for i in 0:n-1];v=UInt8[UInt8(mod(i,16)|(mod(7i,16)<<4)) for i in 0:n-1]
 return p,g,ms,vs,m,v
end
function cpu!(p,g,ms,vs,m,v)
 for (_,c2,ss) in PARAMS,block in 1:64
  updates=[update((block-1)*64+t,block,p,g,ms,vs,m,v,c2,ss) for t in 1:64];em=maximum(max(z[3],z[4]) for z in updates);vm=maximum(max(z[5],z[6]) for z in updates);ms[block]=em;vs[block]=vm
  for t in 1:64
   i=(block-1)*64+t;pl,pr,ml,mr,vl,vr=updates[t];p[2i-1]=pl;p[2i]=pr;m[i]=pack_native(quant(EM,EC,ml/em),quant(EM,EC,mr/em));v[i]=pack_native(quant(VM,VC,vl/vm),quant(VM,VC,vr/vm))
  end
 end
end
function flatten(p,ms,vs,m,v)
 return vcat(Float64.(p),Float64.(m),Float64.(v),Float64.(ms),Float64.(vs))
end
function study(run!)
 initial=fixture();p,g,ms,vs,m,v=deepcopy(initial);cpu!(p,g,ms,vs,m,v);expected=flatten(p,ms,vs,m,v);dp,dg,dms,dvs,dm,dv=map(CuArray,initial);template=CUDA.zeros(Float32,128);args=(dp,dg,dms,dvs,dm,dv,template)
 run!(args...);CUDA.synchronize()
 for trial in 1:5
  for (dest,src) in zip((dp,dg,dms,dvs,dm,dv),initial);copyto!(dest,src);end;CUDA.synchronize();t=time_ns();run!(args...);CUDA.synchronize();seconds=(time_ns()-t)*1e-9/2
  got=flatten(Array(dp),Array(dms),Array(dvs),Array(dm),Array(dv));err=maximum(abs.(got.-expected));ok=all(isfinite,got)&&err<=1e-3;path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"trial_$(trial).bin");open(io->write(io,got),path,"w")
  println("STUDY_TRIAL {\"trial\":",trial,",\"time_seconds\":",seconds,",\"correct\":",ok,",\"checked_values\":",length(got),",\"max_abs_error\":",err,",\"timing_region\":\"mean_two_sequential_optimizer_steps_excluding_reset\",\"jit_excluded\":true,\"internal_repeats\":2,\"output_artifact\":{\"path\":\"",path,"\",\"dtype\":\"float64\",\"count\":",length(got),",\"byte_order\":\"little\"}}")
  ok||error("Native-behavior optimizer CPU companion failed")
 end
end

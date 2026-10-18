using CUDA,Printf
struct FastDiv
 d::Int32
 M::Int32
 s::Int32
 add::Int32
end
function FastDiv(d::Int32)
 if d==1;return FastDiv(d,0,-1,1);elseif d == -1;return FastDiv(d,0,-1,-1);end
 ad=UInt32(abs(d));two31=UInt32(0x80000000);t=two31+(reinterpret(UInt32,d)>>31);anc=t-UInt32(1)-rem(t,ad);p=31;q1=div(two31,anc);r1=two31-q1*anc;q2=div(two31,ad);r2=two31-q2*ad
 while true
  p+=1;q1+=q1;r1+=r1
  if r1>=anc;q1+=UInt32(1);r1-=anc;end
  q2+=q2;r2+=r2
  if r2>=ad;q2+=UInt32(1);r2-=ad;end
  delta=ad-r2
  if !(q1<delta||(q1==delta&&r1==0));break;end
 end
 M=reinterpret(Int32,q2+UInt32(1));if d<0;M=-M;end
 add=d>0&&M<0 ? Int32(1) : d<0&&M>0 ? Int32(-1) : Int32(0)
 FastDiv(d,M,Int32(p-32),add)
end
@inline plain_d(d::Int32)=d
@inline plain_d(d::FastDiv)=d.d
@inline quotient_value(n::Int32,d::Int32)=div(n,d)
@inline function quotient_value(n::Int32,d::FastDiv)
 q=Int32((Int64(d.M)*Int64(n))>>32)+n*d.add
 if d.s>=0
  q=(q>>d.s)+Int32(reinterpret(UInt32,q)>>31)
 end
 q
end
function idivide_study(factory)
 fixture=read(joinpath(@__DIR__,"input_i32.bin"));host=collect(reinterpret(Int32,fixture));oracle=collect(reinterpret(Int32,read(joinpath(@__DIR__,"oracle_i32.bin"))));n=length(host);g=Int32(-123456789);ds=Int32[s*d for d in [1,2,3,5,7,11,16,31,127,255,1024,65535,65537,2147483647] for s in [1,-1]];fast=FastDiv.(ds);stride=2*n+3;x=CuArray(vcat(g,host,g));out=CUDA.fill(g,56*stride);checks=CUDA.fill(g,28*6);launch!,check! =factory(x,out,checks,Int32(n));R=5
 function suite!()
  for r in 1:R,j in 1:28
   launch!(ds[j],Int32((2*j-2)*stride));launch!(fast[j],Int32((2*j-1)*stride))
  end
  CUDA.device_synchronize(blocking=true)
 end
 suite!();allok=true
 for t in 1:5
  fill!(out,g);CUDA.device_synchronize(blocking=true);start=time_ns();suite!();secs=(time_ns()-start)*1e-9/R;got=Array(out);back=Array(x);ok=reinterpret(UInt32,back)==reinterpret(UInt32,vcat(g,host,g))
  # Same functional diagnostic on coordinates +/-0..4095, outside timed ROI.
  checkhost=fill(g,28*6)
  for j in 1:28;checkhost[(j-1)*6+2:(j-1)*6+5].=0;end
  copyto!(checks,checkhost)
  for j in 1:28;check!(fast[j],Int32((j-1)*6));end
  CUDA.device_synchronize(blocking=true);checkback=Array(checks);ok &=checkback==checkhost;bad=0
  for j in 1:28,m in 0:1
   base=(2*j-2+m)*stride;bad+=got[base+1]!=g||got[base+stride]!=g||got[base+2]!=ds[j]
   for k in 1:2*n;bad+=got[base+2+k]!=oracle[(j-1)*2*n+k];end
  end
  ok &=bad==0;allok &=ok;artifact=vcat(back,got,checkback);path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"output_trial$(t).bin");write(path,artifact)
  @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":%d,\"max_rel_error\":0,\"timing_region\":\"observable_56_quotient_remainder_maps_per_repeat\",\"jit_excluded\":true,\"internal_repeats\":%d,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"int32\",\"count\":%d,\"byte_order\":\"little\"}}\n",t,secs,ok ? "true" : "false",length(artifact),bad==0 ? 0 : 1,R,path,length(artifact))
 end
 allok||error("quotient/remainder/input/guard/functional check failed");println("PASS")
end

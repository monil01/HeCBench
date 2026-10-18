using Printf
include(joinpath(@__DIR__,"study_cases.jl"))
const G=16
const GUARD=UInt32(0x4a5a5a5a)
const POISON=reinterpret(UInt32,-1234567.25f0)
function study(allocate,launch)
 fixture=collect(reinterpret(UInt32,read(joinpath(@__DIR__,"input_u32.bin"))))
 oracle=collect(reinterpret(Float64,read(joinpath(@__DIR__,"oracle_f64.bin"))))
 states=[]
 for c in CASES
  types=[BYield,BYield,BDate,BDate,Float32,BBond,Float32,Float32,Float32,Float32,Float32]
  widths=[16,16,4,4,1,9,1,1,1,1,1];host=Vector{UInt32}[];off=c.io
  for k in 1:11
   h=fill(GUARD,c.n*widths[k]+32)
   if k<=7;copyto!(h,G+1,fixture,off+1,c.n*widths[k]);off+=c.n*widths[k]
   else;fill!(view(h,G+1:length(h)-G),POISON)
   end
   push!(host,h)
  end
  raw=[allocate(h) for h in host]
  dev=[reinterpret(T,view(a,G+1:length(a)-G)) for (T,a) in zip(types,raw)]
  push!(states,(c=c,host=host,raw=raw,dev=dev))
 end
 function reset(s)
  for k in [1,8,9,10,11];copyto!(s.raw[k],s.host[k]);end
  CUDA.device_synchronize()
 end
 function capture_pre!(out,s)
  for k in 1:11
   v=Array(s.raw[k]);append!(out,k<=7 ? Float64.(v) : Float64.(reinterpret(Float32,v)))
  end
 end
 function check_capture!(out,s)
  c=s.c;ok=true;err=0.;rel=0.
  for k in 1:11
   v=Array(s.raw[k]);payload=k<=7 ? Float64.(v) : Float64.(reinterpret(Float32,v))
   ok&=all(==(GUARD),view(v,1:G))&&all(==(GUARD),view(v,length(v)-G+1:length(v)))
   if k<=7
    for i in G+1:length(v)-G
     rate=k==1&&(i-G-1)%16==0
     if !rate;ok&=v[i]==s.host[k][i]
     else;payload[i]=Float64(reinterpret(Float32,v[i]))
     end
    end
   end
   if k>=8 || k==1
    column=k==1 ? 4 : k-8
    want=view(oracle,c.oo+column*c.n+1:c.oo+(column+1)*c.n)
    for j in 1:c.n
     i=k==1 ? G+16*(j-1)+1 : G+j
     actual=Float64(reinterpret(Float32,v[i]));e=abs(actual-want[j]);err=max(err,e);rel=max(rel,e/max(1e-300,abs(want[j])));ok&=isfinite(actual)&&e<=.002+.0001*abs(want[j])
     if k>=8&&k<=10;ok&=e<=.002;end
     if k==1
      coupon=50*Float64(reinterpret(Float32,s.host[6][G+9*(j-1)+9]))
      npv=(100+coupon)*(1+actual/2)^(-1/180)
      ok&=abs(npv-oracle[c.oo+j])<=.002
     end
    end
   end
   append!(out,payload)
  end
  ok,err,rel
 end
 for s in states
  launch(s.dev,s.c.n);CUDA.device_synchronize();out=Float64[];ok,e,r=check_capture!(out,s);println("WARM n=",s.c.n," pass=",ok," maxabs=",e," maxrel=",r);ok||error("Independent datedcashflow/NPV/price oracle")
 end
 for trial in 1:5
  pre=Float64[];post=Float64[];seconds=0.;ok=true;maxabs=0.;maxrel=0.;times=Float64[]
  for s in states
   reset(s);capture_pre!(pre,s);start=time_ns();launch(s.dev,s.c.n);CUDA.device_synchronize();sec=(time_ns()-start)*1e-9;seconds+=sec;push!(times,sec)
   passed,e,r=check_capture!(post,s);ok&=passed;maxabs=max(maxabs,e);maxrel=max(maxrel,r)
  end
  append!(pre,post);path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"bonds_trial$(trial).bin");open(io->write(io,pre),path,"w")
  @printf("STUDY_TRIAL {\"trial\":%d,\"correct\":%s,\"jit_excluded\":true,\"time_seconds\":%.12g,\"timing_region\":\"sum_four_original_dated_bond_kernel_and_device_completion\",\"max_abs_error\":%.17g,\"max_rel_error\":%.17g,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":%d,\"shape\":[%d],\"byte_order\":\"little\"},\"case_times_seconds\":[%s]}\n",trial,ok ? "true" : "false",seconds,maxabs,maxrel,path,length(pre),length(pre),join(times,','))
  ok||error("Full typedinputs/metadata/guards/oracle")
 end
end

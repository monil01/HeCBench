using Printf
include(joinpath(@__DIR__,"study_cases.jl"))
const G=16
const GUARD=-987654321.25
const POISON=-1234567.25
function study(allocate,launch)
 f=collect(reinterpret(Float64,read(joinpath(@__DIR__,"input_f64.bin"))))
 oracle=collect(reinterpret(Float64,read(joinpath(@__DIR__,"oracle_paths_f64.bin"))))
 states=[]
 for c in CASES
  sizes=[c.n*c.steps,c.n*c.steps,16*c.steps,8192];host=Vector{Float64}[]
  for k in 1:4
   h=fill(GUARD,sizes[k]+2G);fill!(view(h,G+1:length(h)-G),POISON)
   if k==1;copyto!(h,G+1,f,c.io+1,sizes[1]);end
   push!(host,h)
  end
  raw=[allocate(h) for h in host];dev=[view(a,G+1:length(a)-G) for a in raw]
  hf=fill(Int32(-55555),c.steps+2G);rf=allocate(hf);flags=view(rf,G+1:length(rf)-G)
  push!(states,(c=c,host=host,raw=raw,dev=dev,hf=hf,rf=rf,flags=flags))
 end
 function capture!(payload,s)
  for a in s.raw;append!(payload,Array(a));end
  append!(payload,Float64.(Array(s.rf)))
 end
 function run(s)
  c=s.c;n=Int32(c.n);steps=Int32(c.steps);d=s.dev
  copyto!(d[1],view(s.host[1],G+1:length(s.host[1])-G))
  launch(1,d[1],d[2],n,steps,c.K,c.S0,c.r,c.sigma,c.dt,c.put;threads=256,blocks=cld(c.n,256))
  # Source cudaMemsetAsync(flags,0) occurs after paths launch on the same stream.
  GC.@preserve s CUDA.memset(convert(CUDA.CuPtr{UInt32},pointer(s.flags)),UInt32(0),length(s.flags))
  launch(2,d[2],s.flags,d[3],n,c.K,c.put;threads=256,blocks=c.steps-1)
  discount=exp(-c.r*c.dt);grid=cld(c.n,128);waves=grid*128/Float64(256*112)
  updategrid=(waves<10 && waves-trunc(Int,waves)<.6) ? max(1,trunc(Int,waves))*224 : grid
  for step in c.steps-2:-1:0
   launch(3,d[2],s.flags,d[3],d[4],n,Int32(step),c.K,c.put;threads=128,blocks=128)
   launch(4,s.flags,d[4],Int32(step);threads=128,blocks=1)
   launch(5,d[2],s.flags,d[4],n,Int32(step),c.K,c.put,discount;threads=128,blocks=updategrid)
  end
  launch(6,d[2],d[4],n;threads=128,blocks=grid)
  launch(7,d[4],n,Int32(grid),discount;threads=128,blocks=1)
  price=Array(view(d[4],1:1))[1]
  price
 end
 function reset(s)
  for k in 2:4;copyto!(s.raw[k],s.host[k]);end
  copyto!(s.rf,s.hf);CUDA.device_synchronize()
 end
 function check(s)
  ok=true;err=0.
  for k in 1:4
   v=Array(s.raw[k]);ok&=all(==(GUARD),view(v,1:G))&&all(==(GUARD),view(v,length(v)-G+1:length(v)))
   if k==1;ok&=v==s.host[k]
   elseif k==2
    want=view(oracle,s.c.oo+1:s.c.oo+s.c.n*s.c.steps);actual=view(v,G+1:length(v)-G)
    err=max(err,maximum(abs.(actual.-want)));ok&=all(abs.(actual.-want).<=2e-6.+1e-5.*abs.(want))
   else;ok&=all(isfinite,v)
   end
  end
  v=Array(s.rf);ok&=all(==(-55555),view(v,1:G))&&all(==(-55555),view(v,length(v)-G+1:length(v)))&&all(x->x in (0,1),view(v,G+1:length(v)-G))
  ok,err
 end
 for s in states
  price=run(s);ok,err=check(s);println("WARM n=",s.c.n," steps=",s.c.steps," price=",price," maxoracleerr=",err," pass=",ok);ok||error("Independent path/cashflow oracle")
 end
 for trial in 1:5
  pre=Float64[];post=Float64[];seconds=0.;ok=true;err=0.;times=Float64[];prices=Float64[]
  for s in states
   reset(s);capture!(pre,s);start=time_ns();price=run(s);sec=(time_ns()-start)*1e-9;push!(times,sec);push!(prices,price);seconds+=sec
   passed,e=check(s);ok&=passed;err=max(err,e);capture!(post,s)
  end
  append!(pre,post);path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"aop_trial$(trial).bin");open(io->write(io,pre),path,"w")
  @printf("STUDY_TRIAL {\"trial\":%d,\"correct\":%s,\"jit_excluded\":true,\"time_seconds\":%.12g,\"timing_region\":\"sum_seven_source_do_run_sample_H2D_through_scalar_D2H\",\"max_abs_error\":%.17g,\"max_rel_error\":null,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":%d,\"shape\":[%d],\"byte_order\":\"little\"},\"case_times_seconds\":[%s],\"prices\":[%s]}\n",trial,ok ? "true" : "false",seconds,err,path,length(pre),length(pre),join(times,','),join(prices,','))
  ok||error("Complete finalstate oracle")
 end
end

using CUDA,Libdl
const RL_ROOT=@__DIR__
function rl_study(alloc,launch)
 cases=open(joinpath(RL_ROOT,"fixtures_compact.bin"),"r")do io
  read(io,UInt32)==3||error("casecount")
  cs=[]
  for ci in 1:3
   n=read(io,UInt32);steps=read(io,UInt32);np=read(io,UInt32);t=read(io,Float64);dt=read(io,Float64)
   patterns=read!(io,Vector{Float64}(undef,Int(np)*72));pat=reshape(patterns,72,Int(np))
   s=Float64[pat[f,mod(i-1,Int(np))+1]for f in 1:19 for i in 1:Int(n)]
   p=Float64[pat[f+19,mod(i-1,Int(np))+1]for f in 1:53 for i in 1:Int(n)]
   rs=alloc(fill(-777.,length(s)+128));rp=alloc(fill(-777.,length(p)+128));ds=view(rs,65:64+length(s));dp=view(rp,65:64+length(p))
   push!(cs,(n=Int32(n),steps=Int(steps),np=np,t=t,dt=dt,s=s,p=p,rs=rs,rp=rp,ds=ds,dp=dp))
  end
  eof(io)||error("fixturetrailing");cs
 end
 check=Libdl.dlsym(Libdl.dlopen(joinpath(pwd(),"studycheck.so")),:study_check)
 function reset(c)
  a=fill(-777.,length(c.s)+128);a[65:64+length(c.s)]=c.s;copyto!(c.rs,a)
  b=fill(-777.,length(c.p)+128);b[65:64+length(c.p)]=c.p;copyto!(c.rp,b)
 end
 for c in cases;reset(c);launch(c,c.t);CUDA.device_synchronize();end
 for trial in 1:5
  seconds=0.;path="trial$(trial).bin"
  open(path,"w")do io
   write(io,3.)
   for c in cases
    reset(c);CUDA.device_synchronize();write(io,Float64[c.n,c.steps,c.np,c.t,c.dt]);write(io,Array(c.ds));write(io,Array(c.dp))
   end
   for c in cases
    t=c.t;CUDA.device_synchronize();start=time_ns()
    for step in 1:c.steps;launch(c,t);t+=c.dt;end
    CUDA.device_synchronize();seconds+=(time_ns()-start)*1e-9
    write(io,Array(c.ds));reinterpret(UInt64,Array(c.dp))==reinterpret(UInt64,c.p)||error("readonlyparameters")
    copyto!(c.ds,c.s);launch(c,c.t);CUDA.device_synchronize();write(io,Array(c.ds))
    for raw in (c.rs,c.rp);g=Array(raw);write(io,g[1:64]);write(io,g[end-63:end]);end
   end
  end
  metrics=zeros(Float64,3);ccall(check,Cint,(Cstring,Ptr{Float64}),path,metrics)==0||error("independent/fullstatecheck")
  println("STUDY_TRIAL {\"trial\":",trial,",\"time_seconds\":",seconds,",\"correct\":true,\"checked_values\":11032684,\"max_abs_error\":",metrics[1],",\"max_rel_error\":",metrics[2],",\"independent_gate_max_abs\":",metrics[3],",\"timing_region\":\"sum3_full_source_RushLarsen_timestepping_context_completion\",\"jit_excluded\":true,\"internal_repeats\":201,\"readonly_parameters_checked_count\":5315370,\"first_step_evidence\":\"untimed_replay_after_capturing_timed_final\",\"output_artifact\":{\"path\":\"",path,"\",\"dtype\":\"float64\",\"count\":11032684,\"byte_order\":\"little\"}}")
 end
end

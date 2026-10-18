const CASES=((16,1,0),(16,3,0),(24,3,1),(16,3,2))
function study(allocate,dispatch)
 fixture=reinterpret(Float64,read(joinpath(@__DIR__,"fixtures_f64.bin")))|>collect;oracle=reinterpret(Float64,read(joinpath(@__DIR__,"oracle_f64.bin")))|>collect
 states=[];off=0
 for (n,steps,mode) in CASES
  v=n^3;raw=[allocate(Float64,v+2) for k in 1:7];a=[view(t,2:v+1) for t in raw];push!(states,(n,steps,mode,raw,a,copy(fixture[off+1:off+v]),copy(fixture[off+v+1:off+2v])));off+=2v
 end
 function run(c)
  n,steps,mode,raw,a,phi,u=c;v=n^3
  for k in 1:7
   h=fill(-8192.25,v+2);h[1]=h[end]=-1048576.5;k==1&&(h[2:end-1]=phi);k==3&&(h[2:end-1]=u);copyto!(raw[k],h)
  end
  mod=n==16 ? Ace16 : Ace24;h=.4;ep=.07;w=1.;t=1.25/sqrt(2.)*.64;la=1.25/sqrt(2.)/.5
  CUDA.synchronize(CUDA.context());start=time_ns()
  for st in 1:steps
   dispatch(mod,n,1,a[1],a[5],a[6],a[7],h,h,h,ep,w,t)
   dispatch(mod,n,2,a[2],a[1],a[3],a[5],a[6],a[7],ep,w,t,la,.01,h,h,h)
   dispatch(mod,n,3,a[2]);dispatch(mod,n,4,a[4],a[3],a[2],a[1],2.,.01,h,h,h);dispatch(mod,n,5,a[4],.8);dispatch(mod,n,6,a[2],a[1]);dispatch(mod,n,6,a[4],a[3])
  end
  CUDA.synchronize(CUDA.context());elapsed=(time_ns()-start)*1e-9
  arrays=Array.(raw);ok=all(t->t[1]==-1048576.5&&t[end]==-1048576.5,arrays)
  return vcat([t[2:end-1] for t in arrays]...),elapsed,ok
 end
 for c in states;run(c);end
 for trial in 1:5
  out=copy(fixture);vals=Float64[];elapsed=0.;ok=true
  for c in states;v,t,g=run(c);append!(vals,v);elapsed+=t;ok&=g;end
  err=maximum(abs.(vals.-oracle));ok&=all(isfinite,vals)&&all(abs.(vals.-oracle).<=1e-10 .+1e-10.*abs.(oracle));append!(out,vals)
  path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"trial_$(trial).bin");open(io->write(io,out),path,"w")
  println("STUDY_TRIAL {\"trial\":",trial,",\"time_seconds\":",elapsed,",\"correct\":",ok,",\"checked_values\":",length(out),",\"max_abs_error\":",err,",\"timing_region\":\"sum4_preallocated_source_sevenlaunch_steps_complete\",\"jit_excluded\":true,\"internal_repeats\":1,\"output_artifact\":{\"path\":\"",path,"\",\"dtype\":\"float64\",\"count\":",length(out),",\"byte_order\":\"little\"}}")
  ok||error("AceFullFieldOracleOrGuard")
 end
end

const CASES=((33,1,0),(257,1,1),(129,3,2));const SENT=-8192.25f0;const NSENT=Int8(-99)
function study(allocate,launch)
 fixture=collect(reinterpret(Float32,read(joinpath(@__DIR__,"fixtures_f32.bin"))));oracle=collect(reinterpret(Float32,read(joinpath(@__DIR__,"oracle_f32.bin"))));states=[];off=0
 for(n,steps,mode)in CASES
  dt=fixture[off+4];clock=fixture[off+5];P=Tuple(fixture[off+6:off+38]);inputs=[copy(fixture[off+38+(k-1)*n+1:off+38+k*n]) for k in 1:7];raw=[allocate(Float32,n+2) for k in 1:7];nr=allocate(Int8,n+2);args=[view(a,2:n+1) for a in raw];push!(states,(n,steps,mode,dt,clock,P,inputs,raw,nr,args));off+=38+7n
 end
 function run(c)
  n,steps,mode,dt,clock,P,inputs,raw,nr,args=c
  for k in 1:7;h=vcat(SENT,inputs[k],SENT);copyto!(raw[k],h);end;fill!(nr,NSENT)
  CUDA.synchronize(CUDA.context());start=time_ns()
  for step in 1:steps;launch(args...,view(nr,2:n+1),Int32(n),dt,clock,P);end
  CUDA.synchronize(CUDA.context());elapsed=(time_ns()-start)*1e-9/steps;arrays=Array.(raw);flags=Array(nr);guards=Float32[];ok=true
  for a in arrays;append!(guards,Float32[a[1],a[end]]);ok&=a[1]===SENT&&a[end]===SENT;end
  append!(guards,Float32[flags[1],flags[end]]);ok&=flags[1]===NSENT&&flags[end]===NSENT&&reinterpret(UInt8,arrays[7][2:end-1])==reinterpret(UInt8,inputs[7])
  vals=vcat([a[2:end-1] for a in arrays]...,Float32.(flags[2:end-1]));return vals,guards,elapsed,ok
 end
 for c in states;run(c);end
 for trial in 1:5
  out=copy(fixture);oi=0;elapsed=0.;ok=true;err=0.
  for c in states
   vals,guards,t,pass=run(c);ref=oracle[oi+1:oi+length(vals)];e=maximum(abs.(vals.-ref));ok&=pass&&all(isfinite,vals)&&all(abs.(vals.-ref).<=2f-4 .+2f-4.*abs.(ref));n=c[1];ok&=vals[7n+1:8n]==ref[7n+1:8n];oi+=length(vals);err=max(err,Float64(e));append!(out,vals);append!(out,guards);elapsed+=t
  end
  path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"trial_$(trial).bin");open(io->write(io,out),path,"w");println("STUDY_TRIAL {\"trial\":",trial,",\"time_seconds\":",elapsed,",\"correct\":",ok,",\"checked_values\":",length(out),",\"max_abs_error\":",err,",\"timing_region\":\"sum3_source_frozenclock_iteration_means_complete\",\"jit_excluded\":true,\"internal_repeats\":1,\"case_internal_repeats\":[1,1,3],\"output_artifact\":{\"path\":\"",path,"\",\"dtype\":\"float32\",\"count\":",length(out),",\"byte_order\":\"little\"}}")
  ok||error("CobahhActualSourceStateFlagsAndGuards")
 end
end

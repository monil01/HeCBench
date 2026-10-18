const NAMES=(16,129);const REPEATS=1000;const SENT=Float32(-1048576.5)
function study(allocate,launch)
 fixture=collect(reinterpret(Float32,read(joinpath(@__DIR__,"fixtures_f32.bin"))));expected=collect(reinterpret(Float64,read(joinpath(@__DIR__,"oracle_f64.bin"))));total=sum(NAMES);states=[];off=0
 for n in NAMES
  params=reshape(copy(fixture[5total+5off+1:5total+5off+5n]),5,n);raw=[allocate(Float32,n+2)for k in 1:6];args=[view(a,2:n+1)for a in raw]
  for k in 1:5;h=fill(SENT,n+2);h[2:end-1]=params[k,:];copyto!(raw[k],h);end
  fill!(raw[6],SENT);push!(states,(n,raw,args));off+=n
 end
 for(n,raw,args)in states;launch(n,args...);end;CUDA.synchronize(CUDA.context())
 for trial in 1:5
  out=copy(fixture);elapsed=0.;ok=true;err=0.;off=0
  for(n,raw,args)in states
   fill!(raw[6],SENT);CUDA.synchronize(CUDA.context());start=time_ns()
   for rep in 1:REPEATS;launch(n,args...);end
   CUDA.synchronize(CUDA.context());elapsed+=(time_ns()-start)*1e-9/REPEATS
   arrays=Array.(raw);ok&=all(a->a[1]===SENT&&a[end]===SENT,arrays);vals=arrays[6][2:end-1];reference=expected[off+1:off+n];err=max(err,maximum(abs.(Float64.(vals).-reference)));ok&=all(isfinite,vals)&&all(abs.(Float64.(vals).-reference).<=2e-4 .+5e-4.*abs.(reference));append!(out,vals);append!(out,Float32[arrays[6][1],arrays[6][end]]);off+=n
  end
  path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"trial_$(trial).bin");open(io->write(io,out),path,"w");println("STUDY_TRIAL {\"trial\":",trial,",\"time_seconds\":",elapsed,",\"correct\":",ok,",\"checked_values\":",length(out),",\"max_abs_error\":",err,",\"timing_region\":\"sum2_preallocated_1000kernel_launch_means_complete\",\"jit_excluded\":true,\"internal_repeats\":1000,\"output_artifact\":{\"path\":\"",path,"\",\"dtype\":\"float32\",\"count\":",length(out),",\"byte_order\":\"little\"}}")
  ok||error("BinomialDiscreteExpectationOrGuard")
 end
end

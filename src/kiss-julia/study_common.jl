const SENT=UInt64(0xdeadbeef12345678)
function study(allocate,launch)
 f=collect(reinterpret(UInt64,read(joinpath(@__DIR__,"fixtures_u64.bin"))));oracle=collect(reinterpret(UInt64,read(joinpath(@__DIR__,"oracle_u64.bin"))));states=[]
 for ci in 0:2
  n=Int(f[3ci+1]);seed=UInt32(f[3ci+2]);reps=Int(f[3ci+3]);raw=allocate(UInt64,n+8);fill!(raw,SENT);push!(states,(n,seed,reps,raw,view(raw,5:n+4)))
 end
 for (n,seed,reps,raw,out) in states;launch(out,UInt64(n),seed);CUDA.synchronize(CUDA.context());end
 for trial in 1:5
  payload=copy(f);ok=true;elapsed=0.;times=Float64[];oi=1
  for (n,seed,reps,raw,out) in states
   fill!(raw,SENT);CUDA.synchronize(CUDA.context());start=time_ns()
   for repeat in 1:reps;launch(out,UInt64(n),seed);end
   CUDA.synchronize(CUDA.context());tm=(time_ns()-start)*1e-9/reps;elapsed+=tm;push!(times,tm);got=Array(raw);ok&=all(x==SENT for x in got[1:4])&&all(x==SENT for x in got[end-3:end])&&got[5:end-4]==oracle[oi:oi+n-1];oi+=n;append!(payload,got)
  end
  path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"trial_$(trial).bin");open(io->write(io,payload),path,"w");println("STUDY_TRIAL {\"trial\":",trial,",\"time_seconds\":",elapsed,",\"correct\":",ok,",\"checked_values\":",length(payload),",\"max_abs_error\":",ok ? 0 : "null",",\"timing_region\":\"sum3_KISS64_3launch_means_complete\",\"jit_excluded\":true,\"internal_repeats\":3,\"timed_kernel_calls\":9,\"case_times_seconds\":",times,",\"output_artifact\":{\"path\":\"",path,"\",\"dtype\":\"uint64\",\"count\":",length(payload),",\"byte_order\":\"little\"}}")
  ok||error("AllKISS64StreamsIndependentModularOracleCanaries")
 end
end

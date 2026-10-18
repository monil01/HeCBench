const SENT=UInt32(0xdeadbeef)
function study(allocate,launch,probe)
 f=collect(reinterpret(UInt32,read(joinpath(@__DIR__,"fixtures_u32.bin"))));oracle=collect(reinterpret(UInt32,read(joinpath(@__DIR__,"oracle_u32.bin"))));po=collect(reinterpret(UInt32,read(joinpath(@__DIR__,"probe_oracle_u32.bin"))));states=[];pos=1
 for case in 1:3
  n,reps=Int.(f[pos:pos+1]);pos+=2;h=vcat(fill(SENT,4),f[pos:pos+n-1],fill(SENT,4));pos+=n;raw=allocate(UInt32,n+8);out=allocate(UInt32,n+8);copyto!(raw,h);fill!(out,SENT);push!(states,(n,reps,h,raw,out,view(raw,5:n+4),view(out,5:n+4)))
 end
 ops,p=Int.(f[pos:pos+1]);pos+=2;nt=ops*p;h=vcat(fill(SENT,4),f[pos:end],fill(SENT,4));@assert length(h)==3nt+8;raw=allocate(UInt32,length(h));copyto!(raw,h);out=allocate(UInt32,nt+8);fill!(out,SENT);pv=view(raw,5:3nt+4);ov=view(out,5:nt+4)
 for (n,reps,host,rin,rout,iv,vv) in states;launch(iv,vv,n);CUDA.synchronize(CUDA.context());end
 probe(pv,ov,nt,p);CUDA.synchronize(CUDA.context())
 for trial in 1:5
  payload=copy(f);correct=true;elapsed=0.;times=Float64[];oi=1
  for (n,reps,host,rin,rout,iv,vv) in states
   fill!(rout,SENT);CUDA.synchronize(CUDA.context());start=time_ns()
   for repeat in 1:reps;launch(iv,vv,n);end
   CUDA.synchronize(CUDA.context());tm=(time_ns()-start)*1e-9/reps;elapsed+=tm;push!(times,tm);ib=Array(rin);ob=Array(rout);correct&=ib==host&&all(x==SENT for x in ob[1:4])&&all(x==SENT for x in ob[end-3:end])&&ob[5:end-4]==oracle[oi:oi+n-1];oi+=n;append!(payload,ib);append!(payload,ob)
  end
  # Auxiliary native intrinsic / portable operation table, outside all timed regions.
  fill!(out,SENT);probe(pv,ov,nt,p);CUDA.synchronize(CUDA.context());ib=Array(raw);ob=Array(out);correct&=ib==h&&all(x==SENT for x in ob[1:4])&&all(x==SENT for x in ob[end-3:end])&&ob[5:end-4]==po;append!(payload,ib);append!(payload,ob)
  path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"trial_$(trial).bin");open(io->write(io,payload),path,"w");println("STUDY_TRIAL {\"trial\":",trial,",\"time_seconds\":",elapsed,",\"correct\":",correct,",\"checked_values\":",length(payload),",\"max_abs_error\":",correct ? 0 : "null",",\"timing_region\":\"sum3_simd_100launch_means_complete\",\"jit_excluded\":true,\"internal_repeats\":100,\"timed_kernel_calls\":300,\"untimed_individual_operation_results\":",nt,",\"case_times_seconds\":",times,",\"output_artifact\":{\"path\":\"",path,"\",\"dtype\":\"uint32\",\"count\":",length(payload),",\"byte_order\":\"little\"}}")
  correct||error("ExactAll118SIMDOperationsAndSourceXorReadonlyGuards")
 end
end

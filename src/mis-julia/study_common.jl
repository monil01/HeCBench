const IGUARD=Int32(123456789)
const BGUARD=UInt8(165)
function study(allocate,launch)
 f=collect(reinterpret(Int32,read(joinpath(@__DIR__,"fixtures_i32.bin"))));oracle=collect(reinterpret(Int32,read(joinpath(@__DIR__,"oracle_i32.bin"))));nc=Int(f[1]);pos=2;states=[];oi=1
 function guarded(T,values)
  raw=allocate(T,length(values)+8);fill!(raw,T==Int32 ? IGUARD : BGUARD);copyto!(view(raw,5:length(raw)-4),T.(values));(raw,view(raw,5:length(raw)-4))
 end
 for _ in 1:nc
  n,e,reps=Int.(f[pos:pos+2]);pos+=3;idx=copy(f[pos:pos+n]);pos+=n+1;list=copy(f[pos:pos+e-1]);pos+=e
  ir,id=guarded(Int32,idx);lr,ld=guarded(Int32,list);sr,sd=guarded(UInt8,zeros(UInt8,n));pr,pd=guarded(UInt8,zeros(UInt8,n));expected_priority=copy(oracle[oi:oi+n-1]);oi+=n;expected_final=copy(oracle[oi:oi+n-1]);oi+=n
  push!(states,(n=Int32(n),e=Int32(e),reps=reps,idx=idx,list=list,ir=ir,id=id,lr=lr,ld=ld,sr=sr,sd=sd,pr=pr,pd=pd,expected_priority=expected_priority,expected_final=expected_final))
 end
 for c in states
  launch(1,c.n,c.e,c.id,c.ld,c.sd);launch(2,c.n,c.e,c.id,c.ld,c.sd);CUDA.synchronize(CUDA.context())
 end
 for trial in 1:5
  payload=copy(f);ok=true;elapsed=0.;times=Float64[]
  for c in states
   fill!(c.sd,0);fill!(c.pd,0);CUDA.synchronize(CUDA.context());start=time_ns()
   for _ in 1:c.reps;launch(1,c.n,c.e,c.id,c.ld,c.sd);launch(2,c.n,c.e,c.id,c.ld,c.sd);end
   CUDA.synchronize(CUDA.context());tm=(time_ns()-start)*1e-9/c.reps;elapsed+=tm;push!(times,tm)
   final=Array(c.sr) # Actual timedfinalstatus captured BEFORE deterministicinitializationreplay.
   launch(1,c.n,c.e,c.id,c.ld,c.pd);CUDA.synchronize(CUDA.context());priority=Array(c.pr)
   for (raw,want) in [(c.ir,c.idx),(c.lr,c.list)]
    got=Array(raw);ok&=all(==(IGUARD),got[1:4])&&all(==(IGUARD),got[end-3:end])&&got[5:end-4]==want;append!(payload,got)
   end
   for got in [final,priority];ok&=all(==(BGUARD),got[1:4])&&all(==(BGUARD),got[end-3:end]);append!(payload,Int32.(got));end
   values=final[5:end-4];ok&=Int32.(values)==c.expected_final&&Int32.(priority[5:end-4])==c.expected_priority
   for v in 0:Int(c.n)-1
    if values[v+1]==IN_STATUS
     for j in Int(c.idx[v+1])+1:Int(c.idx[v+2]);ok&=values[Int(c.list[j])+1]!=IN_STATUS;end
    elseif values[v+1]==OUT_STATUS
     ok&=any(values[Int(c.list[j])+1]==IN_STATUS for j in Int(c.idx[v+1])+1:Int(c.idx[v+2]))
    else;ok=false;end
   end
  end
  path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"trial_$(trial).bin");open(io->write(io,payload),path,"w")
  println("STUDY_TRIAL {\"trial\":",trial,",\"time_seconds\":",elapsed,",\"correct\":",ok,",\"checked_values\":",length(payload),",\"max_abs_error\":",ok ? 0 : "null",",\"timing_region\":\"sum7_init_asyncMIS_3repeat_means_complete\",\"jit_excluded\":true,\"internal_repeats\":3,\"timed_kernel_calls\":42,\"priority_capture\":\"deterministic_untimed_init_replay_after_actualtimedfinal_capture\",\"case_times_seconds\":",times,",\"output_artifact\":{\"path\":\"",path,"\",\"dtype\":\"int32\",\"count\":",length(payload),",\"byte_order\":\"little\"}}")
  ok||error("FullMISpriority_membership_CSR_guards_independence_maximality")
 end
end

@inline step_rng(s::UInt64)=(s<<1)⊻(reinterpret(Int64,s)<0 ? UInt64(7) : UInt64(0))
@inline function starts(n::Int64)
 while n<0;n+=Int64(1317624576693539401);end
 while n>Int64(1317624576693539401);n-=Int64(1317624576693539401);end
 n==0&&return UInt64(1)
 m2=MVector{64,UInt64}(undef);temp=UInt64(1)
 for i in 1:64
  @inbounds m2[i]=temp
  temp=step_rng(step_rng(temp))
 end
 i=62
 while i>=0 && ((n>>i)&1)==0;i-=1;end
 ran=UInt64(2)
 while i>0
  temp=UInt64(0)
  for j in 0:63
   if ((ran>>j)&UInt64(1))!=0;@inbounds temp⊻=m2[j+1];end
  end
  ran=temp;i-=1
  if ((n>>i)&1)!=0;ran=step_rng(ran);end
 end
 return ran
end
function study(run!)
 n=16384;expected=UInt64.(0:n-1);rng=UInt64(1)
 for i in 1:4n
  rng=step_rng(rng);expected[Int(rng&UInt64(n-1))+1]⊻=rng
 end
 table=CUDA.zeros(UInt64,n);run!(table);CUDA.synchronize()
 for trial in 1:5
  CUDA.synchronize();t=time_ns();for repeat in 1:10;run!(table);end;CUDA.synchronize();seconds=(time_ns()-t)*1e-9/10
  got=Array(table);ok=got==expected;path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"trial_$(trial).bin");open(io->write(io,got),path,"w")
  println("STUDY_TRIAL {\"trial\":",trial,",\"time_seconds\":",seconds,",\"correct\":",ok,",\"checked_values\":",n,",\"max_abs_error\":",ok ? 0 : 1,",\"timing_region\":\"mean_init_plus_random_atomic_updates\",\"jit_excluded\":true,\"internal_repeats\":10,\"output_artifact\":{\"path\":\"",path,"\",\"dtype\":\"uint64\",\"count\":",n,",\"byte_order\":\"little\"}}")
  ok||error("Exact HPCC sequential-stream oracle failed")
 end
end

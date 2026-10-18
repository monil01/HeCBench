const GUARD=UInt64(0xdeadbeef12345678)
function study(allocate,launch)
 f=collect(reinterpret(UInt64,read(joinpath(@__DIR__,"fixtures_u64.bin"))));oracle=collect(reinterpret(UInt64,read(joinpath(@__DIR__,"oracle_u64.bin"))));nc=Int(f[1]);pos=2
 function guarded(values)
  raw=allocate(UInt64,length(values)+8);fill!(raw,GUARD);copyto!(view(raw,5:length(raw)-4),values);(raw,view(raw,5:length(raw)-4))
 end
 mr,mds=guarded(f[pos:pos+143]);pos+=144;ar,a1=guarded(f[pos:pos+83]);pos+=84;br,a2=guarded(f[pos:pos+83]);pos+=84;states=[]
 for _ in 1:nc
  n=Int(f[pos]);pos+=1;lv=copy(f[pos:pos+4n-1]);pos+=4n;lr,leaves=guarded(lv);orr,out=guarded(zeros(UInt64,4n));push!(states,(n,lv,lr,leaves,orr,out))
 end
 for(n,lv,lr,leaves,orr,out)in states;launch(leaves,out,n,mds,a1,a2);end
 for trial in 1:5
  payload=copy(f);ok=true;total=0.;times=Float64[];oi=1
  for (raw,want)in[(mr,f[2:145]),(ar,f[146:229]),(br,f[230:313])]
   got=Array(raw);ok&=all(==(GUARD),got[1:4])&&all(==(GUARD),got[end-3:end])&&got[5:end-4]==want;append!(payload,got)
  end
  for(n,lv,lr,leaves,orr,out)in states
   fill!(out,0);CUDA.synchronize(CUDA.context());start=time_ns();launch(leaves,out,n,mds,a1,a2);elapsed=(time_ns()-start)*1e-9;total+=elapsed;push!(times,elapsed)
   l=Array(lr);o=Array(orr);ok&=all(==(GUARD),l[1:4])&&all(==(GUARD),l[end-3:end])&&l[5:end-4]==lv;ok&=all(==(GUARD),o[1:4])&&all(==(GUARD),o[end-3:end]);back=o[5:end-4];ok&=all((back[j]%PRIME)==oracle[oi+j-1] for j in eachindex(back));oi+=4n;append!(payload,l);append!(payload,o)
  end
  path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"trial_$(trial).bin");open(io->write(io,payload),path,"w")
  println("STUDY_TRIAL {\"trial\":",trial,",\"time_seconds\":",total,",\"correct\":",ok,",\"checked_values\":",length(payload),",\"max_abs_error\":",ok ? 0 : "null",",\"timing_region\":\"sum3_full_merklize_contextsynchronized_alllevels\",\"jit_excluded\":true,\"internal_repeats\":1,\"timed_kernel_calls\":21,\"case_times_seconds\":",times,",\"output_artifact\":{\"path\":\"",path,"\",\"dtype\":\"uint64\",\"count\":",length(payload),",\"byte_order\":\"little\"}}")
  ok||error("AllInternalNodesBigIntegerModularRescueOracleGuards")
 end
end

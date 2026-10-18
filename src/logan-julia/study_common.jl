const GUARD=Int32(123456789)
function study(allocate,launch)
 f=collect(reinterpret(Int32,read(joinpath(@__DIR__,"fixtures_i32.bin"))));oracle=collect(reinterpret(Int32,read(joinpath(@__DIR__,"oracle_i32.bin"))));n=Int(f[1]);pos=2;q=Int32[];t=Int32[];oq=Int32[];ot=Int32[];seed=Int32[];drop=Int32(100);off=Int32(0)
 for i in 1:n
  nq=Int(f[pos]);nt=Int(f[pos+1]);drop=f[pos+2];pos+=3;append!(seed,f[pos:pos+9]);pos+=10;append!(q,f[pos:pos+nq-1]);pos+=nq;append!(t,f[pos:pos+nt-1]);pos+=nt;push!(oq,length(q));push!(ot,length(t));off=max(off,Int32(max(nq,nt)+2))
 end
 function guarded(T,data)
  a=allocate(T,length(data)+8);fill!(a,T(T==Int16 ? 12345 : T==UInt8 ? 77 : GUARD));copyto!(view(a,5:length(a)-4),T.(data));(a,view(a,5:length(a)-4))
 end
 qr,qd=guarded(UInt8,q);tr,td=guarded(UInt8,t);oqr,oqd=guarded(Int32,oq);otr,otd=guarded(Int32,ot)
 states=[];streams=[CUDA.CuStream(flags=CUDA.STREAM_NON_BLOCKING) for _ in 1:2]
 for dir in 1:2
  sr,sd=guarded(Int32,seed);rr,rd=guarded(Int32,zeros(Int32,n));ar,ad=guarded(Int16,zeros(Int16,3off*n));push!(states,(sr,sd,rr,rd,ar,ad))
 end
 for dir in 1:2
  sr,sd,rr,rd,ar,ad=states[dir];launch(sd,qd,td,rd,oqd,otd,off,ad,drop,Int32(dir),streams[dir])
 end
 CUDA.synchronize(CUDA.context())
 for trial in 1:5
  for (sr,sd,rr,rd,ar,ad) in states;copyto!(sd,seed);fill!(rd,0);fill!(ad,0);end
  CUDA.synchronize(CUDA.context());start=time_ns()
  for dir in 1:2
   sr,sd,rr,rd,ar,ad=states[dir];launch(sd,qd,td,rd,oqd,otd,off,ad,drop,Int32(dir),streams[dir])
  end
  CUDA.synchronize(CUDA.context());elapsed=(time_ns()-start)*1e-9;payload=copy(f);ok=true
  for (raw,original) in [(qr,q),(tr,t),(oqr,oq),(otr,ot)]
   got=Array(raw);g=eltype(got)==UInt8 ? 77 : GUARD;ok&=all(==(g),got[1:4])&&all(==(g),got[end-3:end])&&got[5:end-4]==original;encoded=Int32.(got);encoded[1:4].=GUARD;encoded[end-3:end].=GUARD;append!(payload,encoded)
  end
  for dir in 1:2
   sr,sd,rr,rd,ar,ad=states[dir];gs=Array(sr);gr=Array(rr);ga=Array(ar)
   for raw in (gs,gr);ok&=all(==(GUARD),raw[1:4])&&all(==(GUARD),raw[end-3:end]);end
   ok&=all(==(Int16(12345)),ga[1:4])&&all(==(Int16(12345)),ga[end-3:end])
   for i in 1:n
    s,c,row,lo,hi=oracle[5i-4:5i];expected=copy(seed[10i-9:10i]);expected[5]=0;expected[10]=0
    if dir==1;expected[1]-=row;expected[2]-=c;expected[6]=min(expected[6],expected[8]+lo);expected[7]=max(expected[7],expected[8]+hi)
    else;expected[3]+=row;expected[4]+=c;expected[7]=max(expected[7],expected[9]-lo);expected[6]=min(expected[6],expected[9]-hi);end
    gs[4+10i-5]=0 # dead source copy-constructor member not part of numerical contract
    ok&=gs[4+10i-9:4+10i]==expected && gr[i+4]==s
   end
   append!(payload,gs);append!(payload,gr);append!(payload,Int32.(ga))
  end
  path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"trial_$(trial).bin");open(io->write(io,payload),path,"w")
  println("STUDY_TRIAL {\"trial\":",trial,",\"time_seconds\":",elapsed,",\"correct\":",ok,",\"checked_values\":",length(payload),",\"max_abs_error\":",ok ? 0 : "null",",\"timing_region\":\"two_stream_left_right_kernel_batch_context_complete\",\"jit_excluded\":true,\"internal_repeats\":1,\"timed_kernel_calls\":2,\"output_artifact\":{\"path\":\"",path,"\",\"dtype\":\"int32\",\"count\":",length(payload),",\"byte_order\":\"little\"}}")
  ok||error("fullscores_seedmetadata_scratch_guards_exact_oracle")
 end
end

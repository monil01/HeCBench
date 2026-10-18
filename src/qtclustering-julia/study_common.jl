using CUDA,Printf
function qt_study(launch)
 pre=collect(reinterpret(Float64,read(joinpath(@__DIR__,"fixture.bin"))));pos=2;cases=Any[]
 function guarded(h)
  guard=eltype(h)<:Int8 ? Int8(-99) : convert(eltype(h),-12345)
  full=CuArray(vcat(fill(guard,8),h,fill(guard,8)))
  (full=full,p=view(full,9:length(h)+8),guard=guard,n=length(h),initial=copy(h))
 end
 for ci in 1:Int(pre[1])
  N,D=Int.(pre[pos:pos+1]);threshold=Float32(pre[pos+2]);pos+=3;points=pre[pos:pos+2*N-1];pos+=2*N
  idx=Int32.(pre[pos:pos+N*D-1]);pos+=N*D;dist=Float32.(pre[pos:pos+N*D-1]);pos+=N*D
  arrays=(dist=guarded(dist),idx=guarded(idx),degree=guarded(fill(Int32(-12345),N)),ungrp=guarded(Int32.(0:N-1)),ai=guarded(zeros(Int8,N*N)),cache=guarded(zeros(Float32,N*D)),clustered=guarded(zeros(Int8,N)),card=guarded(fill(Int32(-12345),2*N)),result=guarded(fill(Int32(-12345),N)))
  x=map(v->v.p,arrays);gold=collect(reinterpret(Float64,read(joinpath(@__DIR__,"oracle_case$(ci).bin"))))
  push!(cases,(N=N,D=D,threshold=threshold,ci=ci,a=arrays,x=x,gold=gold,idx=idx,dist=dist))
 end
 @assert pos==length(pre)+1
 function execute(c,measured)
  for a in c.a;copyto!(a.p,a.initial);end;CUDA.synchronize()
  times=Float64[];labels=String[];aux=Float64[];logical=Float64[c.N,c.D,0];pc=c.N;iteration=0;ok=true
  function stage(op,blocks,threads,seed)
   t=time_ns();launch(op,c.x,pc,c.N,c.D,c.threshold,blocks,seed,threads);CUDA.synchronize();push!(times,(time_ns()-t)*1e-9);push!(labels,"case$(c.ci):$(op):$(iteration)")
  end
  stage(:degree,c.N,64,0);append!(logical,Float64.(Array(c.x.degree)))
  while pc>0 && iteration<=c.N
   iteration+=1;blocks=pc;stage(:qtc,blocks,64,0)
   if blocks>1;stage(:reduce,blocks,1,0);end
   card=Array(c.x.card)[1:2];count,seed=Int.(card)
   if !(1<=count<=pc && 0<=seed<c.N);ok=false;break;end
   stage(:trim,blocks,64,seed);stage(:update,blocks,64,seed)
   rem=pc-count;append!(logical,Float64[pc,count,seed,rem]);append!(logical,Float64.(Array(c.x.result)[1:count]));append!(logical,Float64.(Array(c.x.ungrp)[1:rem]));append!(logical,Float64.(Array(c.x.clustered)))
   # Capture all actual mutable payloads, including stale tails, without pretending
   # the independent logical oracle predicts unobservable workspace contents.
   for name in (:degree,:ungrp,:ai,:cache,:clustered,:card,:result);append!(aux,Float64.(Array(getproperty(c.x,name))));end
   pc=rem
   if count<=1;break;end
  end
  logical[3]=iteration;ok &= logical==c.gold
  guards=Float64[]
  for a in c.a
   h=Array(a.full);g=vcat(h[1:8],h[end-7:end]);ok &= all(==(a.guard),g);append!(guards,Float64.(g))
  end
  index=Array(c.x.idx);dist=Array(c.x.dist);ok &= index==c.idx && reinterpret(UInt8,dist)==reinterpret(UInt8,c.dist)
  art=Float64[length(logical)];append!(art,logical);push!(art,length(aux));append!(art,aux);append!(art,Float64.(index));append!(art,Float64.(dist));append!(art,guards)
  ok,art,times,labels,iteration,pc
 end
 for c in cases;execute(c,false);end
 for trial in 1:5
  art=copy(pre);times=Float64[];labels=String[];iters=Int[];residual=Int[];ok=true
  for c in cases
   pass,v,ts,ls,ni,rem=execute(c,true);ok &=pass;append!(art,v);append!(times,ts);append!(labels,ls);push!(iters,ni);push!(residual,rem)
  end
  f=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"output_trial$(trial).bin");write(f,art)
  labeltext="["*join(("\""*s*"\"" for s in labels),",")*"]"
  @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":null,\"max_rel_error\":null,\"timing_region\":\"sum_complete_QT_degree_candidate_reduce_trim_update_launch_completion_pipeline\",\"jit_excluded\":true,\"internal_repeats\":1,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":%d,\"byte_order\":\"little\"},\"path_times_seconds\":%s,\"path_labels\":%s,\"convergence_iterations\":%s,\"source_stop_remaining\":%s}\n",trial,sum(times),ok ? "true" : "false",length(art),f,length(art),string(times),labeltext,string(iters),string(residual))
 end
end

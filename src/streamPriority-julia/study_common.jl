function initial(n,seed)
 pref=zeros(Float32,n);weights=seed==0 ? ones(Float32,n) : Float32[Float32(mod(i,17)-8)/8f0 for i in 0:n-1];gin=seed==0 ? ones(Float32,n) : Float32[Float32(mod(i,9)-4)/8f0 for i in 0:n-1];(pref,weights,gin,zeros(Float32,n),zeros(Float32,n))
end
function host_reference(init,batches)
 pref,weights,gin,act,gout=map(copy,init)
 for batch in 1:batches
  for j in eachindex(pref);i=Int32(j-1);act[j]=forward_value(pref[j],i);gout[j]=backward_value(act[j],gin[j],i);weights[j]=muladd(-0.01f0,gout[j],weights[j]);end
  for j in eachindex(pref);pref[j]=prefetch_value(Int32(j-1),Int32(batch));end
 end
 (pref,weights,gin,act,gout)
end
function study(allocate,launch)
 pr=CUDA.priority_range();lo=first(pr);hi=last(pr);println("STUDY_PRIORITIES least=",lo," greatest=",hi)
 train=CUDA.CuStream(flags=CUDA.STREAM_NON_BLOCKING,priority=hi);prefetch=CUDA.CuStream(flags=CUDA.STREAM_NON_BLOCKING,priority=lo);println("STUDY_ACTUAL_PRIORITIES train=",CUDA.priority(train)," prefetch=",CUDA.priority(prefetch));cases=[];prefix=Float32[]
 for(n,batches,seed)in((1025,4,0),(8193,5,1)),mode in 0:1
  init=initial(n,seed);dev=map(x->begin;d=allocate(Float32,length(x));copyto!(d,x);d;end,init);ref=host_reference(init,batches)
  for x in init;append!(prefix,x);end
  append!(prefix,Float32[n,batches,seed,mode,256,256,1,0.01f0,lo,hi]);push!(cases,(n,batches,seed,mode,init,dev,ref))
 end
 function run(c)
  n,batches,seed,mode,init,dev,ref=c;pref,weights,gin,act,gout=dev
  for(d,x)in zip(dev,init);copyto!(d,x);end
  fwd_done=CUDA.CuEvent();fetch_done=CUDA.CuEvent();CUDA.device_synchronize(blocking=true);start=time_ns()
  for batch in 1:batches
   if mode==0
    CUDA.stream!(CUDA.legacy_stream()) do
     launch(1,pref,weights,gin,act,gout,n,batch);launch(2,pref,weights,gin,act,gout,n,batch);launch(3,pref,weights,gin,act,gout,n,batch);launch(4,pref,weights,gin,act,gout,n,batch)
    end
   else
    CUDA.wait(fetch_done,train)
    CUDA.stream!(train) do
     launch(1,pref,weights,gin,act,gout,n,batch);CUDA.record(fwd_done,train);launch(2,pref,weights,gin,act,gout,n,batch);launch(3,pref,weights,gin,act,gout,n,batch)
    end
    CUDA.wait(fwd_done,prefetch)
    CUDA.stream!(prefetch) do;launch(4,pref,weights,gin,act,gout,n,batch);end
    CUDA.record(fetch_done,prefetch)
   end
  end
  if mode==0;CUDA.device_synchronize(blocking=true);else;CUDA.synchronize(train;blocking=true);end
  elapsed=(time_ns()-start)*1e-9;pending=mode==1 ? !CUDA.isdone(prefetch) : false
  if mode==1;CUDA.synchronize(prefetch;blocking=true);end
  CUDA.device_synchronize(blocking=true);out=Float32[];err=0.;ok=true
  for(d,x)in zip(dev,ref);v=Array(d);append!(out,v);e=maximum(abs.(v.-x));err=max(err,e);ok&=all(isfinite,v)&&e<=1e-3;end
  finalize(fwd_done);finalize(fetch_done)
  return out,elapsed,ok,err,pending
 end
 for c in cases;run(c);end
 for trial in 1:5
  out=copy(prefix);elapsed=0.;err=0.;ok=true;components=Float64[];pending=Bool[]
  for c in cases;v,t,pass,e,pend=run(c);append!(out,v);elapsed+=t;err=max(err,e);ok&=pass;push!(components,t);push!(pending,pend);end
  path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"trial_$(trial).bin");open(io->write(io,out),path,"w");println("STUDY_COMPONENT_SECONDS ",join(components,","));println("STUDY_PREFETCH_PENDING_AT_TIMER_END ",join(pending,","));println("STUDY_TRIAL {\"trial\":",trial,",\"time_seconds\":",elapsed,",\"correct\":",ok,",\"checked_values\":",length(out),",\"max_abs_error\":",err,",\"timing_region\":\"sum4_source_permode_completion_pipelines\",\"jit_excluded\":true,\"internal_repeats\":1,\"output_artifact\":{\"path\":\"",path,"\",\"dtype\":\"float32\",\"count\":",length(out),",\"byte_order\":\"little\"}}");ok||error("StreamFullState")
 end
 finalize(train);finalize(prefetch)
end

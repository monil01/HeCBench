const BATCH=2;const CLASSES=5;const PRIORS=1024;const TOPK=10
@inline function histogram_body(tid,group,scores,bins,threshold,priors)
 i=tid
 while i<priors
  @inbounds confidence=scores[group*priors+i+1]
  if confidence>threshold
   scaled=(confidence-threshold)/(1f0-threshold);bin=clamp(unsafe_trunc(Int32,scaled*2048f0),Int32(0),Int32(2047))-Int32(1)
   if bin>=0;CUDA.atomic_add!(pointer(bins,bin+1),Int32(1));end
  end
  i+=Int32(256)
 end
 return
end
@inline function suffix_body(tid,bins)
 if tid<Int32(32)
  previous=Int32(0)
  for iter in Int32(63):-Int32(1):Int32(0)
   idx=iter*Int32(32)+tid+Int32(1);@inbounds value=bins[idx];offset=Int32(1)
   while offset<Int32(32)
    other=CUDA.shfl_down_sync(UInt32(0xffffffff),value,offset)
    if Int32(31)-tid>=offset;value+=other;end
    offset*=Int32(2)
   end
   value+=previous;@inbounds bins[idx]=value;previous=CUDA.shfl_sync(UInt32(0xffffffff),value,Int32(1))
  end
 end
 return
end
@inline function select_body(tid,group,scores,indices,counts,bins,threshold,priors,topk)
 i=tid
 while i<priors
  @inbounds confidence=scores[group*priors+i+1]
  if confidence>threshold
   scaled=(confidence-threshold)/(1f0-threshold);bin=clamp(unsafe_trunc(Int32,scaled*2048f0),Int32(0),Int32(2047));idx=CUDA.atomic_add!(pointer(bins,bin+1),Int32(1))
   if idx<topk
    @inbounds indices[group*topk+idx+1]=i
    CUDA.atomic_add!(pointer(counts,group+1),Int32(1))
   end
  end
  i+=Int32(256)
 end
 return
end
function study(run!)
 scores=Float32[Float32(mod(73*i+19*g,PRIORS))/Float32(PRIORS) for g in 0:BATCH*CLASSES-1 for i in 0:PRIORS-1];S=CuArray(scores);I=CUDA.zeros(Int32,BATCH*CLASSES*TOPK);C=CUDA.zeros(Int32,BATCH*CLASSES);template=CUDA.zeros(Int32,2048);want=Int32[]
 for g in 0:BATCH*CLASSES-1
  selected=sortperm(scores[g*PRIORS+1:(g+1)*PRIORS];rev=true)[1:TOPK].-1;append!(want,sort(Int32.(selected)));push!(want,Int32(TOPK))
 end
 run!(I,C,S,template,0.4f0,Int32(PRIORS),Int32(TOPK));CUDA.synchronize()
 for trial in 1:5
  fill!(I,Int32(0));CUDA.synchronize();start=time_ns();for repeat in 1:10;run!(I,C,S,template,0.4f0,Int32(PRIORS),Int32(TOPK));end;CUDA.synchronize();elapsed=(time_ns()-start)*1e-9/10;raw=Array(I);counts=Array(C);got=Int32[]
  for g in 0:BATCH*CLASSES-1;append!(got,sort(raw[g*TOPK+1:(g+1)*TOPK]));push!(got,counts[g+1]);end
  ok=got==want;path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"trial_$(trial).bin");open(io->write(io,got),path,"w")
  println("STUDY_TRIAL {\"trial\":",trial,",\"time_seconds\":",elapsed,",\"correct\":",ok,",\"checked_values\":110,\"max_abs_error\":",ok ? 0 : 1,",\"timing_region\":\"approximate_histogram_topk_kernel_mean\",\"jit_excluded\":true,\"internal_repeats\":10,\"output_artifact\":{\"path\":\"",path,"\",\"dtype\":\"int32\",\"count\":110,\"byte_order\":\"little\"}}")
  ok||error("Fullselectedsets/counts vsindependentexactTopK failed")
 end
end

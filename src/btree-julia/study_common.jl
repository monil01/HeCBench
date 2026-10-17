const STUDY_CASES=(256,4096)
const POOL_WORDS=Int32(1<<18)
function input_fixture(n)
 keys=UInt32[(i*197+53)%n for i in 0:n-1]
 values=UInt32[((Int(k)*104729+97)%10000019)+11 for k in keys]
 queries=UInt32[(i*313+19)%(2*n) for i in 0:2*n-1]
 return keys,values,queries
end
function canonical(pool,nodes,n)
 records=Tuple{UInt32,UInt32}[]
 for node in 0:Int(nodes)-1
  base=node*32
  if (pool[base+1]&HI)==0
   for j in 0:2:28
    key=pool[base+j+1]&LO
    if key>=UInt32(2);push!(records,(key-UInt32(2),(pool[base+j+2]&LO)-UInt32(2)));end
   end
  end
 end
 sort!(records);words=UInt32[]
 for(k,v)in records;push!(words,k,v);end
 return words,length(records)==n
end
function study(init!,insert!,search!)
 cases=[];prefix=UInt32[]
 for n in STUDY_CASES
  k,v,q=input_fixture(n);append!(prefix,k);append!(prefix,v);append!(prefix,q);append!(prefix,UInt32[n,2*n,POOL_WORDS,256])
  wantq=UInt32[x<n ? ((Int(x)*104729+97)%10000019)+11 : 0 for x in q]
  wantpairs=UInt32[]
  for x in 0:n-1;push!(wantpairs,UInt32(x),UInt32(((x*104729+97)%10000019)+11));end
  push!(cases,(n,CuArray(k),CuArray(v),CuArray(q),CUDA.zeros(UInt32,2*n),wantq,wantpairs,CUDA.zeros(UInt32,POOL_WORDS),CUDA.zeros(UInt32,1),CUDA.zeros(UInt32,1)))
 end
 function trialcase(c)
  n,k,v,q,out,wantq,wantpairs,pool,count,root=c
  CUDA.fill!(pool,0);CUDA.fill!(count,0);CUDA.fill!(root,0);CUDA.fill!(out,0);init!(pool,count,root);CUDA.synchronize()
  t=time_ns();insert!(pool,count,root,k,v,Int32(n));search!(pool,root,q,out,Int32(2*n));CUDA.synchronize();elapsed=(time_ns()-t)*1e-9
  used=Array(count)[1];used<=POOL_WORDS÷32||Base.error("TreePoolCapacityExceeded")
  pairs,valid=canonical(Array(pool),used,n);got=Array(out);ok=valid&&pairs==wantpairs&&got==wantq
  return vcat(got,pairs),elapsed,ok
 end
 for c in cases;trialcase(c);end
 for trial in 1:5
  data=copy(prefix);elapsed=0.;ok=true
  for c in cases;got,t,pass=trialcase(c);append!(data,got);elapsed+=t;ok&=pass;end
  path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"trial_$(trial).bin");open(io->write(io,data),path,"w")
  println("STUDY_TRIAL {\"trial\":",trial,",\"time_seconds\":",elapsed,",\"correct\":",ok,",\"checked_values\":",length(data),",\"timing_region\":\"sum_two_build_and_search_kernel_pairs\",\"jit_excluded\":true,\"internal_repeats\":1,\"output_artifact\":{\"path\":\"",path,"\",\"dtype\":\"uint32\",\"count\":",length(data),",\"byte_order\":\"little\"}}")
  ok||Base.error("TreeQueryOrCompleteLeafDictionaryMismatch")
 end
end

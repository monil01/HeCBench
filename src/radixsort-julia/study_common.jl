function fixture(n)
 out=Vector{UInt32}(undef,n)
 for i in 0:n-1
  out[i+1]=n==32768 ? (i%11==0 ? UInt32(0) : (i%13==0 ? typemax(UInt32) : (i%17==0 ? (UInt32(0x80000000)|UInt32(i&255)) : ((UInt32(i)*UInt32(2654435761))⊻UInt32(i>>7)⊻UInt32(0xa5a55a5a))))) : (UInt32((n-1-i)%1024)|(UInt32((i÷1024)%16)<<28))
 end
 return out
end
function scratch_oracle(input)
 n=length(input);nb=n÷256;cn=n÷16;keys=sort(input);pre=sort(keys,by=x->x&UInt32(0x0fffffff),alg=Base.Sort.MergeSort);tmp=similar(keys)
 for base in 0:512:n-1;tmp[base+1:base+512]=sort(pre[base+1:base+512],by=x->x>>28,alg=Base.Sort.MergeSort);end
 counts=zeros(UInt32,cn);blocks=zeros(UInt32,cn)
 for g in 0:nb-1,pos in 0:255
  d=Int(tmp[g*256+pos+1]>>28);index=d*nb+g+1
  if counts[index]==0;blocks[g*16+d+1]=UInt32(pos);end
  counts[index]+=UInt32(1)
 end
 scan=zeros(UInt32,cn);running=UInt32(0)
 for i in 1:cn;scan[i]=running;running+=counts[i];end
 buffer=UInt32[scan[g*1024+1] for g in 0:cn÷1024-1]
 return vcat(keys,tmp,counts,blocks,scan,buffer)
end
function study(run,allocate)
 cases=[];prefix=UInt32[]
 for n in (32768,65536)
  input=fixture(n);cn=n÷16;bn=cn÷1024;keys=allocate(UInt32,n);tmp=allocate(UInt32,n);counts=allocate(UInt32,cn);blocks=allocate(UInt32,cn);scan=allocate(UInt32,cn);buffer=allocate(UInt32,bn);want=scratch_oracle(input);append!(prefix,input);append!(prefix,zeros(UInt32,n+3cn+bn));append!(prefix,UInt32[n,32,3,128,256,4,1]);push!(cases,(n,input,keys,tmp,counts,blocks,scan,buffer,want))
 end
 function case!(c)
  n,input,keys,tmp,counts,blocks,scan,buffer,want=c;copyto!(keys,input)
  for x in (tmp,counts,blocks,scan,buffer);CUDA.fill!(x,UInt32(0));end
  CUDA.synchronize(CUDA.context());start=time_ns()
  for _ in 1:3;run(keys,tmp,counts,blocks,scan,buffer,Int32(n));end
  CUDA.synchronize(CUDA.context());elapsed=(time_ns()-start)*1e-9/3;got=vcat(Array(keys),Array(tmp),Array(counts),Array(blocks),Array(scan),Array(buffer));return got,elapsed,got==want
 end
 for c in cases;got,t,ok=case!(c);ok||error("WarmupFullRadixScratchMismatch");end
 for trial in 1:5
  out=copy(prefix);elapsed=0.;ok=true
  for c in cases;got,t,pass=case!(c);append!(out,got);elapsed+=t;ok&=pass;end
  path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"trial_$(trial).bin");open(io->write(io,out),path,"w")
  println("STUDY_TRIAL {\"trial\":",trial,",\"time_seconds\":",elapsed,",\"correct\":",ok,",\"checked_values\":",length(out),",\"max_abs_error\":0,\"timing_region\":\"sum_two_three_full48kernel_sort_means\",\"jit_excluded\":true,\"internal_repeats\":3,\"output_artifact\":{\"path\":\"",path,"\",\"dtype\":\"uint32\",\"count\":",length(out),",\"byte_order\":\"little\"}}")
  ok||error("FullSortedKeysAndScratchMismatch")
 end
end

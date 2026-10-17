const SHAPES=((3,32,3,false),(127,64,4,false),(3,128,3,false),(600,32,3,false),(257,128,4,true))
function fixture(tokens,E,K,mapped)
 n=tokens*K;top=Int32[(mapped&&i%17==0) ? E+2 : ((i÷K)*17+(i%K)*13)%E for i in 0:n-1];mapping=Int32[mapped ? (e%7==0 ? -1 : E-1-e) : e for e in 0:E-1];maxpad=cld(n+E*31,32)*32
 if n<E;maxpad=min(n*32,maxpad);end
 groups=[Int32[] for _ in 1:E]
 for i in 0:n-1
  e=top[i+1]
  if e>=E;continue;end
  e=mapped ? mapping[e+1] : e
  if e<0;continue;end
  push!(groups[e+1],Int32(i))
 end
 counts=Int32[length(g) for g in groups];cum=zeros(Int32,E+1)
 for e in 1:E;cum[e+1]=cum[e]+cld(counts[e],Int32(32))*Int32(32);end
 sorted=fill(Int32(n),maxpad);experts=zeros(Int32,maxpad÷32)
 for e in 1:E
  for(i,token)in enumerate(groups[e]);sorted[Int(cum[e])+i]=token;end
  for block in Int(cum[e])÷32:Int(cum[e+1])÷32-1;experts[block+1]=Int32(e-1);end
 end
 small=n<1024&&E<=64;mutcum=copy(cum)
 if !small;for e in 1:E;mutcum[e]+=counts[e];end;end
 return top,mapping,maxpad,groups,cum,vcat(sorted,experts,Int32[cum[end]],small ? Int32[] : mutcum)
end
function study(run)
 cases=[];prefix=Int32[]
 for(tokens,E,K,mapped)in SHAPES
  top,mapping,maxpad,groups,cum,want=fixture(tokens,E,K,mapped);n=length(top);append!(prefix,top);append!(prefix,mapping);append!(prefix,Int32[tokens,E,K,mapped,n,maxpad,maxpad÷32,3]);push!(cases,(tokens,E,K,mapped,top,mapping,maxpad,groups,cum,want,CuArray(top),CuArray(mapping),CUDA.zeros(Int32,maxpad),CUDA.zeros(Int32,maxpad÷32),CUDA.zeros(Int32,1),CUDA.zeros(Int32,E+1)))
 end
 function case!(c)
  tokens,E,K,mapped,top,mapping,maxpad,groups,cum,want,dt,dm,sorted,experts,total,scratch=c;CUDA.synchronize(CUDA.context());t=time_ns()
  for _ in 1:3;run(dt,sorted,experts,total,dm,Int32(E),Int32(length(top)),scratch,Int32(maxpad),Int32(maxpad÷32),mapped);end
  CUDA.synchronize(CUDA.context());elapsed=(time_ns()-t)*1e-9/3;raw=Array(sorted);ids=Array(experts);post=Array(total);ok=length(raw)==maxpad
  # Canonicalize only irrelevant within-expert order; preserve all padding and complete outputs.
  for e in 1:E
   first=Int(cum[e])+1;last=Int(cum[e+1]);if first<=last;sort!(@view raw[first:last]);end
  end
  got=vcat(raw,ids,post,length(top)<1024&&E<=64 ? Int32[] : Array(scratch));return got,elapsed,ok&&got==want
 end
 for c in cases;case!(c);end
 for trial in 1:5
  out=copy(prefix);elapsed=0.;ok=true
  for c in cases;got,t,pass=case!(c);append!(out,got);elapsed+=t;ok&=pass;end
  path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"trial_$(trial).bin");open(io->write(io,out),path,"w")
  println("STUDY_TRIAL {\"trial\":",trial,",\"time_seconds\":",elapsed,",\"correct\":",ok,",\"checked_values\":",length(out),",\"max_abs_error\":0,\"timing_region\":\"sum_five_three_alignment_pipeline_means\",\"jit_excluded\":true,\"internal_repeats\":3,\"output_artifact\":{\"path\":\"",path,"\",\"dtype\":\"int32\",\"count\":",length(out),",\"byte_order\":\"little\"}}")
  ok||error("FullExpertTokenPartitionPaddingPrefixMismatch")
 end
end

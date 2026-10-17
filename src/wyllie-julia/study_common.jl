using CUDA,Printf
function fixture(n,random)
 next=zeros(Int32,n)
 if random
  ccall(:srand,Cvoid,(Cuint,),123);free=trues(n);tail=ccall(:rand,Cint,())%(n-1)+1;free[tail+1]=false;next[tail+1]=-1;prev=tail;left=n
  while (left-=1)>1
   pos=ccall(:rand,Cint,())%(n-1)+1
   while !free[pos+1];pos=ccall(:rand,Cint,())%(n-1)+1;end
   free[pos+1]=false;next[pos+1]=Int32(prev);prev=pos
  end
  next[1]=Int32(prev)
 else
  for i in 1:n;next[i]=i==n ? Int32(-1) : Int32(i);end
 end
 # Independent serial traversal checks allnodes/cycleabsence anddistance-to-tail.
 ranks=zeros(Int64,n);seen=falses(n);node=0;tail=-1
 for dist in n-1:-1:0
  0<=node<n&&!seen[node+1]||error("not a single rooted acyclic list")
  seen[node+1]=true;ranks[node+1]=dist;tail=node;node=next[node+1]
 end
 node==-1&&all(seen)||error("invalid chain")
 packed=Int64[(Int64(next[i])<<32)|Int64(next[i]==-1 ? 0 : 1) for i in 1:n]
 expected=Int64[((i-1==tail ? Int64(-1) : Int64(tail))<<32)|ranks[i] for i in 1:n]
 (n,random,next,packed,ranks,expected,tail)
end
function study_main(launch!)
 rep=isempty(ARGS) ? 5 : parse(Int,ARGS[1]);fs=[fixture(n,random) for n in (513,1027) for random in (false,true)];ds=[CuArray(f[4]) for f in fs];prefix=vcat([vcat(Int64[f[1],f[2]],Int64.(f[3]),f[4]) for f in fs]...)
 for (f,d) in zip(fs,ds);launch!(d,Int32(f[1]));end;CUDA.synchronize()
 for trial in 1:5
  elapsed=0.0;gct=0.0;gcp=0;ok=true
  for _ in 1:rep,(f,d) in zip(fs,ds)
   copyto!(d,f[4]);CUDA.synchronize();gb=Base.gc_num();st=time_ns();launch!(d,Int32(f[1]));CUDA.synchronize();elapsed+=(time_ns()-st)*1e-9/rep;ga=Base.gc_num();gct+=(ga.total_time-gb.total_time)*1e-9;gcp+=ga.pause-gb.pause
  end
  got=Int64[]
  for (f,d) in zip(fs,ds)
   raw=Array(d);ranks=raw.&Int64(0xffffffff);ok&=raw==f[6]&&ranks==f[5];append!(got,raw);append!(got,ranks)
  end
  art=vcat(prefix,got);path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"output_trial$(trial).bin");write(path,art)
  @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":0.0,\"max_rel_error\":0.0,\"timing_region\":\"sum_four_inplace_list_rank_kernels_eachresetexcluded\",\"jit_excluded\":true,\"internal_repeats\":%d,\"gc_time_seconds_whole_region\":%.17g,\"gc_pauses\":%d,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"int64\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,elapsed,ok ? "true" : "false",length(got),rep,gct,gcp,path,length(art));ok||error("exactfullpackedstate andindependentranktraversal")
 end
end

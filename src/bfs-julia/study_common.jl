using CUDA,Printf
function study_graph(f)
 N=f==1 ? 513 : 257;adj=[Int32[] for _ in 1:N]
 if f==1
  for i in 0:480
   push!(adj[i+1],Int32(i))
   for j in (2i+1,2i+2);if j<=480;push!(adj[i+1],Int32(j));push!(adj[i+1],Int32(j));end;end
   if i>0;push!(adj[i+1],Int32((i-1)÷2));end
  end
  for i in 481:512;push!(adj[i+1],Int32(481+(i-481+1)%32));end
 else
  append!(adj[1],Int32.(1:32))
  for i in 1:32
   push!(adj[i+1],Int32(0))
   for k in 0:7;push!(adj[i+1],Int32(33+((i-1)*7+k*17)%128));end
  end
  for i in 33:160
   push!(adj[i+1],Int32(i))
   push!(adj[i+1],Int32(161+((i-33)*5)%64));push!(adj[i+1],Int32(161+((i-33)*5+1)%64))
  end
  for i in 161:224;push!(adj[i+1],Int32(1+(i-161)%32));end
  for i in 225:256;push!(adj[i+1],Int32(225+(i-225+1)%32));end
 end
 starts=Int32[];counts=Int32[];edges=Int32[]
 for a in adj;push!(starts,Int32(length(edges)));push!(counts,Int32(length(a)));append!(edges,a);end
 # Independent queue BFS, unlike level-frontier original source checker.
 dist=fill(Int32(-1),N);dist[1]=0;queue=Int[0];head=1
 while head<=length(queue)
  i=queue[head];head+=1
  for j in adj[i+1]
   if dist[j+1]==-1;dist[j+1]=dist[i+1]+1;push!(queue,Int(j));end
  end
 end
 starts,counts,edges,dist
end
function study_main(expand!,update!)
 graphs=[study_graph(f) for f in 1:2];prefix=Int32[];states=[]
 for (s,n,e,gold) in graphs
  N=length(s);append!(prefix,vcat(s,n,e,Int32[0]));m=zeros(Int8,N);v=copy(m);m[1]=v[1]=1;c=fill(Int32(-1),N);c[1]=0
  push!(states,(CuArray(s),CuArray(n),CuArray(e),CuArray(m),CuArray(zeros(Int8,N)),CuArray(v),CuArray(c),CuArray(Int8[0]),m,v,c))
 end
 function run!(state)
  ds,dn,de,dm,du,dv,dc,dover,m,v,c=state;N=length(m)
  copyto!(dm,m);fill!(du,Int8(0));copyto!(dv,v);copyto!(dc,c);CUDA.synchronize();elapsed=0.0;levels=0
  while true
   copyto!(dover,Int8[0]);CUDA.synchronize();st=time_ns()
   expand!(ds,dn,de,dm,du,dv,dc,N);update!(dm,du,dv,dover,N);CUDA.synchronize();elapsed+=(time_ns()-st)*1e-9
   levels+=1;over=Array(dover)[1];levels<=N||error("termination bound");if over==0;break;end
  end
  out=vcat(Array(dc),Int32.(Array(dv)),Int32.(Array(dm)),Int32.(Array(du)),Int32[0,levels]);out,elapsed,levels
 end
 for st in states;run!(st);end
 for trial in 1:5
  got=Int32[];elapsed=0.0;ok=true
  for (idx,st) in enumerate(states)
   v,t,levels=run!(st);N=length(graphs[idx][1]);gold=graphs[idx][4];ok&=v[1:N]==gold&&v[N+1:2N]==Int32.(gold.>=0)&&all(iszero,v[2N+1:4N+1])&&levels==maximum(gold)+1
   append!(got,v);elapsed+=t
  end
  art=vcat(prefix,got);path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"output_trial$(trial).bin");write(path,art)
  @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":0.0,\"max_rel_error\":0.0,\"timing_region\":\"sum_all_converged_frontier_kernel_pairs_excluding_flag_transfers\",\"jit_excluded\":true,\"internal_repeats\":1,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"int32\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,elapsed,ok ? "true" : "false",length(got),path,length(art));ok||error("FIFO BFS fulltermination contract")
 end
end

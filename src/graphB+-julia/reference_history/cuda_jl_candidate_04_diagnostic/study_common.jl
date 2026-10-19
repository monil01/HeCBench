const ROOT=dirname(@__FILE__)
const CHECKLIB=joinpath(ROOT,"study_check.so")
mutable struct GState{A,D}
 n::Int;e::Int;iterations::Int;idx::Vector{Int32};list::Vector{Int32};wei::Vector{Int32};orig::Vector{Int32};roots::Vector{Int32};buffers::A;device::D;blocks::Int
end
function alloc(::Type{T},n)where T
 storage=CuArray{T}(undef,n+64);CUDA.memset(convert(CUDA.CuPtr{UInt8},pointer(storage)),UInt8(165),length(storage)*sizeof(T));(storage=storage,data=view(storage,33:n+32),n=n)
end
function copyinput!(b,x)
 copyto!(b.data,x);nothing
end
function states()
 a=reinterpret(Int64,read(joinpath(ROOT,"fixtures.bin")));p=1;nc=a[p];p+=1;out=[]
 take(n)=begin v=Int32.(a[p:p+n-1]);p+=n;v end
 for k in 1:nc
  n=Int(a[p]);e=Int(a[p+1]);it=Int(a[p+2]);p+=3;idx=take(n+1);list=take(e);wei=take(e);orig=take(n);roots=take(it)
  for _ in 1:it
   p+=1+5n+1;w=Int(a[p]);p+=1+2w+4*e+n
  end
  bs=(idx=alloc(Int32,n+1),list=alloc(Int32,e),wei=alloc(Int32,e),orig=alloc(Int32,n),roots=alloc(Int32,it),parent=alloc(Int32,n),queue=alloc(Int32,n),label=alloc(Int32,n),tail=alloc(Int32,1),inCC=alloc(Int32,n),inTree=alloc(Int32,e),neg=alloc(Int32,e),ws1=alloc(Int32,e),ws2=alloc(Int32,e),info=alloc(Int32,2*e),minus=alloc(UInt8,e),changed=alloc(UInt8,1),hi=alloc(UInt64,1),wsize=alloc(Int32,1))
  for(f,v)in ((:idx,idx),(:wei,wei),(:orig,orig),(:roots,roots));copyinput!(getproperty(bs,f),v);end
  d=map(b->b.data,bs);push!(out,GState(n,e,it,idx,list,wei,orig,roots,bs,d,640))
 end
 p==length(a)+1||error("fixture parse mismatch");out
end
function reset!(s)
 copyinput!(s.buffers.list,s.list)
 for k in (:parent,:queue,:label,:tail,:inCC,:inTree,:neg,:ws1,:ws2,:info,:minus,:changed,:hi,:wsize)
  b=getproperty(s.buffers,k);CUDA.memset(convert(CUDA.CuPtr{UInt8},pointer(b.data)),UInt8(0),b.n*sizeof(eltype(b.storage)))
 end
 launch(1,s);CUDA.device_synchronize();nothing
end
readback(s,k)=Array(getproperty(s.device,k))
function prefix(s)
 out=Int64[];for k in (:idx,:list,:wei,:orig,:roots);append!(out,readback(s,k));end;out
end
function guards(s)
 bad=0
 for b in values(s.buffers)
  a=Array(b.storage);z=reinterpret(UInt8,a);g=32*sizeof(eltype(a));bad+=count(!=(UInt8(165)),view(z,1:g));bad+=count(!=(UInt8(165)),view(z,length(z)-g+1:length(z)))
 end
 bad
end
function capture(s,dfs,depth,path)
 list=readback(s,:list);info=readback(s,:info);it=readback(s,:inTree);neg=readback(s,:neg);mi=readback(s,:minus);w=Int(only(readback(s,:wsize)));w1=readback(s,:ws1);w2=readback(s,:ws2)
 out=Int64[];append!(out,readback(s,:parent));append!(out,dfs);for k in (:label,:queue,:inCC);append!(out,readback(s,k));end
 for u in 1:s.n
  order=collect(Int(s.idx[u])+1:Int(s.idx[u+1]));sort!(order;by=j->list[j]>>1)
  # Full physical row permutation must preserve each original neighbor exactly once.
  sort(list[order].>>1)==s.list[Int(s.idx[u])+1:Int(s.idx[u+1])]||error("CSR edge loss/duplicate")
  for j in order;append!(out,(list[j],info[2j-1],info[2j],it[j],neg[j],Int(mi[j])));end
 end
 push!(out,reinterpret(Int64,only(readback(s,:hi))));push!(out,w);pairs=sort([(w1[i],w2[i]) for i in 1:w]);for q in pairs;append!(out,q);end
 push!(out,depth);push!(out,guards(s))
 if !isempty(path)
  raw=Int64[];append!(raw,list);append!(raw,info);append!(raw,w1);append!(raw,w2);open(io->write(io,raw),path,"w");open(io->write(io,out),path*".canonical.i64","w")
 end
 out
end
function pipeline(s,replay,folder,trial)
 shots=Vector{Int64}[];dfs=Int64[];depth=0;calls=Int[];CUDA.device_synchronize();t=time_ns()
 for it in 0:s.iterations-1
  launch(2,s;root=s.roots[it+1]);ct=1;level=0;tail=1;border=[0,1]
  while border[level+2]<s.n
   launch(3,s;start=border[level+1],stop=border[level+2],level=level,seed=it+17);ct+=1;tail=Int(only(readback(s,:tail)));level+=1;push!(border,tail)
   (level>s.n||tail==border[level+1])&&error("BFS no progress")
  end
  depth=level
  for l in level:-1:1;launch(4,s;start=border[l+1],stop=border[l+2],level=l);ct+=1;end
  CUDA.memset(convert(CUDA.CuPtr{UInt8},pointer(s.device.label,Int(s.roots[it+1])+1)),UInt8(0),4)
  for l in 0:level;launch(5,s;start=border[l+1],stop=border[l+2],level=l);ct+=1;end
  replay&&(dfs=Int64.(readback(s,:label)))
  for stage in 6:14;launch(stage,s);ct+=1;end
  rounds=0
  while true
   CUDA.memset(convert(CUDA.CuPtr{UInt8},pointer(s.device.changed)),UInt8(0),1)
   for _ in 1:3;launch(15,s);ct+=1;end
   changed=only(readback(s,:changed))!=0;rounds+=1;rounds>s.n+2&&error("BF no convergence");!changed&&break
  end
  launch(16,s);ct+=1;push!(calls,ct)
  if replay;push!(shots,capture(s,dfs,level,joinpath(folder,"$(MODEL)_$(getpid())_trial$(trial)_n$(s.n)_it$(it).raw.i64")));end
 end
 CUDA.device_synchronize();seconds=(time_ns()-t)*1e-9;(seconds=seconds,snapshots=shots,dfs=dfs,depth=depth,launchcounts=calls)
end
function main()
 length(ARGS)==1||error("persistent raw evidence directory required");folder=ARGS[1];mkpath(folder);ss=states()
 for s in ss;reset!(s);pipeline(s,false,folder,0);end
 for trial in 1:5
  pre=Int64[];outputs=Int64[];seconds=0.0
  for s in ss
   reset!(s);append!(pre,prefix(s));timed=pipeline(s,false,folder,trial);seconds+=timed.seconds
   actual=capture(s,zeros(Int64,s.n),timed.depth,joinpath(folder,"$(MODEL)_$(getpid())_trial$(trial)_n$(s.n)_timed.raw.i64"));reset!(s);replay=pipeline(s,true,folder,trial);actual[s.n+1:2*s.n]=replay.dfs
   open(io->write(io,actual),joinpath(folder,"$(MODEL)_$(getpid())_trial$(trial)_n$(s.n)_timed_with_replay_dfs.canonical.i64"),"w");actual==replay.snapshots[end]||error("actual timed final differs from replay")
   push!(outputs,s.iterations+1);push!(outputs,length(actual));append!(outputs,actual)
   for snap in replay.snapshots;push!(outputs,length(snap));append!(outputs,snap);end
   println(stderr,"n",s.n," trial",trial," actual timed launchcounts ",timed.launchcounts)
  end
  append!(pre,outputs);path="trial$(trial).bin";open(io->write(io,pre),path,"w");ok=ccall((:graph_check,CHECKLIB),Cint,(Ptr{Int64},Csize_t),pre,length(pre))==1
  println("STUDY_TRIAL {\"trial\":",trial,",\"time_seconds\":",seconds,",\"correct\":",ok,",\"checked_values\":",length(pre),",\"max_abs_error\":",ok ? "0" : "null",",\"max_rel_error\":",ok ? "0" : "null",",\"timing_region\":\"sum3_full_graph_balance_pipeline\",\"jit_excluded\":true,\"internal_repeats\":1,\"output_artifact\":{\"path\":\"",path,"\",\"dtype\":\"int64\",\"count\":",length(pre),",\"byte_order\":\"little\"}}")
  ok||error("independent graph checker rejected")
 end
end

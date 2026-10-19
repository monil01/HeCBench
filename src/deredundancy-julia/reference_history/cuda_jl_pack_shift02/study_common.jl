using Libdl,Printf
function readcases()
 open(joinpath(@__DIR__,"fixtures.bin"))do f
  cases=[]
  for _ in 1:read(f,UInt32)
   h=read!(f,Vector{UInt32}(undef,6));threshold=read(f,Float32);n=Int(h[1]);L=read!(f,Vector{Int32}(undef,n));offsets=read!(f,Vector{Int64}(undef,n+1));raw=read!(f,Vector{UInt8}(undef,h[3]));skip(f,h[6]);seq=[String(copy(raw[offsets[i]+1:offsets[i]+L[i]])) for i in 1:n]
   push!(cases,(n=Int32(n),k=Int32(h[2]),total=Int(h[3]),np=Int(h[4]),ne=Int(h[5]),threshold=threshold,L=L,offsetinput=offsets,raw=raw,seq=seq))
  end
  cases
 end
end
function guarded(::Type{T},n,sentinel)where T
 h=fill(T(sentinel),n+16);h[9:end-8].=T(0);parent=CuArray(h);(parent=parent,data=view(parent,9:n+8))
end
function setup(raw)
 n,t,np=Int(raw.n),raw.total,raw.np
 c=(;raw...,lengths=guarded(Int32,n,-777),offsets=guarded(Int64,n+1,-777),reads=guarded(UInt8,t,165),compressed=guarded(UInt32,np,0xdeadbeef),gaps=guarded(Int32,n,-777),index=guarded(UInt16,t,65534),orders=guarded(UInt16,t,65534),words=guarded(Int64,n,-777),magic=guarded(Int32,4n,-777),cutoff=guarded(Int32,n,-777),cluster=guarded(Int32,n,-777),table=guarded(UInt16,65536,65534),rep=guarded(Int32,1,-777),hindex=Vector{UInt16}(undef,t),hoff=Vector{Int64}(undef,n+1),hwords=Vector{Int64}(undef,n),hcluster=Vector{Int32}(undef,n),hr=Int32[-1])
 reset!(c);c
end
function reset!(c)
 copyto!(c.lengths.data,c.L);copyto!(c.offsets.data,c.offsetinput);copyto!(c.reads.data,c.raw)
 for b in(c.compressed,c.gaps,c.index,c.orders,c.words,c.magic,c.cutoff,c.table);fill!(b.data,zero(eltype(b.data)));end
 fill!(c.cluster.data,Int32(-1));fill!(c.rep.data,Int32(-1));c.hr[1]=-1;nothing
end
function appendactual!(v,b);append!(v,Int64.(Array(b.data)));nothing;end
function prefix!(v,c)
 for b in(c.lengths,c.offsets,c.reads,c.compressed,c.gaps,c.index,c.orders,c.words,c.magic,c.cutoff,c.cluster,c.table,c.rep);appendactual!(v,b);end
 nothing
end
function pipeline!(c,tracing,rawindex,trace)
 copyto!(c.lengths.data,c.L);copyto!(c.offsets.data,c.offsetinput);copyto!(c.reads.data,c.raw)
 launch_base(c);launch_compress(c);launch_index(c);launch_cutoff(c)
 copyto!(c.hindex,c.index.data);copyto!(c.hoff,c.offsets.data);copyto!(c.hwords,c.words.data)
 if tracing;append!(rawindex,Int64.(c.hindex));end
 for i in 1:Int(c.n)
  start=c.hoff[i];dst=start
  @inbounds for j in start+1:start+c.L[i]
   if c.hindex[j]!=65535;dst+=1;c.hindex[dst]=c.hindex[j];end
  end
  dst-start==c.hwords[i]||error("validwordcount")
  sort!(view(c.hindex,start+1:start+c.hwords[i]))
 end
 copyto!(c.index.data,c.hindex);launch_merge(c)
 r=Int32(-1)
 while r<c.n
  c.hr[1]=r;copyto!(c.rep.data,c.hr);launch_rep(c);copyto!(c.hr,c.rep.data);r=c.hr[1]
  if tracing;push!(trace,Int64(r));appendactual!(trace,c.cluster);end
  r>=c.n-1&&break
  launch_make(c,r);if tracing;appendactual!(trace,c.table);end
  launch_magic(c,r);if tracing;appendactual!(trace,c.cluster);end
  launch_filter(c);if tracing;appendactual!(trace,c.cluster);end
  launch_align(c,r);if tracing;appendactual!(trace,c.cluster);end
  launch_clean(c,r);if tracing;appendactual!(trace,c.table);end
 end
 copyto!(c.hcluster,c.cluster.data);CUDA.device_synchronize();nothing
end
function finalstate(c)
 v=Int64[]
 for b in(c.reads,c.compressed,c.gaps,c.index,c.orders,c.words,c.magic,c.cutoff,c.cluster,c.table,c.rep);appendactual!(v,b);end
 v
end
function fasta(c)
 io=IOBuffer()
 for i in 1:Int(c.n)
  if c.hcluster[i]==i-1;print(io,">read_",i-1,'\n',c.seq[i],'\n');end
 end
 take!(io)
end
function guards!(out,c)
 for b in(c.lengths,c.offsets,c.reads,c.compressed,c.gaps,c.index,c.orders,c.words,c.magic,c.cutoff,c.cluster,c.table,c.rep)
  a=Array(b.parent);append!(out,Int64.(a[1:8]));append!(out,Int64.(a[end-7:end]))
 end
end
function runstudy(cases)
 prefix=Int64[];out=Int64[];seconds=0.
 for raw in cases
  c=setup(raw);prefix!(prefix,c);unused=Int64[];CUDA.device_synchronize();start=time_ns();pipeline!(c,false,unused,unused);seconds+=(time_ns()-start)*1e-9
  timed=finalstate(c);append!(out,timed);text=fasta(c);push!(out,Int64(length(text)));append!(out,Int64.(text))
  reset!(c);rawindex=Int64[];trace=Int64[];pipeline!(c,true,rawindex,trace);finalstate(c)==timed||error("replaychangedtimedfinal")
  append!(out,rawindex);push!(out,length(trace));append!(out,trace)
  Array(c.lengths.data)==c.L&&Array(c.offsets.data)==c.offsetinput||error("readonlyinputs");guards!(out,c)
 end
 append!(prefix,out);seconds,prefix
end
function main()
 cases=readcases();checker=Libdl.dlsym(Libdl.dlopen(joinpath(@__DIR__,"study_check.so")),:study_check);runstudy(cases)
 for trial in 1:5
  sec,v=runstudy(cases);path="trial$(trial).bin";write(path,v);metric=zeros(Cdouble,2)
  ccall(checker,Cint,(Cstring,Ptr{Cdouble}),path,metric)==0||error("independentintegerpipelineFAIL")
  @printf("\nSTUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":true,\"checked_values\":%d,\"max_abs_error\":0,\"max_rel_error\":null,\"timing_region\":\"preallocated_full_pipeline_transfers_hostsort_repclustering_sum4cases\",\"jit_excluded\":true,\"internal_repeats\":1,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"int64\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,sec,length(v),path,length(v))
 end
end

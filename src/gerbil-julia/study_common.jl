const CHECKLIB=joinpath(@__DIR__,"study_check.so")
mutable struct GState{B,C}
 k::Int;ip::Int;n::Int;entries::Int;threshold::Int;unique::Int
 packed::Vector{UInt32}
 table::B;failed::B;echo::B;keys::Vector{B};nf::C
 streams::Vector{CUDA.CuStream};stream_id::Int
end
function galloc(::Type{T},n;guards=32÷sizeof(T))where T
 prefix=256÷sizeof(T);storage=CuArray{T}(undef,prefix+n+guards)
 CUDA.memset(convert(CUDA.CuPtr{UInt8},pointer(storage)),UInt8(165),length(storage)*sizeof(T))
 (storage=storage,data=view(storage,prefix+1:prefix+n),prefix=prefix,n=n,guards=guards)
end
function logical_reset(a,value)
 CUDA.memset(convert(CUDA.CuPtr{UInt8},pointer(a.data)),UInt8(value),a.n*sizeof(eltype(a.storage)))
end
function read_guarded(a)
 Array(view(a.storage,a.prefix-a.guards+1:a.prefix+a.n+a.guards))
end
function reset!(s)
 logical_reset(s.table,0);logical_reset(s.failed,165);logical_reset(s.echo,165)
 for k in s.keys;logical_reset(k,165);end
 logical_reset(s.nf,0);s.stream_id=1;CUDA.device_synchronize();nothing
end
function states()
 nc=ccall((:gerbil_ncases,CHECKLIB),Cuint,());result=[]
 for i in 0:Int(nc)-1
  values=[Int(ccall((:gerbil_field,CHECKLIB),Cuint,(Cuint,Cuint),i,j)) for j in 0:5]
  k,ip,n,entries,threshold,unique=values
  ptr=ccall((:gerbil_keys,CHECKLIB),Ptr{UInt32},(Cuint,),i);packed=copy(unsafe_wrap(Vector{UInt32},ptr,n*ip))
  table=galloc(UInt32,entries*(ip+1));failed=galloc(UInt32,n*ip);echo=galloc(UInt32,n*ip);keys=[galloc(UInt32,65536) for _ in 1:4];nf=galloc(UInt64,1;guards=1);streams=[CUDA.CuStream() for _ in 1:4]
  push!(result,GState(k,ip,n,entries,threshold,unique,packed,table,failed,echo,keys,nf,streams,1))
 end
 result
end
function dispatch(s,stream,n)
 if s.ip==1;launch_add(s,stream,n,Val(1))
 elseif s.ip==2;launch_add(s,stream,n,Val(2))
 elseif s.ip==4;launch_add(s,stream,n,Val(4))
 elseif s.ip==6;launch_add(s,stream,n,Val(6))
 elseif s.ip==10;launch_add(s,stream,n,Val(10))
 else;error("unsupported harness width");end
 nothing
end
function canonical_table(table,ip,entries)
 ipe=ip+1;occupied=Int[];empty=Int[]
 for i in 1:entries
  if table[i*ipe]!=0
   table[i*ipe]!=typemax(UInt32)||error("locked table counter remains");push!(occupied,i)
  else
   all(iszero,view(table,(i-1)*ipe+1:i*ipe))||error("discarded nonzero empty payload");push!(empty,i)
  end
 end
 key(i)=Tuple(view(table,(i-1)*ipe+1:(i-1)*ipe+ip))
 sort!(occupied;by=key)
 for i in 2:length(occupied);key(occupied[i])!=key(occupied[i-1])||error("duplicate occupied key");end
 permutation=vcat(occupied,empty);sort(permutation)==collect(1:entries)||error("not full bijection")
 canonical=UInt32[];for i in permutation;append!(canonical,view(table,(i-1)*ipe+1:i*ipe));end
 canonical,UInt32.(permutation.-1)
end
function run_case(s,trial,save)
 reset!(s);initial=read_guarded(s.table);initialfailed=read_guarded(s.failed);initialcounter=read_guarded(s.nf)
 prefix=UInt64[];append!(prefix,view(initial,9:length(initial)-8));append!(prefix,view(initialfailed,9:length(initialfailed)-8));push!(prefix,initialcounter[2])
 table=Vector{UInt32}(undef,s.entries*(s.ip+1));hist=zeros(UInt32,512);selected=UInt32[];nfref=Ref{UInt64}(0)
 unique=UInt64(0);total=UInt64(0);below=UInt64(0);capacity=65536÷s.ip;bundle=0;packed=s.packed
 CUDA.device_synchronize();start=time_ns()
 GC.@preserve packed begin
  for offset in 0:capacity:s.n-1
   count=min(capacity,s.n-offset);sid=bundle%4+1;s.stream_id=sid;stream=s.streams[sid]
   CUDA.stream!(stream) do
    unsafe_copyto!(pointer(s.keys[sid].data),pointer(packed,offset*s.ip+1),count*s.ip;stream=stream,async=true)
    unsafe_copyto!(pointer(s.echo.data,offset*s.ip+1),pointer(s.keys[sid].data),count*s.ip;stream=stream,async=true)
    dispatch(s,stream,count)
   end
   bundle+=1
  end
  CUDA.device_synchronize()
 end
 GC.@preserve table nfref begin
  unsafe_copyto!(pointer(table),pointer(s.table.data),length(table);async=false)
  unsafe_copyto!(Base.unsafe_convert(Ptr{UInt64},nfref),pointer(s.nf.data),1;async=false)
 end
 for i in 1:s.entries
  count=table[i*(s.ip+1)]
  if count!=0
   unique+=1;total+=count;hist[count<512 ? Int(count)+1 : 1]+=1
   if count>=s.threshold;append!(selected,view(table,(i-1)*(s.ip+1)+1:i*(s.ip+1)));else;below+=1;end
  end
 end
 seconds=(time_ns()-start)*1e-9
 rawtable=read_guarded(s.table);failed=read_guarded(s.failed);echo=read_guarded(s.echo);keys=[read_guarded(k) for k in s.keys];nf=read_guarded(s.nf)
 append!(prefix,view(echo,9:length(echo)-8))
 raw=UInt64[s.k,s.ip,s.n,s.entries,65536,bundle];append!(raw,rawtable)
 for a in keys;append!(raw,a);end
 append!(raw,failed);append!(raw,echo);append!(raw,nf)
 rawpath=joinpath(ARGS[1],"$(MODEL)_$(getpid())_trial$(trial)_k$(s.k).bin")
 if save
  open(io->write(io,raw),rawpath,"w")
  println("STUDY_RAW_ARTIFACT {\"trial\":",trial,",\"k\":",s.k,",\"path\":\"",rawpath,"\",\"dtype\":\"uint64\",\"count\":",length(raw),"}")
 end
 canonical,permutation=canonical_table(table,s.ip,s.entries);out=UInt64[];append!(out,canonical);append!(out,hist);push!(out,length(selected)÷(s.ip+1))
 rows=[Tuple(view(selected,i:i+s.ip)) for i in 1:s.ip+1:length(selected)];sort!(rows)
 for row in rows;append!(out,row);end
 append!(out,[unique,total,below]);append!(out,view(failed,9:length(failed)-8));push!(out,nf[2])
 for a in [rawtable,keys...,failed,echo];append!(out,view(a,1:8));append!(out,view(a,length(a)-7:length(a)));end
 push!(out,nf[1]);push!(out,nf[3])
 save&&open(io->write(io,permutation),rawpath*".permutation.u32","w")
 (prefix=prefix,output=out,seconds=seconds)
end
function main()
 length(ARGS)==1||error("persistent raw evidence directory required");mkpath(ARGS[1]);ss=states()
 for s in ss;run_case(s,0,false);end
 for trial in 1:5
  prefix=UInt64[];outputs=UInt64[];seconds=0.
  for s in ss
   r=run_case(s,trial,true);append!(prefix,r.prefix);append!(outputs,r.output);seconds+=r.seconds
  end
  append!(prefix,outputs);path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"trial$(trial).bin");open(io->write(io,prefix),path,"w")
  ok=ccall((:gerbil_validate,CHECKLIB),Cint,(Ptr{UInt64},UInt64),prefix,length(prefix))==1
  println("STUDY_TRIAL {\"trial\":",trial,",\"time_seconds\":",seconds,",\"correct\":",ok,",\"checked_values\":",length(prefix),",\"max_abs_error\":",ok ? "0" : "null",",\"max_rel_error\":",ok ? "0" : "null",",\"timing_region\":\"sum5_preallocated_active_GPU_counting_upload_echo_insert_fulltable_extract_histogram_threshold\",\"jit_excluded\":true,\"internal_repeats\":1,\"output_artifact\":{\"path\":\"",path,"\",\"dtype\":\"uint64\",\"count\":",length(prefix),",\"byte_order\":\"little\"}}")
  ok||error("complete Gerbil counting contract failed")
 end
 for s in ss
  CUDA.device_synchronize();for stream in s.streams;Base.finalize(stream);end
  for a in [s.table,s.failed,s.echo,s.keys...];CUDA.unsafe_free!(a.storage);end;CUDA.unsafe_free!(s.nf.storage)
 end
end

using Libdl
const ROOT=normpath(@__DIR__)
const CHECKLIB=Libdl.dlopen(joinpath(pwd(),"studycheck.so"))
const CHECKPTR=Libdl.dlsym(CHECKLIB,:study_check)
function guarded(::Type{T},n) where T
 raw=CuArray{UInt8}(undef,n*sizeof(T)+512);fill!(raw,UInt8(0xa5));d=unsafe_wrap(CuArray,convert(CuPtr{T},pointer(raw)+256),n;own=false)
 (raw=raw,d=d,n=n)
end
function bytefill!(a,v)
 CUDA.memset(convert(CuPtr{UInt8},pointer(a.d)),UInt8(v),sizeof(eltype(a.d))*a.n)
 nothing
end
function study()
 CUDA.allowscalar(false);blocks=2*Int(CUDA.attribute(CUDA.device(),CUDA.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT))
 names=("original_first8192_plus_degree_coverage","degree_tail_duplicate_collinear_controls")
 rawdir=normpath(joinpath(ROOT,"../../.porting-state/portable_julia/rsmt_primary",MODEL));mkpath(rawdir);capture_id=string(time_ns())
 for trial in 0:5
  prefix=Int32[];out=Int32[];seconds=0.;rawfile=trial>0 ? open(joinpath(rawdir,"$(capture_id)_trial$(trial).physical.bin"),"w") : nothing
  trial>0&&println("RSMT_RAW trial=",trial," path=",joinpath(rawdir,"$(capture_id)_trial$(trial).physical.bin"))
  for (ci,name) in enumerate(names)
   fixture=collect(reinterpret(Int32,read(joinpath(ROOT,"data",name*".bin"))));nets=Int(fixture[1]);pins=Int(fixture[2]);length(fixture)==nets+3+2pins||error("fixture length")
   c=(nets=nets,pins=pins,idx=guarded(Int32,nets+1),xin=guarded(Int32,pins),yin=guarded(Int32,pins),oi=guarded(Int32,nets+1),x=guarded(Int32,2pins),y=guarded(Int32,2pins),edge=guarded(Int16,4pins),wl=guarded(Int32,nets),state=guarded(Int32,3))
   # Two views are the same interleaved Int16 edge buffer, preserving native4-byte stride.
   c=merge(c,(es=c.edge.d,ed=c.edge.d))
   copyto!(c.idx.d,fixture[3:3+nets]);copyto!(c.xin.d,fixture[4+nets:3+nets+pins]);copyto!(c.yin.d,fixture[4+nets+pins:end]);bytefill!(c.oi,0xd9);bytefill!(c.wl,0xd9);fill!(c.state.d,Int32(0));CUDA.device_synchronize()
   push!(prefix,Int32(nets),Int32(pins));append!(prefix,Array(c.idx.d));append!(prefix,Array(c.xin.d));append!(prefix,Array(c.yin.d))
   start=time_ns();bytefill!(c.x,0xff);bytefill!(c.y,0xff);bytefill!(c.edge,0);launch!(c,blocks);CUDA.device_synchronize();seconds+=(time_ns()-start)*1e-9
   vcat(Array(c.idx.d),Array(c.xin.d),Array(c.yin.d))==fixture[3:end]||error("readonly input")
   append!(out,Array(c.oi.d));append!(out,Array(c.x.d));append!(out,Array(c.y.d));edges=Array(c.edge.d);append!(out,Int32.(edges[1:2:end]));append!(out,Int32.(edges[2:2:end]));state=Array(c.state.d);count=Int(state[3]);0<=count<=nets||error("worklist bounds");wh=Array(c.wl.d);push!(out,Int32(count));append!(out,sort(wh[1:count]));all(==(reinterpret(Int32,UInt32(0xd9d9d9d9))),wh[count+1:end])||error("unused worklist")
   for a in (c.idx,c.xin,c.yin,c.oi,c.x,c.y,c.edge,c.wl,c.state)
    bytes=Array(a.raw);all(==(0xa5),bytes[1:256])&&all(==(0xa5),bytes[end-255:end])||error("actual guard bytes")
    if trial>0;write(rawfile,Int32(length(bytes)));write(rawfile,bytes);end
    if a!==c.state;append!(out,reinterpret(Int32,bytes[1:256]));append!(out,reinterpret(Int32,bytes[end-255:end]));end
   end
   trial>0&&println("RSMT_CASE trial=",trial," case=",ci-1," nets=",nets," pins=",pins," large_worklist=",count," blocks=",blocks," actual_currpos1=",state[1]," actual_currpos2=",state[2])
   for a in (c.idx,c.xin,c.yin,c.oi,c.x,c.y,c.edge,c.wl,c.state);CUDA.unsafe_free!(a.d);CUDA.unsafe_free!(a.raw);end
  end
  if trial>0
   close(rawfile);append!(prefix,out);path="trial$(trial).bin";open(io->write(io,prefix),path,"w");metrics=zeros(Float64,2);ccall(CHECKPTR,Cint,(Cstring,Ptr{Float64}),path,metrics)==0||error("independent complete integer oracle")
   println("STUDY_TRIAL {\"trial\":",trial,",\"time_seconds\":",seconds,",\"correct\":true,\"checked_values\":",length(prefix),",\"max_abs_error\":0,\"max_rel_error\":0,\"jit_excluded\":true,\"timing_region\":\"sum2case_source3memsets_2SteinerKernels_contextcomplete\",\"internal_repeats\":1,\"output_artifact\":{\"path\":\"",path,"\",\"dtype\":\"int32\",\"count\":",length(prefix),",\"byte_order\":\"little\"}}")
  end
 end
end

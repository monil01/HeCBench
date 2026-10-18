using SHA,Libdl,Printf
const BASE=dirname(@__FILE__);const G=16;const SENT=UInt32(0x5a5a5a5a);const POISON=UInt32(0x13579bdf);const REPEAT=1000
const LIBPATH="/usr/lib/x86_64-linux-gnu/libcusparse.so.12.0.1.140";const LIBHASH="d4d0efdb21bcc1bd85ae42bec5744350207b22042ab62bc072d355b80e0d5298";const LIB=Libdl.dlopen(LIBPATH)
for name in [:cusparseCreate,:cusparseDestroy,:cusparseCreateMatDescr,:cusparseDestroyMatDescr,:cusparseSetMatType,:cusparseSetMatIndexBase,:cusparseGetVersion,:cusparseGetPointerMode,:cusparseGetStream,:cusparseXcsrsort_bufferSizeExt,:cusparseXcsrsort,:cusparseGather,:cusparseCreateIdentityPermutation,:cusparseCreateSpVec,:cusparseCreateDnVec,:cusparseDestroySpVec,:cusparseDestroyDnVec]
 @eval const $(name)=Libdl.dlsym(LIB,$(QuoteNode(name)))
end
include("study_cases.jl")
check(x)=x==0||error("cuSPARSE status $x")
ptr(x)=reinterpret(CUDA.CuPtr{Cvoid},pointer(x))
function readwords(name,n)
 b=read(joinpath(BASE,name));length(b)==4*n||error("Unexpectedfixture $name");collect(reinterpret(UInt32,b))
end
function identity!(s)
 check(ccall(cusparseCreateIdentityPermutation,Cint,(Ptr{Cvoid},Cint,CUDA.CuPtr{Cvoid}),s.h,s.case.z,ptr(s.dev[4])))
end
function batch!(s)
 c=s.case
 GC.@preserve s begin
  for rep=1:REPEAT
   check(ccall(cusparseXcsrsort,Cint,(Ptr{Cvoid},Cint,Cint,Cint,Ptr{Cvoid},CUDA.CuPtr{Cvoid},CUDA.CuPtr{Cvoid},CUDA.CuPtr{Cvoid},CUDA.CuPtr{Cvoid}),s.h,c.m,c.n,c.z,s.d,ptr(s.dev[1]),ptr(s.dev[2]),ptr(s.dev[4]),ptr(s.buf)))
   check(ccall(cusparseGather,Cint,(Ptr{Cvoid},Ptr{Cvoid},Ptr{Cvoid}),s.h,s.dn,s.sp))
  end
 end
 CUDA.device_synchronize(blocking=true)
end
function spsort_study(makewords)
 bytes2hex(sha256(read(LIBPATH)))==LIBHASH||error("libraryhash");CUDA.device!(0);h=Ref{Ptr{Cvoid}}();d=Ref{Ptr{Cvoid}}();check(ccall(cusparseCreate,Cint,(Ref{Ptr{Cvoid}},),h));check(ccall(cusparseCreateMatDescr,Cint,(Ref{Ptr{Cvoid}},),d));check(ccall(cusparseSetMatType,Cint,(Ptr{Cvoid},Cint),d[],0));check(ccall(cusparseSetMatIndexBase,Cint,(Ptr{Cvoid},Cint),d[],0));version=Ref{Cint}();pm=Ref{Cint}();stream=Ref{Ptr{Cvoid}}();check(ccall(cusparseGetVersion,Cint,(Ptr{Cvoid},Ref{Cint}),h[],version));check(ccall(cusparseGetPointerMode,Cint,(Ptr{Cvoid},Ref{Cint}),h[],pm));check(ccall(cusparseGetStream,Cint,(Ptr{Cvoid},Ref{Ptr{Cvoid}}),h[],stream));version[]==12001&&pm[]==0&&stream[]==C_NULL||error("vendorpolicy")
 for(name,fn)in [(:cusparseXcsrsort,cusparseXcsrsort),(:cusparseGather,cusparseGather)]
  info=Ref{NTuple{4,Ptr{Cvoid}}}();ccall(:dladdr,Cint,(Ptr{Cvoid},Ref{NTuple{4,Ptr{Cvoid}}}),fn,info)!=0||error("dladdr");resolved=realpath(unsafe_string(Ptr{UInt8}(info[][1])));resolved==LIBPATH||error("DSO");println("STUDY_VENDOR function=",name," path=",resolved," sha256=",LIBHASH," version=",version[]," pointer_mode=host stream=legacy0")
 end
 input=readwords("input_u32.bin",329771);oracle=readwords("oracle_u32.bin",484485);states=Any[]
 for c in CASES
  pos=c.io;host=Vector{UInt32}[]
  for n in (c.m+1,c.z,c.z);push!(host,vcat(input[pos+1:pos+n],fill(SENT,G)));pos+=n;end
  push!(host,vcat(fill(POISON,c.z),fill(SENT,G)));push!(host,vcat(fill(POISON,c.z),fill(SENT,G)));dev=map(makewords,host);sz=Ref{Csize_t}();check(ccall(cusparseXcsrsort_bufferSizeExt,Cint,(Ptr{Cvoid},Cint,Cint,Cint,CUDA.CuPtr{Cvoid},CUDA.CuPtr{Cvoid},Ref{Csize_t}),h[],c.m,c.n,c.z,ptr(dev[1]),ptr(dev[2]),sz));buf=CuArray{UInt8}(undef,Int(sz[]));sp=Ref{Ptr{Cvoid}}();dn=Ref{Ptr{Cvoid}}();check(ccall(cusparseCreateSpVec,Cint,(Ref{Ptr{Cvoid}},Int64,Int64,CUDA.CuPtr{Cvoid},CUDA.CuPtr{Cvoid},Cint,Cint,Cint),sp,c.z,c.z,ptr(dev[4]),ptr(dev[5]),2,0,0));check(ccall(cusparseCreateDnVec,Cint,(Ref{Ptr{Cvoid}},Int64,CUDA.CuPtr{Cvoid},Cint),dn,c.z,ptr(dev[3]),0));s=(case=c,host=host,dev=dev,buf=buf,sp=sp[],dn=dn[],h=h[],d=d[]);CUDA.device_synchronize(blocking=true);identity!(s);CUDA.device_synchronize(blocking=true);batch!(s);println("STUDY_WORKSPACE case=",length(states)," bytes=",sz[]);push!(states,s)
 end
 for trial=1:5
  prefix=UInt32[];values=UInt32[];guards=UInt32[];times=Float64[];ok=true;mismatches=0
  for(ci,s)in enumerate(states)
   c=s.case;copyto!(s.dev[2],s.host[2]);copyto!(s.dev[5],s.host[5]);CUDA.device_synchronize(blocking=true);identity!(s);CUDA.device_synchronize(blocking=true)
   for i=1:4;raw=Array(s.dev[i]);append!(prefix,raw[1:(i==1 ? c.m+1 : c.z)]);end;append!(prefix,UInt32.([c.m,c.n,c.z,REPEAT,32,0,0,0,0,1,5,G,ci-1,1,0,0]))
   start=time_ns();batch!(s);push!(times,(time_ns()-start)*1e-9/REPEAT);raw=map(Array,s.dev);ok &= raw[1]==s.host[1]&&raw[3]==s.host[3];record=vcat(raw[1][1:c.m+1],raw[2][1:c.z],raw[4][1:c.z],raw[5][1:c.z]);expected=oracle[c.oo+1:c.oo+length(record)];mismatches+=sum(record.!=expected);ok &= record==expected;append!(values,record)
   for a in raw;g=a[end-G+1:end];append!(guards,g);ok &= all(==(SENT),g);end
  end
  art=vcat(prefix,values,guards);path=joinpath(ENV["STUDY_OUTPUT_DIR"],"trial$(trial).bin");write(path,reinterpret(UInt8,art))
  @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":%d,\"max_rel_error\":null,\"error_metric_scope\":\"Exactrowptr/column/permutation/valuebitmismatchflag\",\"mismatch_output_words\":%d,\"timing_region\":\"sum3cases_meanR1000_stateful_cusparseXcsrsort_plusGather_device_complete\",\"jit_excluded\":true,\"internal_repeats\":1000,\"component_time_seconds\":[%s],\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"uint32\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,sum(times),ok ? "true" : "false",length(values)+length(guards),mismatches>0 ? 1 : 0,mismatches,join(times,","),path,length(art));flush(stdout);ok||error("CSRsort/P/gatherfullstatefailed")
 end
 for s in states;check(ccall(cusparseDestroySpVec,Cint,(Ptr{Cvoid},),s.sp));check(ccall(cusparseDestroyDnVec,Cint,(Ptr{Cvoid},),s.dn));end;check(ccall(cusparseDestroyMatDescr,Cint,(Ptr{Cvoid},),d[]));check(ccall(cusparseDestroy,Cint,(Ptr{Cvoid},),h[]))
end

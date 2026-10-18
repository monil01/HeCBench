
using SHA,Libdl,Printf
const LIBPATH="/usr/lib/x86_64-linux-gnu/libcusparse.so.12.0.1.140";const LIBSHA="d4d0efdb21bcc1bd85ae42bec5744350207b22042ab62bc072d355b80e0d5298";const G=16;const GUARD=UInt32(0xa5a55a5a);const POISON=UInt32(0x13579bdf);const R=10
function readwords(name)
 b=read(joinpath(@__DIR__,name));length(b)%4==0||error("fixturedtype");collect(reinterpret(UInt32,b))
end
function run_study()
 bytes2hex(sha256(read(LIBPATH)))==LIBSHA||error("libraryhash")
 lib=Libdl.dlopen(LIBPATH);sym(n)=Libdl.dlsym(lib,n)
 create=sym(:cusparseCreate);destroy=sym(:cusparseDestroy);created=sym(:cusparseCreateMatDescr);destroyd=sym(:cusparseDestroyMatDescr);settype=sym(:cusparseSetMatType);setbase=sym(:cusparseSetMatIndexBase);getversion=sym(:cusparseGetVersion);getpm=sym(:cusparseGetPointerMode);getstream=sym(:cusparseGetStream);nnz=sym(:cusparseSnnz)
 check(x)=x==0||error("cuSPARSE status $x")
 h=Ref{Ptr{Cvoid}}();desc=Ref{Ptr{Cvoid}}();check(ccall(create,Cint,(Ref{Ptr{Cvoid}},),h));check(ccall(created,Cint,(Ref{Ptr{Cvoid}},),desc));check(ccall(settype,Cint,(Ptr{Cvoid},Cint),desc[],0));check(ccall(setbase,Cint,(Ptr{Cvoid},Cint),desc[],0))
 version=Ref{Cint}();pm=Ref{UInt32}();stream=Ref{Ptr{Cvoid}}();check(ccall(getversion,Cint,(Ptr{Cvoid},Ref{Cint}),h[],version));check(ccall(getpm,Cint,(Ptr{Cvoid},Ref{UInt32}),h[],pm));check(ccall(getstream,Cint,(Ptr{Cvoid},Ref{Ptr{Cvoid}}),h[],stream));pm[]==0&&stream[]==C_NULL||error("vendorpolicy")
 # dladdr ties the actual called Snnz function address to its loaded DSO.
 info=Ref{NTuple{4,Ptr{Cvoid}}}();ccall(:dladdr,Cint,(Ptr{Cvoid},Ref{NTuple{4,Ptr{Cvoid}}}),nnz,info)!=0||error("dladdr");resolved=realpath(unsafe_string(Ptr{UInt8}(info[][1])));resolved==realpath(LIBPATH)||error("functionresolution")
 println("STUDY_VENDOR function=cusparseSnnz path=",resolved," sha256=",LIBSHA," version=",version[]," pointer_mode=",pm[]," stream=0")
 input=readwords("input_bits_u32.bin");oracle=readwords("oracle_counts_u32.bin");cases=[(n=1,mode=0,input_offset=0,oracle_offset=0,total=0,),(n=17,mode=1,input_offset=1,oracle_offset=1,total=289,),(n=33,mode=2,input_offset=290,oracle_offset=18,total=58,),(n=257,mode=3,input_offset=1379,oracle_offset=51,total=440,),(n=1024,mode=4,input_offset=67428,oracle_offset=308,total=3072,),(n=2048,mode=5,input_offset=1116004,oracle_offset=1332,total=124586,)];prefix=UInt32[];states=Any[]
 function callnnz(s,total)
  GC.@preserve s total begin
   check(ccall(nnz,Cint,(Ptr{Cvoid},Cint,Cint,Cint,Ptr{Cvoid},CUDA.CuPtr{Cfloat},Cint,CUDA.CuPtr{Cint},Ref{Cint}),h[],1,s.n,s.n,desc[],reinterpret(CUDA.CuPtr{Cfloat},pointer(s.da,G+1)),s.n,reinterpret(CUDA.CuPtr{Cint},pointer(s.dy,G+1)),total))
  end
 end
 for c in cases
  n=Int(c.n);io=Int(c.input_offset);oo=Int(c.oracle_offset);a=vcat(fill(GUARD,G),input[io+1:io+n*n],fill(GUARD,G));y=vcat(fill(GUARD,G),fill(POISON,n),fill(GUARD,G));s=(n=n,oo=oo,total=Int(c.total),a=a,y=y,da=CuArray(a),dy=CuArray(y));append!(prefix,Array(s.da));append!(prefix,Array(s.dy));CUDA.device_synchronize();callnnz(s,Ref{Cint}(-1));CUDA.device_synchronize();push!(states,s)
 end
 for trial in 1:5
  art=copy(prefix);seconds=0.;ok=true
  for s in states
   copyto!(s.dy,s.y);CUDA.device_synchronize();total=Ref{Cint}(-1);start=time_ns();for _ in 1:R;callnnz(s,total);end;CUDA.device_synchronize();seconds+=(time_ns()-start)*1e-9/R
   a=Array(s.da);y=Array(s.dy);ok&=a==s.a&&total[]==s.total&&y==vcat(fill(GUARD,G),oracle[s.oo+1:s.oo+s.n],fill(GUARD,G));append!(art,a);append!(art,y);push!(art,UInt32(total[]))
  end
  path="spnnz_trial$trial.bin";open(path,"w")do io;write(io,art);end
  @printf("STUDY_TRIAL {\"trial\":%d,\"correct\":%s,\"jit_excluded\":true,\"time_seconds\":%.12g,\"internal_repeats\":10,\"timing_region\":\"sum_six_legacy_cusparseSnnz_batches_host_total_and_device_completion_per_repeat\",\"max_abs_error\":0,\"max_rel_error\":0,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"uint32\",\"count\":%d,\"shape\":[%d],\"byte_order\":\"little\"}}\n",trial,string(ok),seconds,path,length(art),length(art))
 end
 check(ccall(destroyd,Cint,(Ptr{Cvoid},),desc[]));check(ccall(destroy,Cint,(Ptr{Cvoid},),h[]))
end

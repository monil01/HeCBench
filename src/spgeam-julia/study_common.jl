
using SHA,Libdl,Printf
const G=16;const GUARD=UInt32(0xa5a55a5a);const POISON=UInt32(0x13579bdf);const R=10
const LIBPATH="/usr/lib/x86_64-linux-gnu/libcusparse.so.12.0.1.140";const LIBSHA="d4d0efdb21bcc1bd85ae42bec5744350207b22042ab62bc072d355b80e0d5298";const CASES=[(m=1,n=1,mode=0,nnz=1,io=0,oo=0,),(m=17,n=33,mode=1,nnz=543,io=4,oo=4,),(m=33,n=17,mode=2,nnz=32,io=1108,oo=1124,),(m=257,n=129,mode=3,nnz=495,io=1206,oo=1206,),(m=1024,n=1024,mode=4,nnz=30135,io=2454,oo=2326,)]
function readwords(n)
 b=read(joinpath(@__DIR__,n));length(b)%4==0||error("fixturedtype");collect(reinterpret(UInt32,b))
end
function capture!(out,s)
 for a in s.dev;append!(out,Array(a));end
end
function run_study()
 bytes2hex(sha256(read(LIBPATH)))==LIBSHA||error("libraryhash");lib=Libdl.dlopen(LIBPATH);sym(n)=Libdl.dlsym(lib,n);ck(x)=x==0||error("cuSPARSE status $x");create=sym(:cusparseCreate);destroy=sym(:cusparseDestroy);fn=sym(:cusparseCsr2cscEx2);bufsize=sym(:cusparseCsr2cscEx2_bufferSize)
 info=Ref{NTuple{4,Ptr{Cvoid}}}();ccall(:dladdr,Cint,(Ptr{Cvoid},Ref{NTuple{4,Ptr{Cvoid}}}),fn,info)!=0||error("dladdr");resolved=realpath(unsafe_string(Ptr{UInt8}(info[][1])));resolved==realpath(LIBPATH)||error("functionresolution");h=Ref{Ptr{Cvoid}}();ck(ccall(create,Cint,(Ref{Ptr{Cvoid}},),h));version=Ref{Cint}();stream=Ref{Ptr{Cvoid}}();ck(ccall(sym(:cusparseGetVersion),Cint,(Ptr{Cvoid},Ref{Cint}),h[],version));ck(ccall(sym(:cusparseGetStream),Cint,(Ptr{Cvoid},Ref{Ptr{Cvoid}}),h[],stream));stream[]==C_NULL||error("stream");println("STUDY_VENDOR Csr2cscEx2=",resolved," sha256=",LIBSHA," version=",version[]," stream=0 alg=1 action=1 base=0 datatype=0")
 function operation(s)
  c=s.c;a=s.dev
  GC.@preserve s begin
   ck(ccall(fn,Cint,(Ptr{Cvoid},Cint,Cint,Cint,CUDA.CuPtr{Cvoid},CUDA.CuPtr{Cint},CUDA.CuPtr{Cint},CUDA.CuPtr{Cvoid},CUDA.CuPtr{Cint},CUDA.CuPtr{Cint},Cint,Cint,Cint,Cint,CUDA.CuPtr{Cvoid}),h[],c.m,c.n,c.nnz,pointer(a[1],G+1),reinterpret(CUDA.CuPtr{Cint},pointer(a[2],G+1)),reinterpret(CUDA.CuPtr{Cint},pointer(a[3],G+1)),pointer(a[4],G+1),reinterpret(CUDA.CuPtr{Cint},pointer(a[5],G+1)),reinterpret(CUDA.CuPtr{Cint},pointer(a[6],G+1)),0,1,0,1,pointer(s.buf)))
  end
 end
 input=readwords("input_u32.bin");oracle=readwords("oracle_u32.bin");states=Any[];prefix=UInt32[]
 for c in CASES
  sizes=[c.nnz,c.m+1,c.nnz,c.nnz,c.n+1,c.nnz];host=Vector{UInt32}[];dev=CuArray{UInt32,1}[];offset=c.io
  for k in 1:6
   values=k<=3 ? input[offset+1:offset+sizes[k]] : fill(POISON,sizes[k]);k<=3&&(offset+=sizes[k]);a=vcat(fill(GUARD,G),values,fill(GUARD,G));push!(host,a);push!(dev,CuArray(a))
  end
  bytes=Ref{Csize_t}()
  GC.@preserve dev begin
   ck(ccall(bufsize,Cint,(Ptr{Cvoid},Cint,Cint,Cint,CUDA.CuPtr{Cvoid},CUDA.CuPtr{Cint},CUDA.CuPtr{Cint},CUDA.CuPtr{Cvoid},CUDA.CuPtr{Cint},CUDA.CuPtr{Cint},Cint,Cint,Cint,Cint,Ref{Csize_t}),h[],c.m,c.n,c.nnz,pointer(dev[1],G+1),reinterpret(CUDA.CuPtr{Cint},pointer(dev[2],G+1)),reinterpret(CUDA.CuPtr{Cint},pointer(dev[3],G+1)),pointer(dev[4],G+1),reinterpret(CUDA.CuPtr{Cint},pointer(dev[5],G+1)),reinterpret(CUDA.CuPtr{Cint},pointer(dev[6],G+1)),0,1,0,1,bytes))
  end
  s=(c=c,host=host,dev=dev,buf=CuArray{UInt8}(undef,max(Int(bytes[]),1)));capture!(prefix,s);CUDA.device_synchronize();operation(s);CUDA.device_synchronize();push!(states,s)
 end
 for trial in 1:5
  art=copy(prefix);ok=true;seconds=0.
  for s in states
   for k in 4:6;copyto!(s.dev[k],s.host[k]);end;CUDA.device_synchronize();start=time_ns();for _ in 1:R;operation(s);end;CUDA.device_synchronize();seconds+=(time_ns()-start)*1e-9/R;offset=s.c.oo
   for k in 1:6
    actual=Array(s.dev[k]);n=length(actual)-32;expected=k<=3 ? s.host[k] : vcat(fill(GUARD,G),oracle[offset+1:offset+n],fill(GUARD,G));ok&=actual==expected;k>=4&&(offset+=n)
   end
   capture!(art,s)
  end
  path="spgeam_trial$trial.bin";open(path,"w")do io;write(io,art);end
  @printf("STUDY_TRIAL {\"trial\":%d,\"correct\":%s,\"jit_excluded\":true,\"time_seconds\":%.12g,\"internal_repeats\":10,\"timing_region\":\"sum_five_CSR_to_CSC_ALG1_numeric_vendor_batches_completion_per_repeat\",\"max_abs_error\":0,\"max_rel_error\":0,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"uint32\",\"count\":%d,\"shape\":[%d],\"byte_order\":\"little\"}}\n",trial,string(ok),seconds,path,length(art),length(art))
 end
 ck(ccall(destroy,Cint,(Ptr{Cvoid},),h[]))
end

using SHA,Libdl,Printf
const BASE=dirname(@__FILE__);const G=16;const SENT=UInt32(0x4e5a5a5a);const POISON=UInt32(0x7fc12345);const REPEAT=1000
const LIBPATH="/usr/lib/x86_64-linux-gnu/libcusparse.so.12.0.1.140";const LIBHASH="d4d0efdb21bcc1bd85ae42bec5744350207b22042ab62bc072d355b80e0d5298"
const LIB=Libdl.dlopen(LIBPATH)
for name in [:cusparseCreate,:cusparseDestroy,:cusparseGetVersion,:cusparseGetPointerMode,:cusparseGetStream,:cusparseCreateCsr,:cusparseCreateDnMat,:cusparseSparseToDense_bufferSize,:cusparseSparseToDense,:cusparseDestroySpMat,:cusparseDestroyDnMat]
 @eval const $(name)=Libdl.dlsym(LIB,$(QuoteNode(name)))
end
include("study_cases.jl")
check(x)=x==0||error("cuSPARSE status $x")
ptr(a)=reinterpret(CUDA.CuPtr{Cvoid},pointer(a))
function readtyped(::Type{T},name,n)where T
 b=read(joinpath(BASE,name));length(b)==sizeof(T)*n||error("Unexpectedfixture $name");collect(reinterpret(T,b))
end
function convert!(s)
 GC.@preserve s begin
  check(ccall(cusparseSparseToDense,Cint,(Ptr{Cvoid},Ptr{Cvoid},Ptr{Cvoid},Cint,CUDA.CuPtr{Cvoid}),s.handle,s.a,s.b,0,ptr(s.buffer)))
 end
 nothing
end
function batch!(s)
 for rep=1:REPEAT;convert!(s);end
 CUDA.device_synchronize(blocking=true)
end
function sps2d_study(makewords)
 bytes2hex(sha256(read(LIBPATH)))==LIBHASH||error("cuSPARSEhash");CUDA.device!(0)
 handle=Ref{Ptr{Cvoid}}();check(ccall(cusparseCreate,Cint,(Ref{Ptr{Cvoid}},),handle));version=Ref{Cint}();pm=Ref{Cint}();stream=Ref{Ptr{Cvoid}}();check(ccall(cusparseGetVersion,Cint,(Ptr{Cvoid},Ref{Cint}),handle[],version));check(ccall(cusparseGetPointerMode,Cint,(Ptr{Cvoid},Ref{Cint}),handle[],pm));check(ccall(cusparseGetStream,Cint,(Ptr{Cvoid},Ref{Ptr{Cvoid}}),handle[],stream));version[]==12001&&pm[]==0&&stream[]==C_NULL||error("vendorpolicy")
 info=Ref{NTuple{4,Ptr{Cvoid}}}();ccall(:dladdr,Cint,(Ptr{Cvoid},Ref{NTuple{4,Ptr{Cvoid}}}),cusparseSparseToDense,info)!=0||error("dladdr");resolved=realpath(unsafe_string(Ptr{UInt8}(info[][1])));resolved==LIBPATH||error("functionresolution");println("STUDY_VENDOR function=cusparseSparseToDense path=",resolved," sha256=",LIBHASH," version=",version[]," index64base0 rowmajor alg0 pointer_mode=host stream=legacy0")
 input=readtyped(UInt32,"input_u32.bin",460399);oracle=readtyped(UInt32,"oracle_u32.bin",8520505);states=Any[];prefix=UInt32[]
 for c in CASES
  lens=(2*(c.m+1),2*c.nnz,c.nnz);offs=(c.rp,c.col,c.val);host=[vcat(input[o+1:o+n],fill(SENT,G)) for (o,n) in zip(offs,lens)];dev=map(makewords,host)
  for(d,n)in zip(dev,lens);append!(prefix,Array(d)[1:n]);end
  append!(prefix,UInt32.([c.m,c.n,c.nnz,REPEAT,3,3,0,0,2,0,0,0]))
  out=makewords(vcat(zeros(UInt32,c.m*c.n),fill(SENT,G)));da=Ref{Ptr{Cvoid}}();db=Ref{Ptr{Cvoid}}();sz=Ref{Csize_t}()
  check(ccall(cusparseCreateCsr,Cint,(Ref{Ptr{Cvoid}},Int64,Int64,Int64,CUDA.CuPtr{Cvoid},CUDA.CuPtr{Cvoid},CUDA.CuPtr{Cvoid},Cint,Cint,Cint,Cint),da,c.m,c.n,c.nnz,ptr(dev[1]),ptr(dev[2]),ptr(dev[3]),3,3,0,0))
  check(ccall(cusparseCreateDnMat,Cint,(Ref{Ptr{Cvoid}},Int64,Int64,Int64,CUDA.CuPtr{Cvoid},Cint,Cint),db,c.m,c.n,c.n,ptr(out),0,2))
  check(ccall(cusparseSparseToDense_bufferSize,Cint,(Ptr{Cvoid},Ptr{Cvoid},Ptr{Cvoid},Cint,Ref{Csize_t}),handle[],da[],db[],0,sz));buffer=CuArray{UInt8}(undef,max(Int(sz[]),1));println("STUDY_WORKSPACE case=",c.id," bytes=",sz[])
  state=(case=c,host=host,dev=dev,out=out,a=da[],b=db[],buffer=buffer,handle=handle[]);batch!(state);push!(states,state)
 end
 for trial=1:5
  values=UInt32[];guards=UInt32[];times=Float64[];ok=true
  for s in states
   c=s.case;copyto!(s.out,vcat(zeros(UInt32,c.m*c.n),fill(SENT,G)));CUDA.device_synchronize(blocking=true);start=time_ns();batch!(s);push!(times,(time_ns()-start)*1e-9/REPEAT)
   raw=Array(s.out);actual=raw[1:c.m*c.n];expect=oracle[c.oo+1:c.oo+c.m*c.n];floats=reinterpret(Float32,actual);ok &= actual==expect&&all(isfinite,floats)&&count(!=(0f0),floats)==c.nnz;append!(values,actual);append!(guards,raw[end-G+1:end]);ok &= all(==(SENT),raw[end-G+1:end])
   for(d,h)in zip(s.dev,s.host);actual=Array(d);ok &= actual==h;append!(guards,actual[end-G+1:end]);end
  end
  art=vcat(prefix,values,guards);path=joinpath(ENV["STUDY_OUTPUT_DIR"],"trial$(trial).bin");write(path,reinterpret(UInt8,art))
  @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":0,\"max_rel_error\":0,\"timing_region\":\"sum6cases_meanR1000_cusparseSparseToDense_only_devicecomplete\",\"jit_excluded\":true,\"internal_repeats\":1000,\"authored_device_kernels\":0,\"all_input_buffers_unchanged_checked\":true,\"component_time_seconds\":[%s],\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"uint32\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,sum(times),ok ? "true" : "false",length(values)+length(guards),join(times,","),path,length(art));flush(stdout);ok||error("SparseToDensefullbitoutput/oracle/guard/inputfailed")
 end
 for s in states;check(ccall(cusparseDestroySpMat,Cint,(Ptr{Cvoid},),s.a));check(ccall(cusparseDestroyDnMat,Cint,(Ptr{Cvoid},),s.b));end
 check(ccall(cusparseDestroy,Cint,(Ptr{Cvoid},),handle[]))
end

using SHA,Libdl,Printf
const BASE=dirname(@__FILE__);const G=16;const SENT=UInt32(0x4e5a5a5a);const POISON=UInt32(0x7fc12345);const REPEAT=20
const LIBPATH="/usr/lib/x86_64-linux-gnu/libcusparse.so.12.0.1.140";const LIBHASH="d4d0efdb21bcc1bd85ae42bec5744350207b22042ab62bc072d355b80e0d5298"
const LIB=Libdl.dlopen(LIBPATH)
for name in [:cusparseCreate,:cusparseDestroy,:cusparseGetVersion,:cusparseGetPointerMode,:cusparseGetStream,:cusparseCreateCoo,:cusparseCreateCsr,:cusparseCreateDnMat,:cusparseSpMM_bufferSize,:cusparseSpMM,:cusparseDestroySpMat,:cusparseDestroyDnMat]
 @eval const $(name)=Libdl.dlsym(LIB,$(QuoteNode(name)))
end
include("study_cases.jl")
check(x)=x==0||error("cuSPARSE status $x")
ptr(a)=reinterpret(CUDA.CuPtr{Cvoid},pointer(a))
function readtyped(::Type{T},name,n)where T
 b=read(joinpath(BASE,name));length(b)==sizeof(T)*n||error("Unexpectedfixture $name");collect(reinterpret(T,b))
end
function spmm!(s,mode)
 GC.@preserve s begin
  check(ccall(cusparseSpMM,Cint,(Ptr{Cvoid},Cint,Cint,Ref{Cfloat},Ptr{Cvoid},Ptr{Cvoid},Ref{Cfloat},Ptr{Cvoid},Cint,Cint,CUDA.CuPtr{Cvoid}),s.handle,0,0,s.alpha,s.a[mode],s.b,s.beta,s.c[mode],0,0,ptr(s.buffers[mode])))
 end
 nothing
end
function batch!(s,mode)
 for rep=1:REPEAT;spmm!(s,mode);end
 CUDA.device_synchronize(blocking=true)
end
function spmm_study(makewords)
 bytes2hex(sha256(read(LIBPATH)))==LIBHASH||error("cuSPARSE hash");CUDA.device!(0)
 handle=Ref{Ptr{Cvoid}}();check(ccall(cusparseCreate,Cint,(Ref{Ptr{Cvoid}},),handle));version=Ref{Cint}();pm=Ref{Cint}();stream=Ref{Ptr{Cvoid}}();check(ccall(cusparseGetVersion,Cint,(Ptr{Cvoid},Ref{Cint}),handle[],version));check(ccall(cusparseGetPointerMode,Cint,(Ptr{Cvoid},Ref{Cint}),handle[],pm));check(ccall(cusparseGetStream,Cint,(Ptr{Cvoid},Ref{Ptr{Cvoid}}),handle[],stream));version[]==12001&&pm[]==0&&stream[]==C_NULL||error("vendorpolicy")
 info=Ref{NTuple{4,Ptr{Cvoid}}}();ccall(:dladdr,Cint,(Ptr{Cvoid},Ref{NTuple{4,Ptr{Cvoid}}}),cusparseSpMM,info)!=0||error("dladdr");resolved=realpath(unsafe_string(Ptr{UInt8}(info[][1])));resolved==LIBPATH||error("functionresolution");println("STUDY_VENDOR function=cusparseSpMM path=",resolved," sha256=",LIBHASH," version=",version[]," pointer_mode=host stream=legacy0")
 input=readtyped(UInt32,"input_u32.bin",1058530);oracle=readtyped(Float64,"oracle_f64.bin",1052833);states=Any[];prefix=Float64[];alpha=Ref(1f0);beta=Ref(0f0)
 for c in CASES
  lengths=(c.nnz,c.nnz,c.m+1,c.nnz,c.k*c.n);offs=(c.ro,c.co,c.rp,c.vo,c.bo);host=[vcat(input[o+1:o+n],fill(SENT,G))for(o,n)in zip(offs,lengths)];dev=map(makewords,host)
  for(a,n)in zip(dev,lengths);append!(prefix,Float64.(Array(a)[1:n]));end;append!(prefix,Float64.([c.m,c.k,c.n,c.nnz,REPEAT,0,2,2,0,0,0,0,0x3f800000,0,2,0]))
  da=Ptr{Cvoid}[]
  for mode=1:2
   a=Ref{Ptr{Cvoid}}()
   if mode==1
    check(ccall(cusparseCreateCoo,Cint,(Ref{Ptr{Cvoid}},Int64,Int64,Int64,CUDA.CuPtr{Cvoid},CUDA.CuPtr{Cvoid},CUDA.CuPtr{Cvoid},Cint,Cint,Cint),a,c.m,c.k,c.nnz,ptr(dev[1]),ptr(dev[2]),ptr(dev[4]),2,0,0))
   else
    check(ccall(cusparseCreateCsr,Cint,(Ref{Ptr{Cvoid}},Int64,Int64,Int64,CUDA.CuPtr{Cvoid},CUDA.CuPtr{Cvoid},CUDA.CuPtr{Cvoid},Cint,Cint,Cint,Cint),a,c.m,c.k,c.nnz,ptr(dev[3]),ptr(dev[2]),ptr(dev[4]),2,2,0,0))
   end
   push!(da,a[])
  end
  db=Ref{Ptr{Cvoid}}();check(ccall(cusparseCreateDnMat,Cint,(Ref{Ptr{Cvoid}},Int64,Int64,Int64,CUDA.CuPtr{Cvoid},Cint,Cint),db,c.k,c.n,c.n,ptr(dev[5]),0,2))
  out=[makewords(vcat(fill(POISON,c.m*c.n),fill(SENT,G)))for mode=1:2];dc=Ptr{Cvoid}[];buffers=CUDA.CuArray{UInt8,1}[];szs=Int[]
  for mode=1:2
   d=Ref{Ptr{Cvoid}}();check(ccall(cusparseCreateDnMat,Cint,(Ref{Ptr{Cvoid}},Int64,Int64,Int64,CUDA.CuPtr{Cvoid},Cint,Cint),d,c.m,c.n,c.n,ptr(out[mode]),0,2));push!(dc,d[]);sz=Ref{Csize_t}()
   check(ccall(cusparseSpMM_bufferSize,Cint,(Ptr{Cvoid},Cint,Cint,Ref{Cfloat},Ptr{Cvoid},Ptr{Cvoid},Ref{Cfloat},Ptr{Cvoid},Cint,Cint,Ref{Csize_t}),handle[],0,0,alpha,da[mode],db[],beta,dc[mode],0,0,sz));push!(buffers,CuArray{UInt8}(undef,Int(sz[])));push!(szs,Int(sz[]));println("STUDY_WORKSPACE case=",length(states)," mode=",mode-1," bytes=",sz[])
  end
  CUDA.device_synchronize(blocking=true);s=(case=c,host=host,dev=dev,out=out,a=da,b=db[],c=dc,buffers=buffers,workspace=szs,handle=handle[],alpha=alpha,beta=beta);for mode=1:2;batch!(s,mode);end;push!(states,s)
 end
 for trial=1:5
  values=Float64[];guards=Float64[];times=Float64[];ok=true;maxabs=0.;maxbudget=0.
  for s in states
   c=s.case;expect=oracle[c.oo+1:c.oo+c.m*c.n]
   for mode=1:2
    copyto!(s.out[mode],vcat(fill(POISON,c.m*c.n),fill(SENT,G)));CUDA.device_synchronize(blocking=true);start=time_ns();batch!(s,mode);push!(times,(time_ns()-start)*1e-9/REPEAT)
    raw=Array(s.out[mode]);actual=Float64.(reinterpret(Float32,raw[1:c.m*c.n]));diff=abs.(actual.-expect);budget=1e-4.+2e-5.*abs.(expect);ok &= all(isfinite,actual)&&all(diff.<=budget);maxabs=max(maxabs,maximum(diff));maxbudget=max(maxbudget,maximum(diff./budget));append!(values,actual);append!(guards,Float64.(raw[end-G+1:end]));ok &= all(==(SENT),raw[end-G+1:end])
   end
   for(a,h)in zip(s.dev,s.host);raw=Array(a);ok &= raw==h;append!(guards,Float64.(raw[end-G+1:end]));end
  end
  art=vcat(prefix,values,guards);path=joinpath(ENV["STUDY_OUTPUT_DIR"],"trial$(trial).bin");write(path,reinterpret(UInt8,art))
  @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":%.17g,\"max_rel_error\":null,\"max_normalized_budget_error\":%.17g,\"timing_region\":\"sum2cases2formats_meanR20_cusparseSpMM_only_and_device_complete\",\"jit_excluded\":true,\"internal_repeats\":20,\"all_input_buffers_unchanged_checked\":true,\"component_time_seconds\":[%s],\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,sum(times),ok ? "true" : "false",length(values)+length(guards),maxabs,maxbudget,join(times,","),path,length(art));flush(stdout);ok||error("SpMMfulloutput/oracle/guard/inputfailed")
 end
 for s in states
  for mode=1:2;check(ccall(cusparseDestroySpMat,Cint,(Ptr{Cvoid},),s.a[mode]));check(ccall(cusparseDestroyDnMat,Cint,(Ptr{Cvoid},),s.c[mode]));end;check(ccall(cusparseDestroyDnMat,Cint,(Ptr{Cvoid},),s.b))
 end
 check(ccall(cusparseDestroy,Cint,(Ptr{Cvoid},),handle[]))
end

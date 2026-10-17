using Libdl
include(joinpath(@__DIR__,"study_fixture.jl"))
const STUDY_LIB="/usr/lib/x86_64-linux-gnu/libcublas.so.12.0.2.224"
function callgels(fn,handle,Aptr,Bptr,info)
 ccall(fn,Cint,(Ptr{Cvoid},Cint,Cint,Cint,Cint,CuPtr{Cvoid},Cint,CuPtr{Cvoid},Cint,Ref{Cint},CuPtr{Cvoid},Cint),handle,0,5,5,1,pointer(Aptr),5,pointer(Bptr),5,info,CuPtr{Cvoid}(0),2)
end
function study(allocate)
 lib=Libdl.dlopen(STUDY_LIB);handle=Ref{Ptr{Cvoid}}(C_NULL);ccall(Libdl.dlsym(lib,:cublasCreate_v2),Cint,(Ref{Ptr{Cvoid}},),handle)==0||error("CublasCreateFailed");version=Ref{Cint}();ccall(Libdl.dlsym(lib,:cublasGetVersion_v2),Cint,(Ptr{Cvoid},Ref{Cint}),handle[],version);println("STUDY_LIBRARY path=",realpath(STUDY_LIB)," version=",version[])
 cases=[];prefix=Float64[]
 for(T,fn,code)in ((Float32,:cublasSgelsBatched,0),(ComplexF32,:cublasCgelsBatched,1),(Float64,:cublasDgelsBatched,2),(ComplexF64,:cublasZgelsBatched,3))
  ah=T<:Complex ? T.(complex.(0.,INPUT_A)) : T.(INPUT_A);bh=T<:Complex ? T.(complex.(0.,INPUT_B)) : T.(INPUT_B)
  for data in (ah,bh),x in data;push!(prefix,Float64(real(x)),Float64(imag(x)));end
  append!(prefix,Float64[code,5,5,1,2,sizeof(T)])
  a=allocate(T,50);b=allocate(T,10);ap=allocate(UInt64,2);bp=allocate(UInt64,2);copyto!(ap,UInt64[UInt64(pointer(a,1)),UInt64(pointer(a,26))]);copyto!(bp,UInt64[UInt64(pointer(b,1)),UInt64(pointer(b,6))]);push!(cases,(a,b,ap,bp,ah,bh,Libdl.dlsym(lib,fn),Ref{Cint}(0),T<:Union{Float32,ComplexF32} ? 1e-6 : 1e-8))
 end
 function solve(c)
  a,b,ap,bp,ah,bh,fn,info,bound=c;copyto!(a,ah);copyto!(b,bh);CUDA.synchronize(CUDA.context());start=time_ns();status=callgels(fn,handle[],ap,bp,info);CUDA.synchronize(CUDA.context());elapsed=(time_ns()-start)*1e-9;got=Array(b);err=maximum(abs.(got.-one(eltype(b))));ok=status==0&&info[]==0&&all(isfinite,got)&&err<bound;vals=Float64[status,info[]]
  for x in got;push!(vals,Float64(real(x)),Float64(imag(x)));end
  return vals,elapsed,ok,Float64(err)
 end
 for c in cases;solve(c);end
 for trial in 1:5
  out=copy(prefix);elapsed=0.;ok=true;err=0.
  for c in cases;got,t,pass,e=solve(c);append!(out,got);elapsed+=t;ok&=pass;err=max(err,e);end
  path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"trial_$(trial).bin");open(io->write(io,out),path,"w")
  println("STUDY_TRIAL {\"trial\":",trial,",\"time_seconds\":",elapsed,",\"correct\":",ok,",\"checked_values\":",length(out),",\"max_abs_error\":",err,",\"timing_region\":\"sum_four_cublas_gels_batched_calls\",\"jit_excluded\":true,\"internal_repeats\":1,\"output_artifact\":{\"path\":\"",path,"\",\"dtype\":\"float64\",\"count\":",length(out),",\"byte_order\":\"little\"}}")
  ok||error("GelsFullComplexSolutionBoundFailed")
 end
 ccall(Libdl.dlsym(lib,:cublasDestroy_v2),Cint,(Ptr{Cvoid},),handle[])
end

using Libdl
const STUDY_LIB="/usr/lib/x86_64-linux-gnu/libcublas.so.12.0.2.224"
const PGS=UInt64(0xdecafbad12345678)
for T in (Float16,Float32,Float64)
 @eval function gemmcall(fn,h,m,alpha::Ref{$T},ap,bp,beta::Ref{$T},cp,batch)
  ccall(fn,Cint,(Ptr{Cvoid},Cint,Cint,Cint,Cint,Cint,Ref{$T},CuPtr{Cvoid},Cint,CuPtr{Cvoid},Cint,Ref{$T},CuPtr{Cvoid},Cint,Cint),h,0,0,m,1,m,alpha,pointer(ap,2),17,pointer(bp,2),17,beta,pointer(cp,2),17,batch)
 end
end
function study(allocate)
 fixture=collect(reinterpret(Float64,read(joinpath(@__DIR__,"fixtures_f64.bin"))));oracle=collect(reinterpret(Float64,read(joinpath(@__DIR__,"oracle_f64.bin"))));bounds=collect(reinterpret(Float64,read(joinpath(@__DIR__,"bounds_f64.bin"))))
 lib=Libdl.dlopen(STUDY_LIB);href=Ref{Ptr{Cvoid}}();ccall(Libdl.dlsym(lib,:cublasCreate_v2),Cint,(Ref{Ptr{Cvoid}},),href)==0||error("CreateHandle")
 ccall(Libdl.dlsym(lib,:cublasSetStream_v2),Cint,(Ptr{Cvoid},Ptr{Cvoid}),href[],Ptr{Cvoid}(CUDA.stream().handle))==0||error("BindStream")
 ver=Ref{Cint}();ccall(Libdl.dlsym(lib,:cublasGetVersion_v2),Cint,(Ptr{Cvoid},Ref{Cint}),href[],ver)==0||error("LibraryVersion");println("STUDY_LIBRARY path=",realpath(STUDY_LIB)," version=",ver[]," task_stream_bound=true")
 states=[];off=0
 for(T,symbol)in ((Float16,:cublasHgemmBatched),(Float32,:cublasSgemmBatched),(Float64,:cublasDgemmBatched)),mode in 0:1,n in (3,17)
  na=289n;nb=17n;ah=vcat(T(-1024),T.(fixture[off+7:off+6+na]),T(-1024));bh=vcat(T(-1024),T.(fixture[off+7+na:off+6+na+nb]),T(-1024));a=allocate(T,na+2);b=allocate(T,nb+2);c=allocate(T,nb+2);copyto!(a,ah);copyto!(b,bh)
  ph=[vcat(PGS,UInt64[UInt64(pointer(x,2+(j-1)*stride)) for j in 1:n],PGS) for(x,stride)in ((a,289),(b,17),(c,17))];ap=[allocate(UInt64,n+2) for k in 1:3];for k in 1:3;copyto!(ap[k],ph[k]);end
  push!(states,(T,n,a,b,c,ap,ah,bh,ph,Libdl.dlsym(lib,symbol)));off+=6+na+nb
 end
 function call(c,m)
  T,n,a,b,out,ap,ah,bh,ph,fn=c;gemmcall(fn,href[],Cint(m),Ref{T}(one(T)),ap[1],ap[2],Ref{T}(zero(T)),ap[3],Cint(n))==0||error("GemmBatchedStatus")
 end
 for c in states;call(c,17);end;CUDA.synchronize(CUDA.context())
 for trial in 1:5
  out=copy(fixture);oi=0;elapsed=0.;ok=true;err=0.
  for c in states
   T,n,a,b,dev,ap,ah,bh,ph,fn=c
   for m in 2:17
    fill!(dev,T(-1024))
    for rep in 0:10
     CUDA.synchronize(CUDA.context());start=time_ns();call(c,m);CUDA.synchronize(CUDA.context());dt=(time_ns()-start)*1e-9;rep>0&&(elapsed+=dt/10)
    end
    got=Array(dev);ok&=isequal(got[1],T(-1024))&&isequal(got[end],T(-1024))
    for j in 1:17n
     v=Float64(got[j+1]);e=abs(v-oracle[oi+j]);ok&=isfinite(v)&&e<=bounds[oi+j]+1e-30
     if mod(j-1,17)>=m;ok&=isequal(got[j+1],T(-1024));else;err=max(err,e);end
     push!(out,v)
    end
    append!(out,Float64[got[1],got[end],0]);oi+=17n
   end
   ok&=reinterpret(UInt8,Array(a))==reinterpret(UInt8,ah)&&reinterpret(UInt8,Array(b))==reinterpret(UInt8,bh)&&all(Array(ap[k])==ph[k] for k in 1:3)
  end
  path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"trial_$(trial).bin");open(io->write(io,out),path,"w")
  println("STUDY_TRIAL {\"trial\":",trial,",\"time_seconds\":",elapsed,",\"correct\":",ok,",\"checked_values\":",length(out),",\"max_abs_error\":",err,",\"timing_region\":\"sum192_batchedGEMV_10callmeans_skip_internal0_contextcomplete\",\"jit_excluded\":true,\"internal_repeats\":10,\"output_artifact\":{\"path\":\"",path,"\",\"dtype\":\"float64\",\"count\":",length(out),",\"byte_order\":\"little\"}}")
  ok||error("BatchedGEMVFullPayloadPaddingPointersAndDotOracle")
 end
 ccall(Libdl.dlsym(lib,:cublasDestroy_v2),Cint,(Ptr{Cvoid},),href[])==0||error("DestroyHandle")
end

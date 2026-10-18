using Libdl
include(joinpath(@__DIR__,"study_fixture.jl"))
const STUDY_LIB="/usr/lib/x86_64-linux-gnu/libcublas.so.12.0.2.224"
function study(allocate)
 lib=Libdl.dlopen(STUDY_LIB);handle=Ref{Ptr{Cvoid}}();ccall(Libdl.dlsym(lib,:cublasCreate_v2),Cint,(Ref{Ptr{Cvoid}},),handle)==0||error("create")
 a=allocate(Float32,9216);ap=allocate(UInt64,4);pv=allocate(Int32,192);info=allocate(Int32,4);copyto!(ap,UInt64[UInt64(pointer(a,1+b*2304)) for b in 0:3]);fn=Libdl.dlsym(lib,:cublasSgetrfBatched)
 function run()
  copyto!(a,INPUT_A);fill!(pv,0);fill!(info,0);CUDA.synchronize(CUDA.context());start=time_ns();status=ccall(fn,Cint,(Ptr{Cvoid},Cint,CuPtr{Cvoid},Cint,CuPtr{Cvoid},CuPtr{Cvoid},Cint),handle[],48,pointer(ap),48,pointer(pv),pointer(info),4);CUDA.synchronize(CUDA.context());t=(time_ns()-start)*1e-9
  lu=Array(a);p=Array(pv);inf=Array(info);ok=status==0&&inf==Int32[0,0,1,0];err=0.
  for b in 0:3
   perm=collect(1:48)
   for k in 1:48;q=Int(p[b*48+k]);if !(k<=q<=48);ok=false;continue;end;perm[k],perm[q]=perm[q],perm[k];end
   for j in 1:48,i in 1:48
    x=0.
    for k in 1:48;l=i==k ? 1. : i>k ? Float64(lu[b*2304+(k-1)*48+i]) : 0.;u=k<=j ? Float64(lu[b*2304+(j-1)*48+k]) : 0.;x+=l*u;end
    err=max(err,abs(x-Float64(INPUT_A[b*2304+(j-1)*48+perm[i]])))
   end
  end
  out=vcat(Float64.(INPUT_A),Float64[48,4,1,4,1],Float64.(lu),Float64.(p),Float64.(inf),Float64[status]);return out,t,ok&&err<2e-6,err
 end
 run()
 for trial in 1:5
  out,t,ok,e=run();path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"trial_$(trial).bin");open(io->write(io,out),path,"w");println("STUDY_TRIAL {\"trial\":",trial,",\"time_seconds\":",t,",\"correct\":",ok,",\"checked_values\":",length(out),",\"max_abs_error\":",e,",\"timing_region\":\"one_cublas_getrf_batched_call\",\"jit_excluded\":true,\"internal_repeats\":1,\"output_artifact\":{\"path\":\"",path,"\",\"dtype\":\"float64\",\"count\":",length(out),",\"byte_order\":\"little\"}}");ok||error("LUIndependentIdentity")
 end
 ccall(Libdl.dlsym(lib,:cublasDestroy_v2),Cint,(Ptr{Cvoid},),handle[])
end

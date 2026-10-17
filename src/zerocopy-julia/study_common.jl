const RUNTIME_ZERO=CUDA.CUDACore.CUDA_Runtime.libcudart
const STUDY_N=1048579
const STUDY_REPEATS=10
rtcheck(x)=x==0 ? nothing : error("cudartmappedhostAPIstatus",x)
@inline function add_body(i,a,b,c,n)
 if i<n;@inbounds c[i+1]=a[i+1]+b[i+1];end
 return
end
function case!(run!,generic)
 raw=Ptr{Cvoid}[];ptrs=Ptr{Cvoid}[];bytes=4STUDY_N;t=time_ns()
 for j in 1:3
  if generic
   p=Libc.malloc(bytes+4096);push!(raw,p);aligned=Ptr{Cvoid}((UInt(p)+4095)&~UInt(4095));rtcheck(ccall((:cudaHostRegister,RUNTIME_ZERO),Cint,(Ptr{Cvoid},Csize_t,Cuint),aligned,bytes,2));push!(ptrs,aligned)
  else
   out=Ref{Ptr{Cvoid}}();rtcheck(ccall((:cudaHostAlloc,RUNTIME_ZERO),Cint,(Ref{Ptr{Cvoid}},Csize_t,Cuint),out,bytes,2));push!(ptrs,out[])
  end
 end
 alloc=(time_ns()-t)*1e-9;host=[unsafe_wrap(Vector{Float32},Ptr{Float32}(p),STUDY_N;own=false) for p in ptrs]
 for i in 0:STUDY_N-1;host[1][i+1]=Float32(mod(37i+11,1024)-512)/4096;host[2][i+1]=Float32(mod(53i+7,2048)-1024)/8192;host[3][i+1]=Float32(-99);end
 t=time_ns();gpu=Any[]
 for p in ptrs
  out=Ref{Ptr{Cvoid}}();rtcheck(ccall((:cudaHostGetDevicePointer,RUNTIME_ZERO),Cint,(Ref{Ptr{Cvoid}},Ptr{Cvoid},Cuint),out,p,0));g=unsafe_wrap(CuArray,CuPtr{Float32}(UInt(out[])),(STUDY_N,);own=false);occursin("HostMemory",string(typeof(g)))||error("Notactualmappedhostmemory");push!(gpu,g)
 end
 binding=(time_ns()-t)*1e-9;CUDA.synchronize();t=time_ns();for repeat in 1:STUDY_REPEATS;run!(gpu[1],gpu[2],gpu[3],Int32(STUDY_N));end;CUDA.synchronize();kernel=(time_ns()-t)*1e-9/STUDY_REPEATS;got=copy(host[3]);expected=Float32[Float32(2*(mod(37i+11,1024)-512)+(mod(53i+7,2048)-1024))/8192 for i in 0:STUDY_N-1];ok=got==expected
 t=time_ns()
 for j in 1:3
  if generic;rtcheck(ccall((:cudaHostUnregister,RUNTIME_ZERO),Cint,(Ptr{Cvoid},),ptrs[j]));Libc.free(raw[j])
  else;rtcheck(ccall((:cudaFreeHost,RUNTIME_ZERO),Cint,(Ptr{Cvoid},),ptrs[j]));end
 end
 freed=(time_ns()-t)*1e-9
 return got,ok,kernel,alloc,binding,freed
end
function study(run!)
 CUDA.functional()||error("CUDAunavailable")
 for generic in (false,true);case!(run!,generic);end
 for trial in 1:5
  total=0.;got=Float32[];ok=true;stages=[]
  for generic in (false,true)
   data,good,elapsed,alloc,binding,freed=case!(run!,generic);total+=elapsed;ok&=good;append!(got,data);push!(stages,(alloc,binding,freed))
  end
  path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"trial_$(trial).bin");open(io->write(io,got),path,"w")
  println("STUDY_TRIAL {\"trial\":",trial,",\"time_seconds\":",total,",\"correct\":",ok,",\"checked_values\":2097158,\"max_abs_error\":",ok ? 0 : 1,",\"timing_region\":\"sum_both_mapped_host_modes_add_kernel_sync_means\",\"jit_excluded\":true,\"internal_repeats\":10,\"mapped_host_verified\":true,\"excluded_allocation_binding_free_seconds\":[[",join(stages[1],","),"],[",join(stages[2],","),"]],\"output_artifact\":{\"path\":\"",path,"\",\"dtype\":\"float32\",\"count\":2097158,\"byte_order\":\"little\"}}")
  ok||error("Mappedhostfulloutputfailed")
 end
end

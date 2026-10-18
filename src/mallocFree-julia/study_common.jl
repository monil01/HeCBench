
using Libdl,Printf
const BRIDGE="/home/imo/HeCBench_julia/.porting-state/portable_julia/runtime/mallocFree/libstudy_runtime.so"
function mallocfree_study()
 handle=Libdl.dlopen(BRIDGE);init=Libdl.dlsym(handle,:study_init);batch=Libdl.dlsym(handle,:study_batch)
 path=zeros(UInt8,1024);ver=Ref{Cint}(0);status=ccall(init,Cint,(Ptr{UInt8},Cint,Ref{Cint}),path,1024,ver);println("STUDY_RUNTIME {\"loaded_path\":\"",unsafe_string(pointer(path)),"\",\"runtime_version\":",ver[],",\"init_status\":",status,",\"device_kernels\":0}");flush(stdout);status==0||error("Runtime initialization failed")
 count=500;p=fill(Ptr{Cvoid}(C_NULL),count);ac=zeros(Cint,count);fr=zeros(Cint,count);nn=zeros(UInt8,count);times=zeros(Float64,2)
 function run(type,bytes,count)
  ccall(batch,Cint,(Cint,Csize_t,Cint,Ptr{Ptr{Cvoid}},Ptr{Cint},Ptr{UInt8},Ptr{Cint},Ptr{Cdouble}),type,bytes,count,p,ac,nn,fr,times)
 end
 for type=0:2,i=0:18;run(type,1<<(i+6),1)==0||error("Warmup allocation failed");end
 for trial=1:5
  trace=IOBuffer();print(trace,"[");components=Float64[];ok=true
  for type=0:2,i=0:18
   bytes=1<<(i+6);status=run(type,bytes,count);ok &= status==0 && all(iszero,ac) && all(==(1),nn) && all(iszero,fr) && all(==(C_NULL),p)
   append!(components,times)
   position(trace)>1 && print(trace,",")
   print(trace,"{\"type\":",type,",\"bytes\":",bytes,",\"batch_status\":",status,",\"allocation_return_codes\":[",join(ac,","),"],\"nonnull_allocations\":[",join(nn.==1,","),"],\"free_return_codes\":[",join(fr,","),"]}")
  end
  print(trace,"]")
  @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":%s,\"execution_checked\":true,\"checked_values\":0,\"max_abs_error\":null,\"max_rel_error\":null,\"source_empty_kernel\":false,\"source_no_device_kernels\":true,\"no_numerical_outputs\":true,\"no_output_reason\":\"runtime_allocation_lifecycle\",\"runtime_api_status_checked\":true,\"nonnull_allocations_checked\":true,\"allocation_lifecycle_checked\":true,\"jit_excluded\":true,\"timing_region\":\"sum114_mean_host_runtime_alloc_free_loop_intervals\",\"internal_repeats\":500,\"allocated_objects_per_trial\":28500,\"freed_objects_per_trial\":28500,\"component_time_seconds\":[%s],\"runtime_api_status_trace\":%s}\n",trial,sum(components),ok ? "true" : "false",join(components,","),String(take!(trace)));flush(stdout);ok||error("Actual allocation/nonnull/free lifecycle failed")
 end
end

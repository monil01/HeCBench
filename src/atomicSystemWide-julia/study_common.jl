using CUDA,Printf,Libdl
function atomic_study(factory)
 loops=Int32(1000);ng=Int32(8);initial=zeros(Int32,10);initial[8]=initial[10]=255
 hlib=Libdl.dlopen(ENV["STUDY_HOST_LIBRARY"]);cpu=Libdl.dlsym(hlib,:study_cpu)
 a=CuArray{Int32,1,CUDA.UnifiedMemory}(undef,42);run=factory(a,loops)
 function reset!()
  CUDA.device_synchronize(blocking=true)
  CUDA.@allowscalar for i in 1:42;a[i]=Int32(1515870810);end
  unsafe_copyto!(pointer(a,17;type=CUDA.HostMemory),pointer(initial),10)
 end
 function complete!()
  run();ccall(cpu,Cvoid,(Ptr{Int32},Cint,Cint),pointer(a,17;type=CUDA.HostMemory),ng,loops);CUDA.device_synchronize(blocking=true)
 end
 reset!();complete!();allok=true
 for trial in 1:5
  reset!();start=time_ns();complete!();seconds=(time_ns()-start)*1e-9
  raw=Array(a);v=raw[17:26];guards=vcat(raw[1:16],raw[27:42]);expected=Int32[160000,15,0,0,0,1,65535,255];got=v[[1,3,4,5,6,8,9,10]];ex=Int32(0<=v[2]<16);cas=Int32(0<=v[7]<16);ok=got==expected&&ex==1&&cas==1&&all(==(Int32(1515870810)),guards);allok&=ok
  artifact=vcat(Int32[8,8,loops],initial,got,Int32[ex,cas],guards);path=joinpath(ENV["STUDY_OUTPUT_DIR"],"trial$(trial).bin");write(path,artifact)
  @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":42,\"max_abs_error\":%d,\"max_rel_error\":null,\"verification_contract\":\"native_property_scores\",\"raw_state\":[%s],\"timing_region\":\"shared_CPU_GPU_systematomic_coexecution_completion\",\"jit_excluded\":true,\"internal_repeats\":1,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"int32\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,seconds,ok ? "true" : "false",maximum(abs.(Int64.(got).-Int64.(expected))),join(v,","),path,length(artifact))
 end
 allok||error("systematomiccompleteproperties/guards failed");println("PASS")
end

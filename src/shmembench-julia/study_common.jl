using CUDA,Printf
include("study_device.jl")
function shmem_study(factory)
 sizeof(SHMEM_VEC4)==16||error("vectorwidth");Base.datatype_alignment(SHMEM_VEC4)==16||error("vectoralignment")
 expected=reinterpret(Float32,read(joinpath(@__DIR__,"expected_f32_guards.bin")));guard=ntuple(_->VecElement(16777215f0),Val(4));out=CuArray(fill(guard,524288));launch=factory(out);repeat=isempty(ARGS) ? 1000 : parse(Int,ARGS[1]);launch();CUDA.device_synchronize(blocking=true);allok=true
 println("SHMEM_VECTOR_BYTES=",sizeof(SHMEM_VEC4)," SHMEM_VECTOR_ALIGNMENT=",Base.datatype_alignment(SHMEM_VEC4))
 for trial in 1:5
  CUDA.device_synchronize(blocking=true);start=time_ns();for r in 1:repeat;launch();end;CUDA.device_synchronize(blocking=true);elapsed=(time_ns()-start)*1e-9/repeat
  result=reinterpret(Float32,Array(out));ok=result==expected;allok&=ok;err=maximum(abs.(result.-expected));file=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"shmembench_trial$(trial).bin");write(file,result)
  @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%d,\"defined_payload_values\":1048576,\"unwritten_guard_values\":1048576,\"max_abs_error\":%.12g,\"max_rel_error\":null,\"internal_repeats\":%d,\"jit_excluded\":true,\"timing_region\":\"source1000_kernel_loopmean_1024_swapiterations_devicecompletion\",\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float32\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,elapsed,string(ok),length(result),err,repeat,file,length(result))
 end
 println(allok ? "PASS" : "FAIL");allok||error("oraclefailed")
end

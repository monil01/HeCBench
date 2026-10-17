using CUDA,Printf
function nms_study(factory)
 mode=length(ARGS)>0 ? ARGS[1] : "map";raw=read(joinpath(@__DIR__,"study_input_f32.bin"));host=reshape(collect(reinterpret(Float32,raw)),4,4096);expected=read(joinpath(@__DIR__,"study_expected_u8.bin"));inputs=[CuArray(collect(host[j,:])) for j in 1:4];bitmap=CUDA.fill(UInt8(1),4096*4096);points=CUDA.zeros(UInt8,4096);map,reduce=factory(inputs,bitmap,points);CUDA.device_synchronize(blocking=true);map();CUDA.device_synchronize(blocking=true);reduce();CUDA.device_synchronize(blocking=true)
 allok=true
 for trial in 1:5
  fill!(bitmap,UInt8(1));fill!(points,UInt8(0));CUDA.device_synchronize(blocking=true)
  if mode=="reduce";map();CUDA.device_synchronize(blocking=true);end
  start=time_ns()
  for rep in 1:10
   if mode=="map";map();else;reduce();end
  end
  CUDA.device_synchronize(blocking=true);seconds=(time_ns()-start)*1e-9/10
  if mode=="map";reduce();CUDA.device_synchronize(blocking=true);end
  out=vcat(Array(bitmap),Array(points));ok=out==expected;allok&=ok;errorabs=maximum(abs.(Int16.(out).-Int16.(expected)));got=vcat(raw,out);path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"output_trial$(trial).bin");write(path,got)
  @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":%d,\"max_rel_error\":%d,\"timing_region\":\"kernel_launch_loop_per_internal_repeat\",\"jit_excluded\":true,\"internal_repeats\":10,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"uint8\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,seconds,ok ? "true" : "false",length(got),errorabs,errorabs,path,length(got))
 end
 allok||error("FullNMSbitmap/maskmismatch");println("PASS")
end

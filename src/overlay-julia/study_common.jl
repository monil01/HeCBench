using CUDA,Printf
include("study_device.jl")
function overlay_study(factory)
 raw=read(joinpath(@__DIR__,"study_input_f32.bin"));br=read(joinpath(@__DIR__,"study_boxes_i32.bin"));host=collect(reinterpret(NTuple{3,Float32},raw));boxes=collect(reinterpret(OverlayBox,br));expected=collect(reinterpret(Float64,read(joinpath(@__DIR__,"study_expected_f64.bin"))));color=(255f0,204f0,203f0,1f0);prefix=vcat(Float64.(reinterpret(Float32,raw)),Float64.(reinterpret(Int32,br)),Float64[color...],Float64[640,480,length(boxes)]);sizeof(OverlayBox)==16&&sizeof(NTuple{3,Float32})==12||error("Overlay ABI");input=CuArray(host);output=copy(input);launch=factory(input,output,boxes,color);CUDA.device_synchronize(blocking=true);launch();CUDA.device_synchronize(blocking=true);allok=true
 for trial in 1:5
  copyto!(output,input);CUDA.device_synchronize(blocking=true);start=time_ns();launch();CUDA.device_synchronize(blocking=true);seconds=(time_ns()-start)*1e-9
  gotpixels=Array(output);out=Float64.(reinterpret(Float32,gotpixels));diff=abs.(out.-expected);errorabs=maximum(diff);relative=maximum(diff./max.(abs.(expected),1e-3));ok=all(isfinite,out)&&errorabs<=1e-3;allok&=ok;got=vcat(prefix,out);path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"output_trial$(trial).bin");write(path,got)
  @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":%.12g,\"max_rel_error\":%.12g,\"timing_region\":\"kernel_launch_loop_per_internal_repeat\",\"jit_excluded\":true,\"internal_repeats\":1,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,seconds,ok ? "true" : "false",length(got),errorabs,relative,path,length(got))
 end
 allok||error("FulloverlayRGBmismatch");println("PASS")
end

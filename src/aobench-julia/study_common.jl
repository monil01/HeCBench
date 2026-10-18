using CUDA,Printf
function ao_study(factory)
 N=256*256*3;R=100;scene=Float32[-2,0,-3.5,.5,-.5,0,-3,.5,1,0,-2.2,.5,0,-.5,0,0,1,0];bits=read(joinpath(@__DIR__,"study_scene_f32.bin"));@assert reinterpret(UInt8,scene)==bits
 expected=read(joinpath(@__DIR__,"study_expected_u8.bin"));out=CUDA.fill(UInt8(165),N+128);img=view(out,65:N+64);run=factory(img);run();CUDA.device_synchronize(blocking=true);allok=true
 for trial in 1:5
  fill!(out,UInt8(165));CUDA.device_synchronize(blocking=true);start=time_ns();for _ in 1:R;run();end;CUDA.device_synchronize(blocking=true);seconds=(time_ns()-start)*1e-9/R
  oh=Array(out);got=oh[65:end-64];guards=vcat(oh[1:64],oh[end-63:end]);err=abs.(Int16.(got).-Int16.(expected));ma=maximum(err);mr=maximum(Float64.(err)./max.(Float64.(expected),1e-300));rgb=reshape(got,3,:);gray=all(rgb[1,:].==rgb[2,:])&&all(rgb[2,:].==rgb[3,:]);sky=all(iszero,got[1:256*3]);ok=ma<=1&&gray&&sky&&all(==(UInt8(165)),guards);allok&=ok;prefix=vcat(bits,reinterpret(UInt8,Int32[256,256,2,8,R]));a=vcat(prefix,got,guards);path=joinpath(ENV["STUDY_OUTPUT_DIR"],"trial$(trial).bin");write(path,a)
  @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":%d,\"max_rel_error\":%.12g,\"error_metric_domain\":\"quantizedRGBvsindependentFloat64geometry_onegraycodebudget_exactgrayscale_sky_guards\",\"timing_region\":\"kernel_launch_loop_per_internal_repeat\",\"jit_excluded\":true,\"internal_repeats\":%d,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"uint8\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,seconds,ok ? "true" : "false",N+128,ma,mr,R,path,length(a))
 end
 allok||error("AOfullimage/CPUgeometry/gray/sky/guard mismatch");println("PASS")
end

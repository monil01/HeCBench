using CUDA,Printf
function car_study(factory)
 input=collect(reinterpret(Float32,read(joinpath(@__DIR__,"study_inputs_f32.bin"))));expected=collect(reinterpret(Float64,read(joinpath(@__DIR__,"oracle_f64.bin"))));sizes=[6840,7866,7866,7866,2622];bases=[CUDA.fill(12345.125f0,n+32) for  n in sizes];arrays=[view(b,17:n+16) for (b,n) in  zip(bases,sizes)];pos=1
 for i in 1:4;copyto!(arrays[i],collect(view(input,pos:pos+sizes[i]-1)));pos+=sizes[i];end
 run=factory(arrays);run();CUDA.device_synchronize(blocking=true);allok=true
 for trial in 1:5
  CUDA.device_synchronize(blocking=true);start=time_ns();for _ in 1:100;run();end;CUDA.device_synchronize(blocking=true);seconds=(time_ns()-start)*1e-9/100
  h=Array.(arrays);captured=vcat(h[1:4]...);ok=captured==input;got=h[5];err=abs.(Float64.(got).-expected);ma=maximum(err);mr=maximum(err./max.(abs.(expected),1e-300));scaled=maximum(err./(0.00005 .+0.000002.*abs.(expected)));ok&=all(isfinite,got)&&scaled<=1;hull=true
  for j in 0:2621;b=j÷(3*19*23);c=(j÷(19*23))%3;hull&=b*8+c*2-.00005<=got[j+1]<=b*8+c*2+6.734375+.00005;end
  ok&=hull;artifact=vcat(captured,Float32[2,3,19,23,9,37,29,1,1,100],got)
  for (base,n) in  zip(bases,sizes);raw=Array(base);for j in 1:16;push!(artifact,raw[j],raw[n+16+j]);ok&=raw[j]==12345.125f0&&raw[n+16+j]==12345.125f0;end;end
  allok&=ok;path=joinpath(ENV["STUDY_OUTPUT_DIR"],"trial$(trial).bin");write(path,artifact)
  @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":%.12g,\"max_rel_error\":%.12g,\"max_scaled_oracle_error\":%.12g,\"convex_hull_check\":%s,\"timing_region\":\"kernel_launch_loop_per_internal_repeat\",\"jit_excluded\":true,\"internal_repeats\":100,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float32\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,seconds,ok ? "true" : "false",length(artifact),ma,mr,scaled,hull ? "true" : "false",path,length(artifact))
 end
 allok||error("CARcompleteclosedform/input/guards/hull mismatch");println("PASS")
end

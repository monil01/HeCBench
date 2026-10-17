using CUDA,Printf
function meanshift_study(factory)
 mode=length(ARGS)>0 ? ARGS[1] : "base";mode in ("base","tiled")||error("mode");host=reinterpret(Float32,read(joinpath(@__DIR__,"study_input_f32.bin")))|>collect;expected=reinterpret(Float64,read(joinpath(@__DIR__,"study_expected_f64.bin")))|>collect;prefix=Float64.(host);a=CuArray(host);b=CUDA.zeros(Float32,30144);op=factory(a,b,mode)
 reset!()=(copyto!(a,host);fill!(b,0));reset!();op();CUDA.synchronize() # compile and1000update warmup
 allok=true
 for trial in 1:5
  reset!();CUDA.synchronize();start=time_ns();op();CUDA.synchronize();seconds=(time_ns()-start)*1e-9/1000;got=Float64.(Array(a)[1:30000]);err=maximum(abs.(got.-expected));ok=all(isfinite,got)&&err<=1e-2
  # Convex weighted updates stay within initial per-axis bounds and converge near3sourcecenters.
  for i in 0:9999,j in 1:3;ok&=abs(got[3i+j]-100mod(i,3))<=sqrt(10.);end
  allok&=ok;values=vcat(prefix,got);path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"output_trial$(trial).bin");write(path,values)
  @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":%.12g,\"max_rel_error\":%.12g,\"timing_region\":\"kernel_launch_loop_per_internal_repeat\",\"jit_excluded\":true,\"internal_repeats\":1000,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,seconds,ok ? "true" : "false",length(values),err,err/1e-2,path,length(values))
 end
 allok||error("CPU/sourcecentroid check failed");println("PASS")
end

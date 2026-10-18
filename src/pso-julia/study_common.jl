using CUDA,Printf
include("study_device.jl")
@inline function source_random_clamped(seed)
 ccall(:srand,Cvoid,(Cuint,),Cuint(seed));Float32(ccall(:rand,Cint,()))/Float32(2147483647)
end
function pso_study(factory)
 P=Int32(30);R=Int32(10);Q=Int(P)*30;input=collect(reinterpret(Float32,read(joinpath(@__DIR__,"study_input_r10_f32.bin"))));expected=collect(reinterpret(Float32,read(joinpath(@__DIR__,"study_expected_cpu_r10_f32.bin"))));pos=CuArray(input[1:Q]);vel=CuArray(input[Q+1:2Q]);best=CuArray(input[2Q+1:3Q]);globalbest=CuArray(input[3Q+1:end]);prefix=vcat(input,Float32[P,R,.5,1.5,1.5,3.1415]);launch=factory(pos,vel,best,globalbest,P,R)
 reset()=(copyto!(pos,input[1:Q]);copyto!(vel,input[Q+1:2Q]);copyto!(best,input[2Q+1:3Q]);copyto!(globalbest,input[3Q+1:end]);nothing)
 reset();CUDA.device_synchronize(blocking=true);launch();CUDA.device_synchronize(blocking=true);allok=true
 for trial in 1:5
  reset();CUDA.device_synchronize(blocking=true);start=time_ns();launch();CUDA.device_synchronize(blocking=true);seconds=(time_ns()-start)*1e-9/Int(R);state=vcat(Array(pos),Array(vel),Array(best),Array(globalbest));diff=abs.(Float64.(state).-Float64.(expected));ma=maximum(diff);mr=maximum(diff./max.(abs.(Float64.(expected)),1e-12));ok=all(isfinite,state)&&ma<=1e-3;allok&=ok;out=vcat(prefix,state);length(out)==5466||error("artifactshape");path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"output_trial$(trial).bin");write(path,out)
  @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":%.12g,\"max_rel_error\":%.12g,\"timing_region\":\"kernel_launch_loop_per_internal_repeat\",\"jit_excluded\":true,\"internal_repeats\":%d,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float32\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,seconds,ok ? "true" : "false",length(out),ma,mr,R,path,length(out))
 end
 allok||error("PSOfullstate mismatch");println("PASS")
end

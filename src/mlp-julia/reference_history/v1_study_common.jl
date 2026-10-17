using CUDA,Printf
function mlp_read(name,T)
 reinterpret(T,read(joinpath(@__DIR__,name)))|>collect
end
function mlp_study(factory)
 mode=length(ARGS)>0 ? ARGS[1] : "b1_r1";use_bias=occursin("b1",mode);use_relu=occursin("r1",mode);host=mlp_read("study_input_f32.bin",Float32);cpu=mlp_read("study_expected_$(mode)_f64.bin",Float64);prefix=Float64.(host);x=CuArray(host[1:100352]);pos=100353;weights=CuArray{Float32,1}[];biases=CuArray{Float32,1}[]
 for n in (401408,262144,5120);push!(weights,CuArray(host[pos:pos+n-1]));pos+=n;end
 for n in (512,512,10);push!(biases,CuArray(host[pos:pos+n-1]));pos+=n;end
 pos==length(host)+1||error("inputshape");outputs=[CUDA.zeros(Float32,n)for n in (65536,65536,1280)];op,cleanup=factory(x,weights,biases,outputs,use_bias,use_relu);CUDA.synchronize()
 # Source preconditions300forwardwarmups; first also compiles Julia/JIT.
 for warm in 1:300;op();CUDA.synchronize();end
 allok=true
 try
  for trial in 1:5
   for a in outputs;fill!(a,0f0);end;CUDA.synchronize();start=time_ns();for rep in 1:10;op();end;CUDA.synchronize();seconds=(time_ns()-start)*1e-9/10
   final=Float64.(vcat((Array(a)for a in outputs)...));errorabs=maximum(abs.(final.-cpu));errorrel=maximum(abs.(final.-cpu)./max.(abs.(cpu),1e-2));ok=all(isfinite,final)&&errorabs<=1e-2;if use_relu;ok&=all(final.>=0);end;allok&=ok;got=vcat(prefix,final);path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"output_trial$(trial).bin");write(path,got)
   @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":%.12g,\"max_rel_error\":%.12g,\"timing_region\":\"kernel_launch_loop_per_internal_repeat\",\"jit_excluded\":true,\"internal_repeats\":10,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,seconds,ok ? "true" : "false",length(got),errorabs,errorrel,path,length(got))
  end
 finally;CUDA.synchronize();cleanup();end
 allok||error("Full3layerCPUforwardmismatch");println("PASS")
end

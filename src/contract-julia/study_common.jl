using CUDA,Printf
function contract_study(factory)
 T=get(ENV,"STUDY_DTYPE","32")=="32" ? Float32 : Float64
 root=@__DIR__;tensor=T.(reinterpret(Float64,read(joinpath(root,"study_tensor_f64.bin"))));adj=T.(reinterpret(Float64,read(joinpath(root,"study_adj_f64.bin"))));oracle=collect(reinterpret(Float64,read(joinpath(root,"oracle_f64.bin"))))
 t=CuArray(tensor);a=CuArray(adj);base=CUDA.fill(T(-8192),26308);out=view(base,33:26276);run=factory(t,a,out);allok=true
 for trial in 1:5
  fill!(base,T(-8192));run();CUDA.device_synchronize(blocking=true)
  start=time_ns();for _ in 1:10;run();end;CUDA.device_synchronize(blocking=true);seconds=(time_ns()-start)*1e-9/10
  raw=Float64.(Array(base));actual=raw[33:26276];guards=vcat(raw[1:32],raw[26277:end]);tt=Float64.(Array(t));aa=Float64.(Array(a));err=maximum(abs.(actual.-oracle));rel=maximum(abs.(actual.-oracle)./max.(abs.(oracle),eps(Float64)));ok=all(isfinite,actual)&&actual==oracle&&all(==(-8192),guards)&&tt==Float64.(tensor)&&aa==Float64.(adj);allok&=ok
  artifact=vcat(tt,aa,Float64[9,18,18,8sizeof(T)],actual,guards);path=joinpath(ENV["STUDY_OUTPUT_DIR"],"trial$(trial).bin");write(path,artifact)
  @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":26308,\"max_abs_error\":%.12g,\"max_rel_error\":%.12g,\"timing_region\":\"full18case_contraction_kernel_launch_loop_per_repeat\",\"jit_excluded\":true,\"internal_repeats\":10,\"source_cases\":18,\"precision_bits\":%d,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,seconds,ok ? "true" : "false",err,rel,8sizeof(T),path,length(artifact))
 end
 allok||error("Fullcontraction/input/guard oracle mismatch");println("PASS")
end

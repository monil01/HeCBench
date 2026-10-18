using CUDA,Printf
include("study_device.jl")
function cache_study(factory)
 B=parse(Int32,ARGS[1]);x=parse(Int32,ARGS[2]);N=Int32(1024);H=Int32(32);D=Int32(128);R=1000;Q=Int(N)*H*D
 rd(T,name)=collect(reinterpret(T,read(joinpath(@__DIR__,name))))
 k=rd(Float32,"study_key_f32.bin");v=rd(Float32,"study_value_f32.bin");slots=rd(Int64,"study_slot_i64.bin");sc=rd(Float32,"study_scales_f32.bin");expected=rd(UInt8,"study_expected_B$(B)_x$(x)_u8.bin");@assert all(isfinite,k)&&all(isfinite,v)&&all(isfinite,sc)&&all(sc.>0)
 dk=CuArray(k);dv=CuArray(v);ds=CuArray(slots);kc=CUDA.zeros(UInt8,Q);vc=CUDA.zeros(UInt8,Q);launch=factory(dk,dv,kc,vc,ds,H,D,B,x,sc[1],sc[2]);for i in 1:30;launch();end;CUDA.device_synchronize(blocking=true)
 prefix=vcat(reinterpret(UInt32,k),reinterpret(UInt32,v),UInt32.(slots),reinterpret(UInt32,sc),UInt32[N,H,D,B,x,R,512]);allok=true
 for trial in 1:5
  CUDA.device_synchronize(blocking=true);t=time_ns();for rep in 1:R;launch();end;CUDA.device_synchronize(blocking=true);seconds=(time_ns()-t)*1e-9/R
  got=vcat(Array(kc),Array(vc));ok=got==expected&&all(z->(z&0x7f)!=0x7f,got);allok&=ok;err=abs.(Int32.(got).-Int32.(expected));ma=maximum(err);mr=maximum(Float64.(err)./max.(Float64.(expected),1e-30));out=vcat(prefix,UInt32.(got));path=joinpath(ENV["STUDY_OUTPUT_DIR"],"output_trial$(trial).bin");write(path,out)
  @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":%d,\"max_rel_error\":%.12g,\"error_metric_domain\":\"exact_finite_FP8_encoding_and_cache_layout\",\"timing_region\":\"kernel_launch_loop_per_internal_repeat\",\"jit_excluded\":true,\"internal_repeats\":1000,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"uint32\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,seconds,ok ? "true" : "false",length(got),ma,mr,path,length(out))
 end
 allok||error("FP8 conversion/layout mismatch");println("PASS")
end

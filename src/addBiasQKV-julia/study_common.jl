using CUDA,Printf
include("study_device.jl")
function qkv_study(factory)
 tag=length(ARGS)>0 ? ARGS[1] : "original";m,H,D,R=tag=="original" ? (Int32(8192),Int32(12),Int32(64),Int32(1000)) : (Int32(5),Int32(3),Int32(16),Int32(10));N=Int(m)*3H*D;B=Int(3H*D);readbin(name,T)=collect(reinterpret(T,read(joinpath(@__DIR__,"study_$(tag)_$(name).bin"))));input=readbin("src_u8",UInt8);bias=readbin("bias_u16",UInt16);scales=readbin("scales_f32",Float32);expected=readbin("expected_u8",UInt8)
 s=CuArray(vcat(fill(UInt8(90),64),input,fill(UInt8(90),64)));b=CuArray(vcat(fill(UInt16(0x5a5a),32),bias,fill(UInt16(0x5a5a),32)));o=CUDA.fill(UInt8(165),N+128);sc=CuArray(scales);run=factory(s,b,o,sc,m,H,D)
 for _ in 1:R;run();end;CUDA.device_synchronize(blocking=true);allok=true
 for trial in 1:5
  fill!(o,UInt8(165));CUDA.device_synchronize(blocking=true);start=time_ns();for _ in 1:R;run();end;CUDA.device_synchronize(blocking=true);seconds=(time_ns()-start)*1e-9/Int(R)
  sh=Array(s);bh=Array(b);oh=Array(o);sch=Array(sc);guards=vcat(oh[1:64],oh[end-63:end],sh[1:64],sh[end-63:end],reinterpret(UInt8,bh[1:32]),reinterpret(UInt8,bh[end-31:end]));got=oh[65:end-64];ok=got==expected&&sh[65:end-64]==input&&bh[33:end-32]==bias&&sch==scales&&guards==vcat(fill(UInt8(165),128),fill(UInt8(90),256));allok&=ok
  prefix=vcat(sh[65:end-64],reinterpret(UInt8,bh[33:end-32]),reinterpret(UInt8,sch),reinterpret(UInt8,Int32[m,H,D,R]));out=vcat(prefix,got,guards);path=joinpath(ENV["STUDY_OUTPUT_DIR"],"trial$(trial).bin");write(path,out);err=maximum(abs.(Int16.(got).-Int16.(expected)));mr=maximum(Float64.(abs.(Int16.(got).-Int16.(expected)))./max.(Float64.(expected),1e-300))
  @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":%d,\"max_rel_error\":%.12g,\"error_metric_domain\":\"exact_FP8bytes_BF16intermediates_transpose_and_guardbytes_not_attentionquality\",\"jit_excluded\":true,\"internal_repeats\":%d,\"timing_region\":\"kernel_launch_loop_per_internal_repeat\",\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"uint8\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,seconds,ok ? "true" : "false",N+384,err,mr,R,path,length(out))
 end
 allok||error("FullFP8bit/originalCPUoracle/guard mismatch");println("PASS")
end

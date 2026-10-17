using CUDA,Printf
include("study_device.jl")
function nqueen_study(factory)
 raw=read(joinpath(@__DIR__,"study_roots.bin"));host=collect(reinterpret(QueenRoot,raw));expected=collect(reinterpret(UInt64,read(joinpath(@__DIR__,"study_expected_u64.bin"))));prefix=collect(reinterpret(UInt64,raw));P=length(host);sizeof(QueenRoot)==16||error("QueenRoot ABI");roots=CuArray(host);tree=CUDA.zeros(UInt64,P);solutions=CUDA.zeros(UInt64,P);launch=factory(roots,tree,solutions,Int32(P));CUDA.device_synchronize(blocking=true);launch();CUDA.device_synchronize(blocking=true);allok=true
 for trial in 1:5
  fill!(tree,UInt64(0));fill!(solutions,UInt64(0));CUDA.device_synchronize(blocking=true);start=time_ns();launch();CUDA.device_synchronize(blocking=true);seconds=(time_ns()-start)*1e-9
  out=vcat(Array(tree),Array(solutions));ok=out==expected&&sum(view(out,P+1:2P))==UInt64(2279184)&&sum(view(out,1:P))+UInt64(2466109)==UInt64(171129071);allok&=ok;diff=map((a,b)->a>=b ? a-b : b-a,out,expected);errorabs=maximum(diff);relative=maximum(Float64.(diff)./max.(Float64.(expected),1.));got=vcat(prefix,out);path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"output_trial$(trial).bin");write(path,got)
  @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":%u,\"max_rel_error\":%.12g,\"timing_region\":\"kernel_launch_loop_per_internal_repeat\",\"jit_excluded\":true,\"internal_repeats\":1,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"uint64\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,seconds,ok ? "true" : "false",length(got),errorabs,relative,path,length(got))
 end
 allok||error("FullperprefixqueenDFSmismatch");println("PASS")
end

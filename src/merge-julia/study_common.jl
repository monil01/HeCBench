using CUDA,Printf
function merge_read(name,T)
 reinterpret(T,read(joinpath(@__DIR__,name)))|>collect
end
function merge_study(factory)
 mode=length(ARGS)>0 ? ARGS[1] : "uint32";T=Dict("uint32"=>UInt32,"uint64"=>UInt64,"float32"=>Float32,"float64"=>Float64)[mode];host=merge_read("study_input_$(mode).bin",T);cpu=merge_read("study_expected_$(mode).bin",T);paths=merge_read("study_paths_$(mode).bin",UInt32);expected=vcat(host,cpu,T.(paths));a=CuArray(host[1:101024]);b=CuArray(host[101025:end]);out=CUDA.zeros(T,201024);path=CUDA.zeros(UInt32,226);op=factory(a,b,path,out)
 reset!()=(fill!(out,zero(T));fill!(path,0));reset!();op();CUDA.synchronize();allok=true
 for trial in 1:5
  reset!();CUDA.synchronize();start=time_ns();for rep in 1:10;op();end;CUDA.synchronize();seconds=(time_ns()-start)*1e-9/10;got=vcat(host,Array(out)[1:200000],T.(Array(path)));ok=all(isfinite,got)&&got==expected;allok&=ok;absmax=relmax=0.
  for (x,y)in zip(got,expected)
   delta=T<:Integer ? Float64(abs(Int128(x)-Int128(y))) : abs(Float64(x)-Float64(y));absmax=max(absmax,delta);relmax=max(relmax,delta/max(abs(Float64(y)),1e-300))
  end
  filename=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"output_trial$(trial).bin");write(filename,got)
  @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":%.12g,\"max_rel_error\":%.12g,\"timing_region\":\"kernel_launch_loop_per_internal_repeat\",\"jit_excluded\":true,\"internal_repeats\":10,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"%s\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,seconds,ok ? "true" : "false",length(got),absmax,relmax,filename,mode,length(got))
 end
 allok||error("Fullmergedarray/independentstablepath mismatch");println("PASS")
end

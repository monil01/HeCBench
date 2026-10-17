using CUDA,Printf
include("study_device.jl")
function mr_read(name)
 collect(reinterpret(UInt32,read(joinpath(@__DIR__,name))))
end
function mr_study(factory)
 distribution=ARGS[1];eff=ARGS[2]=="efficient";host=mr_read("study_$(distribution)_u32.bin");expected=mr_read("study_expected_$(distribution)_u32.bin");bases=CuArray(UInt32[2,7,61]);numbers=[CuArray(host[i*100000+1:(i+1)*100000]) for i in 0:3];val=CUDA.zeros(Int32,1);codes=CUDA.zeros(UInt32,1200000);launch,evaluate=factory(bases,numbers,val,codes,eff);CUDA.synchronize()
 for width in 1:4;fill!(val,0);launch(width);CUDA.synchronize();end;evaluate();CUDA.synchronize()
 allok=true
 for trial in 1:5
  seconds=0.;totals=UInt32[]
  for width in 1:4
   for rep in 1:10
    fill!(val,0);CUDA.synchronize();start=time_ns();launch(width);CUDA.synchronize();seconds+=(time_ns()-start)*1e-9/10
   end
   push!(totals,UInt32(Array(val)[1]))
  end
  evaluate();CUDA.synchronize();out=vcat(Array(codes),totals);ok=out==expected;allok&=ok;got=vcat(UInt32[2,7,61],host,out);errorabs=maximum(abs.(Int64.(out).-Int64.(expected)));relative=maximum(abs.(Float64.(out).-Float64.(expected))./max.(Float64.(expected),1.));path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"output_trial$(trial).bin");write(path,got)
  @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":%d,\"max_rel_error\":%.12g,\"timing_region\":\"kernel_launch_loop_per_internal_repeat\",\"jit_excluded\":true,\"internal_repeats\":10,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"uint32\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,seconds,ok ? "true" : "false",length(got),errorabs,relative,path,length(got))
 end
 allok||error("FullperinputMillerRabin mismatch");println("PASS")
end

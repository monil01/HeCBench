using CUDA,Printf
include(joinpath(@__DIR__,"fp8_math.jl"))
const BITS=reinterpret(UInt32,read(joinpath(@__DIR__,"fixture_bits.bin")))
const N=length(BITS)
const EXPECTED=read(joinpath(@__DIR__,"expected_codes.bin"))
function study_main(launch!)
 rep=isempty(ARGS) ? 20 : parse(Int,ARGS[1]);src=CuArray(reinterpret(Float32,BITS));dst=[CUDA.fill(UInt8(165),N+31) for _ in 1:4]
 for path in 1:4;launch!(path,dst[path],src,N);end;CUDA.synchronize()
 for trial in 1:5
  for a in dst;CUDA.fill!(a,UInt8(165));end;CUDA.synchronize();times=Float64[];gct=0.;gcp=0
  for path in 1:4
   CUDA.synchronize();gb=Base.gc_num();st=time_ns();for _ in 1:rep;launch!(path,dst[path],src,N);end;CUDA.synchronize();push!(times,(time_ns()-st)*1e-9/rep);ga=Base.gc_num();gct+=(ga.total_time-gb.total_time)*1e-9;gcp+=ga.pause-gb.pause
  end
  got=vcat([Array(a) for a in dst]...);ok=got==EXPECTED;err=maximum(abs.(Int.(got).-Int.(EXPECTED)));out=vcat(collect(BITS),UInt32.(got));file=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"output_trial$(trial).bin");write(file,out)
  @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":%d,\"max_rel_error\":%s,\"timing_region\":\"sum4_FP32_to_FP8_encoding_paths_kernel_only\",\"jit_excluded\":true,\"internal_repeats\":%d,\"per_path_times_seconds\":[%s],\"gc_time_seconds_whole_region\":%.17g,\"gc_pauses\":%d,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"uint32\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,sum(times),ok ? "true" : "false",length(got),err,"null",rep,join(times,","),gct,gcp,file,length(out))
 end
end

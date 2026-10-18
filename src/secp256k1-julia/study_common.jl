using CUDA,Printf
const ST_INPUT=collect(reinterpret(UInt32,read(joinpath(@__DIR__,"fixture.bin"))))
const ST_GOLD=read(joinpath(@__DIR__,"expected_bytes.bin"))
function study_main(model,launch!)
 rep=isempty(ARGS) ? 3 : parse(Int,ARGS[1]);frames=[(CuArray(ST_INPUT[2+c*8192:1+(c+1)*8192]),CUDA.fill(UInt8(165),49)) for c in 0:2]
 for f in frames;launch!(f);CUDA.synchronize();end
 for trial in 1:5
  times=Float64[];got=UInt8[];gct=0.;gcp=0
  for f in frames
   CUDA.fill!(f[2],UInt8(165));CUDA.synchronize();gb=Base.gc_num();t=time_ns()
   for _ in 1:rep;launch!(f);end
   CUDA.synchronize();push!(times,(time_ns()-t)*1e-9/rep);ga=Base.gc_num();gct+=(ga.total_time-gb.total_time)*1e-9;gcp+=ga.pause-gb.pause;append!(got,Array(f[2]))
  end
  ok=got==ST_GOLD;ma=maximum(abs(Int(got[i])-Int(ST_GOLD[i])) for i in eachindex(got))
  file=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"output_trial$(trial).bin");write(file,vcat(ST_INPUT,UInt32.(got)))
  @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":%d,\"max_rel_error\":0,\"timing_region\":\"sum3_complete_512point_field_chains_kernel_only\",\"jit_excluded\":true,\"internal_repeats\":%d,\"per_path_times_seconds\":[%s],\"gc_time_seconds_whole_region\":%.17g,\"gc_pauses\":%d,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"uint32\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,sum(times),ok ? "true" : "false",length(got),ma,rep,join(times,","),gct,gcp,file,length(ST_INPUT)+length(got))
 end
end

include(joinpath(@__DIR__,"study_fixture.jl"))
function study(run!)
 z=CuArray(PACK_ZERO);o=CuArray(PACK_ONE);out=CUDA.fill(floatmax(Float32),67*67)
 run!(z,o,out);CUDA.synchronize()
 for trial in 1:5
  CUDA.fill!(out,floatmax(Float32));CUDA.synchronize();start=time_ns()
  for _ in 1:10;run!(z,o,out);end
  CUDA.synchronize();elapsed=(time_ns()-start)*1e-9/10;got=Float64.(Array(out));err=maximum(abs.(got.-ORACLE));ok=err<1e-3
  data=vcat(INPUT_PREFIX,got);path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"trial_$(trial).bin");open(io->write(io,data),path,"w")
  println("STUDY_TRIAL {\"trial\":",trial,",\"time_seconds\":",elapsed,",\"correct\":",ok,",\"checked_values\":",length(data),",\"max_abs_error\":",err,",\"timing_region\":\"ten_epi_kernel_mean\",\"jit_excluded\":true,\"internal_repeats\":10,\"output_artifact\":{\"path\":\"",path,"\",\"dtype\":\"float64\",\"count\":",length(data),",\"byte_order\":\"little\"}}")
  ok||Base.error("FullEpistasisScoreOracleMismatch")
 end
end

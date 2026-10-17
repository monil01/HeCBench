include(joinpath(@__DIR__,"study_fixture.jl"))
const STUDY_REPEATS=10
function study(run!)
 S=CuArray(STUDY_SRC);out=CUDA.zeros(Float32,1024);tmp=CUDA.zeros(Float32,1792)
 for f in STUDY_FOCALS,tiled in (false,true);run!(tiled,S,Int32(257),f,Int32(32),Int32(32),out,tmp);end;CUDA.synchronize()
 for trial in 1:5
  total=0.;got=Float32[];wanted=Float32[]
  for findex in 1:3,tiled in (false,true)
   f=STUDY_FOCALS[findex];CUDA.synchronize();t=time_ns();for repeat in 1:STUDY_REPEATS;run!(tiled,S,Int32(257),f,Int32(32),Int32(32),out,tmp);end;CUDA.synchronize();total+=(time_ns()-t)*1e-9/STUDY_REPEATS;append!(got,Array(out));append!(wanted,STUDY_EXPECTED[(findex-1)*1024+1:findex*1024])
  end
  maxerr=maximum(abs.(Float64.(got).-Float64.(wanted)));ok=all(isfinite,got)&&maxerr<1e-3;path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"trial_$(trial).bin");open(io->write(io,got),path,"w")
  println("STUDY_TRIAL {\"trial\":",trial,",\"time_seconds\":",total,",\"correct\":",ok,",\"checked_values\":6144,\"max_abs_error\":",maxerr,",\"timing_region\":\"sum_three_focals_base_tiled_sync_kernel_means\",\"jit_excluded\":true,\"internal_repeats\":10,\"output_artifact\":{\"path\":\"",path,"\",\"dtype\":\"float32\",\"count\":6144,\"byte_order\":\"little\"}}")
  ok||error("Fullsurfeldepth/geometricwitnessfailed")
 end
end

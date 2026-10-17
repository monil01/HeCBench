function study(run!)
 n=16384;repeat=10;input=Int32[mod(i,17)-8 for i in 0:n-1];dx=CuArray(input)
 sizes=collect(16 .<< (0:10));outputs=[CUDA.zeros(Int32,n÷s) for s in sizes];keys=[similar(v) for v in outputs];templates=CUDA.zeros(Int32,256)
 for (j,s) in enumerate(sizes);run!(dx,outputs[j],keys[j],Int32(s),n÷s,templates);end
 CUDA.synchronize()
 for trial in 1:5
  times=Float64[];values=Int32[];correct=true
  for (j,s) in enumerate(sizes)
   fill!(outputs[j],0);fill!(keys[j],-1);CUDA.synchronize();start=time_ns()
   for _ in 1:repeat;run!(dx,outputs[j],keys[j],Int32(s),n÷s,templates);end
   CUDA.synchronize();push!(times,(time_ns()-start)*1e-9/repeat)
   got=Array(outputs[j]);gotkeys=Array(keys[j]);wanted=Int32[sum(@view input[g*s+1:(g+1)*s]) for g in 0:n÷s-1];correct &=got==wanted&&gotkeys==Int32.(0:n÷s-1);append!(values,gotkeys);append!(values,got)
  end
  path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"trial_$(trial).bin");open(io->write(io,values),path,"w")
  println("STUDY_TRIAL {\"trial\":",trial,",\"time_seconds\":",sum(times),",\"correct\":",correct,",\"checked_values\":",length(values),",\"max_abs_error\":",correct ? 0 : 1,",\"timing_region\":\"sum_eleven_segment_size_operation_means\",\"jit_excluded\":true,\"internal_repeats\":",repeat,",\"subkernel_times_seconds\":",times,",\"output_artifact\":{\"path\":\"",path,"\",\"dtype\":\"int32\",\"count\":",length(values),",\"byte_order\":\"little\"}}")
  correct||error("Full segmented reduction CPU oracle failed")
 end
end

function study(run!)
 n=4*32*64;repeat=10;input=Float32[Float32(i)/Float32(n) for i in 0:n-1]
 x=CuArray(input);y=CUDA.zeros(Float32,n);template=CUDA.zeros(Float32,32);expected=Vector{Vector{Float32}}()
 nhwc=zeros(Float32,n);nchw=zeros(Float32,n)
 for outer in 0:255,c in 0:31;nhwc[outer*32+c+1]=input[outer*32+mod(c,4)*8+div(c,4)+1];end
 for batch in 0:3,c in 0:31,s in 0:63;nchw[(batch*32+c)*64+s+1]=input[(batch*32+mod(c,4)*8+div(c,4))*64+s+1];end
 push!(expected,nhwc,nchw)
 for phase in 1:2;run!(phase,x,y,template);end
 CUDA.synchronize()
 for trial in 1:5
  times=Float64[];values=Float32[];correct=true;maxerr=0.
  for phase in 1:2
   fill!(y,0f0);CUDA.synchronize();start=time_ns()
   for _ in 1:repeat;run!(phase,x,y,template);end
   CUDA.synchronize();push!(times,(time_ns()-start)*1e-9/repeat);got=Array(y);correct &=got==expected[phase];maxerr=max(maxerr,maximum(abs.(got.-expected[phase])));append!(values,got)
  end
  path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"trial_$(trial).bin");open(io->write(io,values),path,"w")
  println("STUDY_TRIAL {\"trial\":",trial,",\"time_seconds\":",sum(times),",\"correct\":",correct,",\"checked_values\":",length(values),",\"max_abs_error\":",maxerr,",\"timing_region\":\"sum_nhwc_nchw_synchronized_loop_means\",\"jit_excluded\":true,\"internal_repeats\":",repeat,",\"subkernel_times_seconds\":",times,",\"output_artifact\":{\"path\":\"",path,"\",\"dtype\":\"float32\",\"count\":",length(values),",\"byte_order\":\"little\"}}")
  correct||error("Permutation oracle failed")
 end
end

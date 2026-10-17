function study(run!)
 mm=64;nn=128;kk=128;repeat=10
 a=Float32[Float32(mod(i,31)-15)/128f0 for i in 0:mm*nn-1];b=Float32[Float32(mod(i,17)+1)/256f0 for i in 0:nn*kk-1]
 da=CuArray(a);db=CuArray(b);dc=CUDA.zeros(Float32,mm*kk)
 expected=Vector{Vector{Float32}}()
 for phase in 1:4
  power=Float32(phase);iv=1f0/power;out=zeros(Float32,mm*kk)
  for row in 1:mm,col in 1:kk
   sum=0f0
   for j in 1:nn;sum+=abs(a[(row-1)*nn+j]-b[(j-1)*kk+col])^power;end
   out[(row-1)*kk+col]=sum^iv
  end
  push!(expected,out);run!(da,db,dc,mm,nn,kk,power,iv)
 end
 CUDA.synchronize()
 for trial in 1:5
  times=Float64[];values=Float32[];maxerr=0.;correct=true
  for phase in 1:4
   fill!(dc,0f0);CUDA.synchronize();start=time_ns()
   for _ in 1:repeat;run!(da,db,dc,mm,nn,kk,Float32(phase),1f0/Float32(phase));end
   CUDA.synchronize();push!(times,(time_ns()-start)*1e-9/repeat);got=Array(dc);e=maximum(abs.(got.-expected[phase]));maxerr=max(maxerr,e);correct &=all(isfinite,got)&&e<=1f-5;append!(values,got)
  end
  path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"trial_$(trial).bin");open(io->write(io,values),path,"w")
  println("STUDY_TRIAL {\"trial\":",trial,",\"time_seconds\":",sum(times),",\"correct\":",correct,",\"checked_values\":",length(values),",\"max_abs_error\":",maxerr,",\"timing_region\":\"sum_four_p_synchronized_kernel_loop_means\",\"jit_excluded\":true,\"internal_repeats\":",repeat,",\"subkernel_times_seconds\":",times,",\"output_artifact\":{\"path\":\"",path,"\",\"dtype\":\"float32\",\"count\":",length(values),",\"byte_order\":\"little\"}}")
  correct||error("CPU distance oracle failed")
 end
end

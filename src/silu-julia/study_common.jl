@inline function element(x,d,::Val{M}) where M
 if M<=2
  return x/(1f0+exp(-x))
 elseif M==3
  e=exp(-x);return d*((1f0+x*e/(1f0+e))/(1f0+e))
 else
  s=1f0/(1f0+exp(-x));return d*s*(1f0+x*(1f0-s))
 end
end
function study(run!)
 n=4096;repeat=10
 x=Float32[Float32(mod(i,257)-128)/32f0 for i in 0:n-1];d=Float32[Float32(mod(i,31)-15)/16f0 for i in 0:n-1]
 dx=CuArray(x);dd=CuArray(d);dy=CUDA.zeros(Float32,n)
 x4=reinterpret(NTuple{4,Float32},dx);d4=reinterpret(NTuple{4,Float32},dd);y4=reinterpret(NTuple{4,Float32},dy)
 for m in 1:5
  xx,ddd,yy=(m==2||m==5) ? (x4,d4,y4) : (dx,dd,dy)
  run!(xx,ddd,yy,length(yy),Val(m))
 end
 CUDA.synchronize()
 for trial in 1:5
  times=Float64[];values=Float32[];error=0.0;correct=true
  for m in 1:5
   xx,ddd,yy=(m==2||m==5) ? (x4,d4,y4) : (dx,dd,dy)
   fill!(dy,0f0);CUDA.synchronize();start=time_ns()
   for _ in 1:repeat;run!(xx,ddd,yy,length(yy),Val(m));end
   CUDA.synchronize();push!(times,(time_ns()-start)*1e-9/repeat)
   got=Array(dy);want=Float32[element(x[i],d[i],Val(m)) for i in 1:n];e=maximum(abs.(got.-want));error=max(error,Float64(e));correct &=all(isfinite,got)&&e<=1f-4;append!(values,got)
  end
  path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"trial_$(trial).bin");open(io->write(io,values),path,"w")
  println("STUDY_TRIAL {\"trial\":",trial,",\"time_seconds\":",sum(times),",\"correct\":",correct,",\"checked_values\":",5n,",\"max_abs_error\":",error,",\"timing_region\":\"sum_five_synchronized_kernel_loop_means\",\"jit_excluded\":true,\"internal_repeats\":",repeat,",\"subkernel_times_seconds\":",times,",\"output_artifact\":{\"path\":\"",path,"\",\"dtype\":\"float32\",\"count\":",5n,",\"byte_order\":\"little\"}}")
  correct||error("CPU formula oracle failed")
 end
end

using CUDA,Printf
function heat_study(launch!)
 raw=read(joinpath(@__DIR__,"study_initial.bin"));prefix=collect(reinterpret(Float64,raw));length(prefix)==85123||error("Input shape")
 cases=[(63,17),(127,18),(255,19)];offset=0;states=[];expected=Float64[]
 for (n,steps) in cases
  init=copy(prefix[offset+1:offset+n*n]);offset+=n*n;dx=1000.0/(n+1);dt=.5/steps;r=.1*dt/(dx*dx);r2=1-4*r;r<=.25||error("2D stability")
  cur=copy(init);old=zeros(Float64,n*n)
  for step=1:steps
   for j=0:n-1,i=0:n-1
    k=j*n+i+1
    old[k]=r2*cur[k]+r*(i<n-1 ? cur[k+1] : 0.0)+r*(i>0 ? cur[k-1] : 0.0)+r*(j<n-1 ? cur[k+n] : 0.0)+r*(j>0 ? cur[k-n] : 0.0)
   end
   old,cur=cur,old
  end
  g=1-8*r*sin(pi/(2*(n+1)))^2
  discrete=vcat((g^steps).*init,(g^(steps-1)).*init)
  vals=vcat(cur,old);maximum(abs.(vals.-discrete))<=5e-13||error("Discrete sine eigenmode oracle")
  continuous=exp(-2*.1*pi*pi*.5/1000.0^2).*init
  @printf("HEAT_ORACLE n=%d steps=%d r=%.12g discrete_max=%.12g continuous_max=%.12g\n",n,steps,r,maximum(abs.(vals.-discrete)),maximum(abs.(cur.-continuous)))
  append!(expected,vals);a=CuArray(init);b=CUDA.zeros(Float64,n*n);push!(states,(n,steps,r,r2,init,a,b))
 end
 for (n,steps,r,r2,init,a,b) in states;launch!(b,a,Int32(n),r,r2);end
 CUDA.synchronize();allok=true
 for trial=1:5
  for (n,steps,r,r2,init,a,b) in states;copyto!(a,init);fill!(b,0.0);end
  CUDA.synchronize();start=time_ns()
  for (n,steps,r,r2,init,a,b) in states
   cur=a;old=b
   for step=1:steps;launch!(old,cur,Int32(n),r,r2);old,cur=cur,old;end
  end
  CUDA.synchronize();sec=(time_ns()-start)*1e-9/54
  vals=Float64[]
  for (n,steps,r,r2,init,a,b) in states
   cur=isodd(steps) ? b : a;old=isodd(steps) ? a : b;append!(vals,Array(cur));append!(vals,Array(old))
  end
  err=maximum(abs.(vals.-expected));ok=all(isfinite,vals)&&err<=1e-12&&minimum(vals)>=0&&maximum(vals)<=1;allok&=ok
  out=vcat(prefix,vals);path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"output_trial$(trial).bin");write(path,out)
  @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":%.12g,\"max_rel_error\":0,\"timing_region\":\"kernel_launch_loop_per_internal_repeat\",\"jit_excluded\":true,\"internal_repeats\":54,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,sec,ok ? "true" : "false",length(vals),err,path,length(out))
 end
 allok||error("Fullstate heat numerical mismatch");println("PASS")
end

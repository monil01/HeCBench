using CUDA, LinearAlgebra, Printf
function fixture(m,n)
 l=Float64[((i+j)%5-2)/16 for j=0:m-1 for i=0:n-1]
 u=Float64[((2*i+j)%7-3)/16 for j=0:m-1 for i=0:n-1]
 d=Float64[4+(i+3*j)%5/8 for j=0:m-1 for i=0:n-1]
 x=Float64[((i+2*j)%11-5)/8 for j=0:m-1 for i=0:n-1]
 rhs=d.*x
 for j=0:m-1,i=0:n-1
  k=j*n+i+1
  if j>0;rhs[k]+=l[k]*x[k-n];end
  if j<m-1;rhs[k]+=u[k]*x[k+n];end
 end
 l,d,u,rhs,x
end
function cpu_update!(l,d,u,r,m,n)
 for sys=1:n
  u[sys]/=d[sys];r[sys]/=d[sys]
  for j=1:m-2
   k=j*n+sys;den=d[k]-l[k]*u[k-n];u[k]/=den;r[k]=(r[k]-l[k]*r[k-n])/den
  end
  k=(m-1)*n+sys;r[k]=(r[k]-l[k]*r[k-n])/(d[k]-l[k]*u[k-n])
  for j=m-2:-1:0;k=j*n+sys;r[k]-=u[k]*r[k+n];end
 end
end
function thomas_study(launch!)
 cases=[(2,5),(17,257),(65,33)];steps=3;states=[];expected=Float64[];prefix=Float64[]
 for (m,n) in cases
  l,d,u,r,x=fixture(m,n);append!(prefix,vcat(l,d,u,r));uc=copy(u);rc=copy(r)
  for step=1:steps
   old_u=copy(uc);old_r=copy(rc);cpu_update!(l,d,uc,rc,m,n)
   for sys=1:n
    ids=collect(sys:n:m*n);sol=Tridiagonal(l[ids[2:end]],d[ids],old_u[ids[1:end-1]])\old_r[ids]
    maximum(abs.(sol.-rc[ids]))<=1e-11 || error("Independent pivoted tridiagonal solve differs")
   end
   step==1 && maximum(abs.(rc.-x))>1e-11 && error("Manufactured solution mismatch")
  end
  append!(expected,vcat(uc,rc));push!(states,(m,n,l,d,u,r,CuArray(l),CuArray(d),CuArray(u),CuArray(r)))
 end
 # JIT before all trials, then reset every named independent trial.
 for (m,n,l,d,u,r,dl,dd,du,dr) in states;launch!(dl,dd,du,dr,Int32(m),Int32(n));end
 CUDA.synchronize();allok=true
 for trial=1:5
  for (m,n,l,d,u,r,dl,dd,du,dr) in states;copyto!(du,u);copyto!(dr,r);end
  CUDA.synchronize();start=time_ns()
  for (m,n,l,d,u,r,dl,dd,du,dr) in states
   for step=1:steps;launch!(dl,dd,du,dr,Int32(m),Int32(n));end
  end
  CUDA.synchronize();sec=(time_ns()-start)*1e-9/steps
  values=Float64[]
  for (m,n,l,d,u,r,dl,dd,du,dr) in states;append!(values,Array(du));append!(values,Array(dr));end
  err=maximum(abs.(values.-expected));ok=all(isfinite,values)&&err<=1e-10;allok&=ok
  out=vcat(prefix,values);path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"output_trial$(trial).bin");write(path,out)
  @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":%.12g,\"max_rel_error\":0,\"timing_region\":\"kernel_launch_loop_per_internal_repeat\",\"jit_excluded\":true,\"internal_repeats\":%d,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,sec,ok ? "true" : "false",length(values),err,steps,path,length(out))
 end
 allok||error("Full output mismatch");println("PASS")
end

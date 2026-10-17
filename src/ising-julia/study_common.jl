using CUDA,Printf
const ISING_CASES=((18,34,.5f0,17),(32,66,1f0,18),(2,4,2f0,19))
function ising_random(n)
 u=Float32[Float32(((37*i+11)%65536)+.5)/65536f0 for i=0:n-1]
 u[1:3]=Float32[0,.5,1];return u
end
# Independent physical-grid oracle: no compressed-neighbor indexing.
function ising_physical(nx,ny,u,beta,steps)
 h=ny÷2;g=zeros(Int8,nx,ny)
 for i=0:nx-1,j=0:ny-1;g[i+1,j+1]=u[i*h+j÷2+1]<.5f0 ? -1 : 1;end
 for step=1:steps,color=0:1,i=0:nx-1,j=0:ny-1
  if (i+j)%2==color
   spin=Int(g[i+1,j+1]);nn=Int(g[mod(i-1,nx)+1,j+1])+g[mod(i+1,nx)+1,j+1]+g[i+1,mod(j-1,ny)+1]+g[i+1,mod(j+1,ny)+1]
   delta=2*spin*nn;threshold=exp(-Float64(beta)*delta)
   if Float64(u[i*h+j÷2+1])<threshold;g[i+1,j+1]=-spin;end
  end
 end
 b=Int8[g[i+1,2*j+(i%2)+1] for i=0:nx-1 for j=0:h-1]
 w=Int8[g[i+1,2*j+1-(i%2)+1] for i=0:nx-1 for j=0:h-1]
 mag=sum(Int,g);energy=-sum(Int(g[i+1,j+1])*(Int(g[mod(i+1,nx)+1,j+1])+g[i+1,mod(j+1,ny)+1]) for i=0:nx-1,j=0:ny-1)
 return vcat(b,w),Float32[mag,energy]
end
function ising_study(launch!)
 states=[];prefix=Float32[];allok=true
 for (nx,ny,alpha,sweeps) in ISING_CASES
  n=nx*(ny÷2);u=ising_random(n);beta=1f0/(alpha*2.26918531421f0);du=CuArray(u);b=CUDA.zeros(Int8,n);w=similar(b)
  append!(prefix,Float32[nx,ny,sweeps,alpha,beta]);append!(prefix,u)
  for kind=0:3;launch!(kind,b,w,du,beta,Int64(nx),Int64(ny÷2));end
  gold=[ising_physical(nx,ny,u,beta,k)[1] for k in (0,2,sweeps+2)];obs=ising_physical(nx,ny,u,beta,sweeps+2)[2]
  gaps=[abs(Float64(v)-exp(-Float64(beta)*de)) for v in u for de in (-8,-4,0,4,8) if Float64(v)!=exp(-Float64(beta)*de)]
  @printf("ISING_INPUT nx=%d ny=%d beta=%.12g fixed_thresholds=true min_nontie_decision_gap=%.12g exact_probability_one_endpoint=true\n",nx,ny,beta,minimum(gaps))
  push!(states,(nx,ny,sweeps,beta,du,b,w,gold,obs))
 end
 CUDA.synchronize()
 for trial=1:5
  captured=[Float32[] for _ in states]
  for (idx,(nx,ny,sweeps,beta,u,b,w,gold,obs)) in enumerate(states)
   launch!(0,b,w,u,beta,Int64(nx),Int64(ny÷2));launch!(1,b,w,u,beta,Int64(nx),Int64(ny÷2));CUDA.synchronize();append!(captured[idx],Float32.(vcat(Array(b),Array(w))))
   for _=1:2;launch!(2,b,w,u,beta,Int64(nx),Int64(ny÷2));launch!(3,b,w,u,beta,Int64(nx),Int64(ny÷2));end
   CUDA.synchronize();append!(captured[idx],Float32.(vcat(Array(b),Array(w))))
  end
  CUDA.synchronize();start=time_ns()
  for (nx,ny,sweeps,beta,u,b,w,_,_) in states, _=1:sweeps
   launch!(2,b,w,u,beta,Int64(nx),Int64(ny÷2));launch!(3,b,w,u,beta,Int64(nx),Int64(ny÷2))
  end
  CUDA.synchronize();sec=(time_ns()-start)*1e-9/54;vals=Float32[];obsvals=Float32[];ok=true
  for (idx,(nx,ny,sweeps,beta,u,b,w,gold,obs)) in enumerate(states)
   append!(captured[idx],Float32.(vcat(Array(b),Array(w))));ok&=captured[idx]==Float32.(vcat(gold...));append!(vals,captured[idx]);append!(obsvals,obs)
  end
  ok&=all(x->x==-1||x==1,vals);allok&=ok;out=vcat(prefix,vals,obsvals);path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"output_trial$(trial).bin");write(path,out)
  @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":8196,\"max_abs_error\":%d,\"max_rel_error\":0,\"timing_region\":\"kernel_launch_loop_per_internal_repeat\",\"jit_excluded\":true,\"internal_repeats\":54,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float32\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,sec,ok ? "true" : "false",ok ? 0 : 2,path,length(out))
 end
 allok||error("Independent physical-grid Ising oracle mismatch");println("PASS")
end

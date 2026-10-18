using Printf
include("study_cases.jl")
const G=16;const GUARD=12345.5f0;const EPS=Float32(1e-6);const R=10
function run_study(launch)
 inputs=collect(reinterpret(Float32,read(joinpath(@__DIR__,"inputs_f32.bin"))));gold=collect(reinterpret(Float64,read(joinpath(@__DIR__,"oracle_f64.bin"))));states=Any[];prefix=Float64[]
 for (n,rows,off,go) in CASES
  x=CuArray(vcat(fill(GUARD,G),inputs[off+1:off+n*rows],fill(GUARD,G)));mean=CUDA.fill(GUARD,rows+2G);rstd=CUDA.fill(GUARD,rows+2G)
  append!(prefix,Array(x));append!(prefix,Array(mean));append!(prefix,Array(rstd));push!(states,(x=x,mean=mean,rstd=rstd,n=n,rows=rows,go=go))
  launch(Int64(n),rows,EPS,view(x,G+1:G+n*rows),view(mean,G+1:G+rows),view(rstd,G+1:G+rows));CUDA.device_synchronize()
 end
 for trial in 1:5
  out=Float64[];times=Float64[];correct=true;maxerr=0.
  for a in states
   fill!(a.mean,GUARD);fill!(a.rstd,GUARD);CUDA.device_synchronize();start=time_ns()
   for _ in 1:R;launch(Int64(a.n),a.rows,EPS,view(a.x,G+1:G+a.n*a.rows),view(a.mean,G+1:G+a.rows),view(a.rstd,G+1:G+a.rows));end
   CUDA.device_synchronize();push!(times,(time_ns()-start)*1e-9/R)
   m=Array(a.mean);s=Array(a.rstd);x=Array(a.x);append!(out,m);append!(out,s);append!(out,x)
   correct &= all(==(GUARD),m[vcat(1:G,G+a.rows+1:G+a.rows+G)])&&all(==(GUARD),s[vcat(1:G,G+a.rows+1:G+a.rows+G)])
   for row in 1:a.rows
    gm=gold[a.go+2row-1];gs=gold[a.go+2row];em=abs(Float64(m[G+row])-gm);es=abs(Float64(s[G+row])-gs);maxerr=max(maxerr,em,es);correct &= isfinite(m[G+row])&&isfinite(s[G+row])&&em<=5e-4+2e-6*abs(gm)&&es<=1e-3+2e-6*abs(gs)
   end
  end
  file=tempname()*".bin";open(file,"w")do io;write(io,vcat(prefix,out));end
  @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"max_abs_error\":%.12g,\"max_rel_error\":null,\"jit_excluded\":true,\"timing_region\":\"sum_eight_Welford_launch_batches_plus_device_completion_per_repeat\",\"internal_repeats\":10,\"component_time_seconds\":[%s],\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":%d,\"shape\":[%d],\"byte_order\":\"little\"}}\n",trial,sum(times),string(correct),maxerr,join(times,","),file,length(prefix)+length(out),length(prefix)+length(out))
 end
end

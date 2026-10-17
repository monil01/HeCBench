using CUDA
include(joinpath(@__DIR__,"..","_portable_study","OwnedStudy.jl"))
function constrained_dtw(a,b,r)
 M=length(a);previous=fill(Inf,M+1);previous[1]=0.
 for i in 1:M
  current=fill(Inf,M+1)
  for j in max(1,i-r):min(M,i+r);current[j+1]=(a[i]-b[j])^2+min(previous[j+1],previous[j],current[j]);end
  previous=current
 end
 previous[end]
end
function keogh_study(factory)
 N=262144;M=256;n=N-M+1;r=8;data=Vector{Float32}(undef,N+2n+3M);open(joinpath(@__DIR__,"study_input_f32.bin"))do io;read!(io,data);end
 subject=data[1:N];avgs=data[N+1:N+n];stds=data[N+n+1:N+2n];lower=data[N+2n+1:N+2n+M];upper=data[N+2n+M+1:N+2n+2M];query=data[N+2n+2M+1:end]
 all(stds.>0)&&all(lower.<=upper)||error("invalidstats/envelope");expected=zeros(Float64,n)
 for start in 0:n-1
  residue=0.
  for i in 0:M-1
   value=(Float64(subject[start+i+1])-avgs[start+1])/stds[start+1];distance=max(Float64(lower[i+1])-value,0.,value-Float64(upper[i+1]));residue+=distance*distance
  end
  expected[start+1]=residue
 end
 starts=[0,1,17,127,1020,8191,131072,n-1];dtw=Float64[]
 for start in starts
  window=(Float64.(subject[start+1:start+M]).-avgs[start+1])./stds[start+1];cost=constrained_dtw(window,Float64.(query),r);push!(dtw,cost);expected[start+1]<=cost+1e-3||error("independentlowerboundtheorem")
 end
 ds=CuArray(subject);da=CuArray(avgs);dt=CuArray(stds);lo=CuArray(lower);up=CuArray(upper);out=CUDA.zeros(Float32,n);op=factory(ds,da,dt,out,lo,up,M,N)
 readback=()->begin
  got=Array(out);all(isfinite,got)&&all(got.>=0)||error("nonfinite/negativeLB")
  for (k,start)in enumerate(starts);Float64(got[start+1])<=dtw[k]+1e-2||error("GPUscoreexceedsDTWbound");end
  println("STUDY_DOMAIN_CHECKS radius8_DTWcosts=",dtw," starts=",starts," score_min=",minimum(got)," score_max=",maximum(got)," normalizedstats_and_envelope_preprocessing_outsideclock")
  got
 end
 OwnedStudy.run_trials!(op,()->fill!(out,0.f0),readback,expected,10;atol=1e-2,rtol=0.)
end

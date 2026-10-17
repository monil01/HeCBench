using CUDA,KernelAbstractions,Atomix
include(joinpath(@__DIR__,"..","fixture_nodes.jl"))
@kernel function lebesgue_kernel!(lmax,interp,xfun,x,n,nfun)
 j=@index(Global,Linear)
 if j<=nfun
  t=0.
  for i1 in 1:n
   offset=(i1-1)*nfun+j;interp[offset]=1.
   for i2 in 1:n
    if i1!=i2;interp[offset]=interp[offset]*(xfun[j]-x[i2])/(x[i1]-x[i2]);end
   end
   t+=abs(interp[offset])
  end
  Atomix.@atomic max(lmax[1],t)
 end
end
function launch(lmax,interp,xfun,x,n,nfun)
 lebesgue_kernel!(get_backend(x),256)(lmax,interp,xfun,x,n,nfun;ndrange=nfun)
end
function emit(trial,seconds,result,expected,repeat;atol=0.,rtol=0.,region="synchronized_kernel_loop_mean",case_times=Float64[])
 error=maximum(abs.(Float64.(result).-Float64.(expected)));correct=all(isfinite,result)&&all(abs.(Float64.(result).-Float64.(expected)).<=atol.+rtol.*abs.(Float64.(expected)))
 artifact=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"trial_$(trial).bin");open(artifact,"w") do io;write(io,result);end
 dtype=eltype(result)==Float64 ? "float64" : eltype(result)==Float32 ? "float32" : eltype(result)==Int32 ? "int32" : "uint32"
 println("STUDY_TRIAL {\"trial\":",trial,",\"time_seconds\":",seconds,",\"correct\":",correct,",\"checked_values\":",length(result),",\"max_abs_error\":",error,",\"timing_region\":\"",region,"\",\"jit_excluded\":true,\"internal_repeats\":",repeat,",\"case_time_seconds\":",case_times,",\"output_artifact\":{\"path\":\"",artifact,"\",\"dtype\":\"",dtype,"\",\"count\":",length(result),",\"byte_order\":\"little\"}}")
 correct||Base.error("numerical failure")
end
function reference_values(x,xfun)
 n=length(x);nfun=length(xfun);values=zeros(Float64,n*nfun);totals=zeros(Float64,nfun)
 for j in 1:nfun
  for i1 in 1:n
   v=1.
   for i2 in 1:n
    if i1!=i2;v=v*(xfun[j]-x[i2])/(x[i1]-x[i2]);end
   end
   values[(i1-1)*nfun+j]=v;totals[j]+=abs(v)
  end
 end
 return vcat(values,maximum(totals))
end
function main(args)
 nfun=length(args)>0 ? parse(Int,args[1]) : 65;repeat=length(args)>1 ? parse(Int,args[2]) : 3
 xfun=Float64[-1+2*i/(nfun-1) for i in 0:nfun-1];df=CuArray(xfun);xs=[CuArray(x) for x in NODE_SETS];interps=[CUDA.zeros(Float64,length(x)*nfun) for x in NODE_SETS];maxima=[CUDA.zeros(Float64,1) for x in NODE_SETS];expected=reduce(vcat,[reference_values(x,xfun) for x in NODE_SETS])
 for k in eachindex(xs);launch(maxima[k],interps[k],df,xs[k],length(NODE_SETS[k]),nfun);end;CUDA.synchronize()
 for trial in 1:5
  result=Float64[];times=Float64[]
  for k in eachindex(xs)
   x=xs[k];interp=interps[k];lmax=maxima[k];n=length(NODE_SETS[k]);fill!(lmax,0.);CUDA.synchronize();start=time_ns()
   for _ in 1:repeat;launch(lmax,interp,df,x,n,nfun);end
   CUDA.synchronize();push!(times,(time_ns()-start)*1e-9/repeat);append!(result,Array(interp));append!(result,Array(lmax))
  end
  emit(trial,sum(times),result,expected,repeat;atol=1e-10,rtol=1e-12,case_times=times)
 end
end
main(ARGS)

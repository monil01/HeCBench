using CUDA,KernelAbstractions
include(joinpath(@__DIR__,"..","study_common.jl"))
@kernel function kernel!(x,d,y,n,::Val{M}) where M
 i=@index(Global,Linear)
 if i<=n
  @inbounds xx=x[i]
  @inbounds dd=d[i]
  if M==2||M==5
   @inbounds y[i]=ntuple(j->element(xx[j],dd[j],Val(M)),Val(4))
  else
   @inbounds y[i]=element(xx,dd,Val(M))
  end
 end
end
run!(x,d,y,n,m)=(kernel!(get_backend(y),256)(x,d,y,Int32(n),m;ndrange=n);nothing)
study(run!)

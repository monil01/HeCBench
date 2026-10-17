using CUDA
include(joinpath(@__DIR__,"..","study_common.jl"))
function kernel!(x,d,y,n,::Val{M}) where M
 i=(blockIdx().x-1)*blockDim().x+threadIdx().x
 if i<=n
  @inbounds xx=x[i]
  @inbounds dd=d[i]
  if M==2||M==5
   @inbounds y[i]=ntuple(j->element(xx[j],dd[j],Val(M)),Val(4))
  else
   @inbounds y[i]=element(xx,dd,Val(M))
  end
 end
 return
end
run!(x,d,y,n,m)=(@cuda threads=256 blocks=cld(n,256) kernel!(x,d,y,Int32(n),m);nothing)
study(run!)

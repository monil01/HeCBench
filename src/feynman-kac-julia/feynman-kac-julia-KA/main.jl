using CUDA,KernelAbstractions
import Atomix
include(joinpath(@__DIR__,"..","study_common.jl"))
@kernel function kernel!(ni,nj,seed,paths,a,b,h,rth,count,err)
 i,j=@index(Global,NTuple)
 if i<=ni && j<=nj
  n,e=point(i,j,ni,nj,seed,paths,a,b,h,rth)
  if n!=0
   Atomix.@atomic count[1]+=n
   Atomix.@atomic err[1]+=e
  end
 end
end
run!(args...)=kernel!(get_backend(args[9]),(16,16))(args...;ndrange=(NI,NJ))
study(run!)

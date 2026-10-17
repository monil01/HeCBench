using CUDA
import JACC
JACC.@init_backend
include(joinpath(@__DIR__,"..","study_common.jl"))
function kernel!(i,x,d,y,n,::Val{M}) where M
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
run!(x,d,y,n,m)=(JACC.parallel_for(JACC.launch_spec(threads=256,blocks=cld(n,256),shmem_size=0,sync=false),n,kernel!,x,d,y,Int32(n),m);nothing)
study(run!)
